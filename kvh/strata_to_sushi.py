"""EXPERIMENTAL (2026-10-07): convert a Strata session dump (kvh/strata_dump.cpp) into a Sushi 1.1.1 prefix-cache
entry (manifest v8, qwen4_exp), so Sushi-4bpw can continue from a prompt that Strata prefilled on an NVIDIA card.

  uv run --with mlx --with numpy python kvh/strata_to_sushi.py DUMP_DIR OUT_ENTRY_DIR [--state cp0|live]

The restore point is Strata's checkpoint (`cp0`, a few tokens before the prompt end) by default: Sushi resumes at an
SSM checkpoint and recomputes the tokens after it, and Strata saves only the deepest one.

Mapping (verified 2026-10-07 against Sushi's own entry for the same 8,262-token prompt; cosine in brackets):
  K/V        Strata int8 [page][head][page_size][256] x fp16 scale/64  ->  Sushi affine 8-bit g64 [1,2,T,64]  (0.94-0.998)
             same YaRN factor 4 on both sides (Strata --rope-scaling yarn --rope-scale 4), so no re-rotation
  GDN ssm    Strata [k, h_s, v] fp32  ->  Sushi [h, v, k] bf16, h = (h_s % 16) * 3 + h_s // 16      (0.978-0.998)
             (Strata/llama.cpp pair v-head h with k-head h % 16; the reference pairs it with h // 3)
  GDN conv   Strata [channel][t] fp32 ->  Sushi [t][channel], channels q 2048 | k 2048 | v 48x128 permuted as above
  PLE        Strata [10240][9] -> Sushi l1.aux [9][10240]; l1.ple = [1, tok[p-2], tok[p-1], 0 x 6]          (0.998)
  indexer    Strata tails[:p % 4] -> Sushi l{L}.aux [1, p % 4, 128]; Strata pooled[:p // 4] -> l{L}.pooled  (0.999)
             Strata's `dead` (the pooled row being built) and `block_pos` have no Sushi counterpart.
Not converted: the MTP head's state (spec.safetensors). It only affects drafting, and Sushi declines a missing spec.
"""
import json
import queue
import struct
import os
import sys
import threading

import mlx.core as mx
import numpy as np

ATTN_LAYERS = list(range(3, 48, 4))
GDN_LAYERS = [l for l in range(48) if l not in ATTN_LAYERS]
S, HV, HK, CONV_CH, CHUNK = 128, 48, 16, 10240, 1024
S2H = np.array([(s % HK) * (HV // HK) + s // HK for s in range(HV)])   # Strata v-head -> reference v-head
H2S = np.argsort(S2H)                                                   # reference v-head -> Strata v-head


def strata_kv(dump, m, l, which, n):
    H, D, P = m["heads"], m["head_dim"], m["page_size"]
    codes = np.fromfile(os.path.join(dump, f"kv{l}_{which}.bin"), dtype=np.int8)
    scales = np.fromfile(os.path.join(dump, f"kv{l}_{which}s.bin"), dtype=np.float16).astype(np.float32)
    pages = codes.size // (H * P * D)
    x = codes.reshape(pages, H, P, D // 64, 64).astype(np.float32) * scales.reshape(pages, H, P, D // 64)[..., None]
    return x.reshape(pages, H, P, D).transpose(1, 0, 2, 3).reshape(H, pages * P, D)[:, :n]


def bf16(a):
    return mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(mx.bfloat16)


def save(path, arrays, metadata):
    mx.save_safetensors(path, arrays, metadata=metadata)
    return os.path.getsize(path)


class EntryWriter:
    """Builds one Sushi v8 entry. KV layers can arrive one at a time (each is quantized as it arrives, so a
    streamed session overlaps the network with the work); `finish` writes the files."""

    def __init__(self, out, ids, pos):
        n = len(ids)
        if pos >= n:
            raise ValueError(f"restore point {pos} must be below the prompt length {n}: Sushi always forwards a token")
        os.makedirs(out, exist_ok=False)
        self.out, self.ids, self.pos, self.n = out, np.asarray(ids, dtype=np.int32), pos, n
        self.kv, self.pooled, self.state = {}, {}, None

    def add_kv_layer(self, l, m, k_codes, k_scales, v_codes, v_scales, pooled):
        """Strata KV layer l (int8 codes + fp16 scales per 64 values, its kv meta m) -> 8-bit affine, kept quantized."""
        L = ATTN_LAYERS[l]
        H, D, P = m["heads"], m["head_dim"], m["page_size"]
        for w, codes, scales in (("k", k_codes, k_scales), ("v", v_codes, v_scales)):
            c = np.frombuffer(codes, dtype=np.int8)
            sc = np.frombuffer(scales, dtype=np.float16).astype(np.float32)
            pages = c.size // (H * P * D)
            x = c.reshape(pages, H, P, D // 64, 64).astype(np.float32) * sc.reshape(pages, H, P, D // 64)[..., None]
            x = x.reshape(pages, H, P, D).transpose(1, 0, 2, 3).reshape(H, pages * P, D)[:, : self.n]
            q, s_, b = mx.quantize(mx.array(np.ascontiguousarray(x[None])), group_size=64, bits=8)
            mx.eval(q, s_, b)
            self.kv[(L, w)] = (q, s_.astype(mx.bfloat16), b.astype(mx.bfloat16))
        self.pooled[L] = np.frombuffer(pooled, dtype=np.float32).reshape(-1, 128)[: self.pos // 4].copy()

    def set_state(self, gdn, ple, tails):
        """The running state AT the restore point (bytes of Strata's gdn / ple / tails, fp32)."""
        self.state = (np.frombuffer(gdn, dtype=np.float32).reshape(len(GDN_LAYERS), -1),
                      np.frombuffer(ple, dtype=np.float32).reshape(CONV_CH, 9).T,
                      np.frombuffer(tails, dtype=np.float32).reshape(len(ATTN_LAYERS), 3, 128))

    def finish(self):
        out, ids, n, pos = self.out, self.ids, self.n, self.pos
        missing = [L for L in ATTN_LAYERS if (L, "k") not in self.kv]
        if missing or self.state is None:
            raise ValueError(f"incomplete session: kv layers missing {missing}, state {'set' if self.state else 'missing'}")
        sizes = {}
        ids.astype(np.uint32).tofile(os.path.join(out, "tokens.bin"))
        chunk_bytes = []
        for c in range((n + CHUNK - 1) // CHUNK):
            a = {}
            for L in ATTN_LAYERS:
                for w in ("k", "v"):
                    q, s_, b = self.kv[(L, w)]
                    sl = slice(c * CHUNK, (c + 1) * CHUNK)
                    a[f"l{L}.{w}"], a[f"l{L}.{w}s"], a[f"l{L}.{w}b"] = q[:, :, sl], s_[:, :, sl], b[:, :, sl]
            chunk_bytes.append(save(os.path.join(out, f"c{c:06d}.safetensors"), a, {}))

        g, ple, tails = self.state
        aux = {"l1.aux": bf16(ple[None])}
        if pos % 4:
            for j, L in enumerate(ATTN_LAYERS):
                aux[f"l{L}.aux"] = bf16(tails[j, : pos % 4][None])
        st = dict(aux)
        st["l1.ple"] = mx.array(np.array([1, ids[pos - 2], ids[pos - 1], 0, 0, 0, 0, 0, 0], dtype=np.uint32))
        for i, L in enumerate(GDN_LAYERS):
            ssm = g[i, : S * HV * S].reshape(S, HV, S).transpose(1, 2, 0)[H2S]          # [h, v, k]
            cv = g[i, S * HV * S:].reshape(CONV_CH, 3)
            cv = np.concatenate([cv[:4096], cv[4096:].reshape(HV, S, 3)[H2S].reshape(-1, 3)]).T   # [t, ch]
            st[f"l{L}.ssm"], st[f"l{L}.conv"] = bf16(ssm[None]), bf16(cv[None])
        smeta = {"layers": "48", "init": ",".join(map(str, GDN_LAYERS)), "qsa_ratio": "4", "qsa_rows": str(pos)}
        sizes["ssm"] = save(os.path.join(out, f"s{pos:07d}.safetensors"), st, smeta)

        qs = dict(aux)
        for L in ATTN_LAYERS:
            qs[f"l{L}.pooled"] = bf16(self.pooled[L][None])
        sizes["qsa"] = save(os.path.join(out, "qsa.safetensors"), qs, {"qsa_rows": str(pos), "qsa_ratio": "4"})

        total = sum(chunk_bytes) + sizes["ssm"] + sizes["qsa"] + n * 4
        manifest = {"v": 8, "kv_len": n, "tokens": n, "has_tools": False, "scheme": "affine", "bits": 8, "group_size": 64,
                    "chunk_tokens": CHUNK, "inherited_chunks": 0, "bytes": total, "chunk_bytes": chunk_bytes,
                    "ssm": [{"pos": pos, "bytes": sizes["ssm"]}],
                    "qsa_history": {"bytes": sizes["qsa"], "rows": pos, "inherited": False}}
        json.dump(manifest, open(os.path.join(out, "meta.json"), "w"))
        print(f"converted {n} tokens, restore point {pos}, {total / 2**20:.1f} MiB -> {out}", flush=True)


def convert(dump, out, state="cp0"):
    """From a strata_dump folder (the original path; kept for comparison and tests)."""
    meta = json.load(open(os.path.join(dump, "meta.json")))
    ids = np.fromfile(os.path.join(dump, "live_ids.i32"), dtype=np.int32)
    pos = meta["live"]["tokens"] if state == "live" else meta["checkpoints"][int(state[2:])]["tokens"]
    w = EntryWriter(out, ids, pos)
    rd = lambda f: open(os.path.join(dump, f), "rb").read()
    for l in range(len(ATTN_LAYERS)):
        w.add_kv_layer(l, meta["kv"][l], rd(f"kv{l}_k.bin"), rd(f"kv{l}_ks.bin"), rd(f"kv{l}_v.bin"), rd(f"kv{l}_vs.bin"),
                       rd(f"kv{l}_pooled.bin"))
    w.set_state(rd(f"{state}_gdn.f32"), rd(f"{state}_ple.f32"), rd(f"{state}_tails.f32"))
    w.finish()


class Prefetch:
    """Reads a stream on its own thread into a bounded queue, so the network transfer keeps going while the main
    thread converts (measured on a 100K session: transfer 1.7 s and conversion 1.8 s ran back to back without it).
    read(n) returns up to n bytes, b"" at the end; an error on the reading side is raised here."""

    def __init__(self, f, block=16 << 20, depth=32):
        self.q, self.buf, self.off, self.done = queue.Queue(depth), b"", 0, False
        threading.Thread(target=self._run, args=(f, block), daemon=True).start()

    def _run(self, f, block):
        try:
            while True:
                b = f.read(block)
                self.q.put(b)
                if not b:
                    return
        except BaseException as e:               # handed to the consumer, which raises it
            self.q.put(e)

    def read(self, n):
        while self.off >= len(self.buf):
            if self.done:
                return b""
            item = self.q.get()
            if isinstance(item, BaseException):
                raise item
            if not item:
                self.done = True
                return b""
            self.buf, self.off = item, 0
        b = self.buf[self.off:self.off + n]
        self.off += len(b)
        return b


def convert_stream(f, out, state="cp0"):
    """Straight from a Strata session file (STRSESS v1) read as a stream, e.g. `ssh gpu-box cat FILE | ... -`.
    Layout (Strata src/core/conversation_file.cpp): 64-byte header; geometry (18 i64), layer_lo, layer_hi, cvec;
    the live checkpoint, a checkpoint count and the checkpoints; a KV layer count, then per layer 7 i64 (format,
    cells, heads, head_dim, page_size, pooled_rows, idx_dim) and 5 length-prefixed buffers (k, v, k_scale,
    v_scale, pooled) — the 12 attention layers, then the MTP draft layer (skipped); payload hash, b"STRSEND\\x01"."""
    got = [0]
    f = Prefetch(f)

    def read(n):
        parts, left = [], n
        while left:
            b = f.read(min(left, 16 << 20))
            if not b:
                raise ValueError(f"session stream ended early ({got[0]} bytes read, {left} more expected)")
            parts.append(b); left -= len(b); got[0] += len(b)
        return b"".join(parts)

    u64 = lambda: struct.unpack("<Q", read(8))[0]
    i64 = lambda: struct.unpack("<q", read(8))[0]

    def checkpoint():
        ids = np.frombuffer(read(u64() * 4), dtype=np.int32)
        for _ in range(u64()):
            read(16)
        c = {"ids": ids}
        for name in ("gdn", "ple", "tails", "dead", "block_pos"):
            c[name] = read(u64())
        u64()                                    # used
        return c

    header = read(64)
    if header[:8] != b"STRSESS\x01" or struct.unpack("<I", header[8:12])[0] != 1:
        raise ValueError("not a STRSESS v1 session file")
    payload = struct.unpack("<Q", header[32:40])[0]
    for _ in range(18):
        i64()
    i64(); i64(); u64()                          # layer range, cvec
    live = checkpoint()
    cps = [checkpoint() for _ in range(u64())]
    src = live if state == "live" else cps[int(state[2:])]
    w = EntryWriter(out, live["ids"], len(src["ids"]))
    w.set_state(src["gdn"], src["ple"], src["tails"])
    n_kv = u64()
    if n_kv < len(ATTN_LAYERS):
        raise ValueError(f"session has {n_kv} kv layers, need {len(ATTN_LAYERS)}")
    for l in range(n_kv):
        m = dict(zip(("format", "cells", "heads", "head_dim", "page_size", "pooled_rows", "idx_dim"), (i64() for _ in range(7))))
        bufs = [read(u64()) for _ in range(5)]
        if l < len(ATTN_LAYERS):
            w.add_kv_layer(l, m, bufs[0], bufs[2], bufs[1], bufs[3], bufs[4])
    read(8)                                      # payload hash (not recomputed: the length and end marker are checked)
    if read(8) != b"STRSEND\x01" or got[0] != 64 + payload + 16:
        raise ValueError("session stream: bad end marker or length")
    w.finish()


if __name__ == "__main__":
    args = sys.argv[1:]
    st = "cp0"
    if "--state" in args:
        i = args.index("--state"); st = args[i + 1]; del args[i:i + 2]
    if args[0] == "-" or args[0].endswith(".bin"):
        src = sys.stdin.buffer if args[0] == "-" else open(args[0], "rb")
        convert_stream(src, args[1], st)
    else:
        convert(args[0], args[1], st)
