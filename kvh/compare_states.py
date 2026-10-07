"""EXPERIMENTAL (2026-10-07): match a Strata session dump (kvh/strata_dump.cpp) against a Sushi cache entry
for the SAME prompt, tensor by tensor, to pin down the layout mapping the converter needs.

  uv run --with mlx --with numpy python kvh/compare_states.py STRATA_DUMP_DIR SUSHI_ENTRY_DIR

For each candidate layout it prints the cosine similarity; the right one should be near 1 (both engines computed
the same model at different precision), the wrong ones near 0.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import sushi_entry as se  # noqa: E402

import mlx.core as mx  # noqa: E402

N_GDN_STATE, N_V_HEADS, CONV_CH, D_CONV = 128, 48, 10240, 4


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def strata_kv(dump, meta, l, which):
    """Dequantize Strata's int8 K or V of KV layer l to float32 [heads, cells, head_dim]."""
    m = meta["kv"][l]
    H, D, P = m["heads"], m["head_dim"], m["page_size"]
    codes = np.fromfile(os.path.join(dump, f"kv{l}_{which}.bin"), dtype=np.int8)
    scales = np.fromfile(os.path.join(dump, f"kv{l}_{which}s.bin"), dtype=np.float16).astype(np.float32)
    pages = codes.size // (H * P * D)
    c = codes.reshape(pages, H, P, D).astype(np.float32)
    s = scales.reshape(pages, H, P, D // 64)
    x = (c.reshape(pages, H, P, D // 64, 64) * s[..., None]).reshape(pages, H, P, D)
    return x.transpose(1, 0, 2, 3).reshape(H, pages * P, D)


def main(dump, entry):
    meta = json.load(open(os.path.join(dump, "meta.json")))
    smeta = json.load(open(os.path.join(entry, "meta.json")))
    n_live = meta["live"]["tokens"]
    s_ids = np.fromfile(os.path.join(entry, "tokens.bin"), dtype=np.uint32)
    t_ids = np.fromfile(os.path.join(dump, "live_ids.i32"), dtype=np.int32)
    n = min(len(t_ids), len(s_ids))
    same = int(np.argmax(t_ids[:n] != s_ids[:n])) if (t_ids[:n] != s_ids[:n]).any() else n
    print(f"tokens: strata live {n_live}, sushi {len(s_ids)}, identical prefix {same}")
    print("strata kv layers:", [(m["cells"], m["heads"], m["head_dim"], m["page_size"], m["pooled_rows"]) for m in meta["kv"]][:3], "...")

    # K/V: position-exact, so compare row by row over the shared prefix
    N = min(same, smeta["kv_len"])
    for l, L in enumerate(se.ATTN_LAYERS):
        out = []
        for w in ("k", "v"):
            a = strata_kv(dump, meta, l, w)[:, :N]
            b = se.kv_full(entry, L, w)[:, :N]
            out.append(f"{w} cos {cos(a, b):.4f} (rot dims {cos(a[..., :64], b[..., :64]):.4f}, "
                       f"head-swapped {cos(a[::-1], b):.4f})")
        print(f"  attn l{L} vs strata kv{l}: " + "; ".join(out))

    # pooled indexer keys
    for l, L in list(enumerate(se.ATTN_LAYERS))[:3]:
        p = np.fromfile(os.path.join(dump, f"kv{l}_pooled.bin"), dtype=np.float32).reshape(-1, 128)
        qa, _ = se.load(os.path.join(entry, "qsa.safetensors"))
        q = np.array(qa[f"l{L}.pooled"].astype(mx.float32))[0]
        r = min(len(p), len(q))
        print(f"  pooled l{L}: strata {p.shape} sushi {q.shape} cos {cos(p[:r], q[:r]):.4f}")

    # GDN running state: Strata's live state is at n_live; pick the Sushi checkpoint nearest to it
    # STATE=cp0 compares Strata's checkpoint instead of its live state; SUSHI_POS picks the Sushi checkpoint
    tag = os.environ.get("STATE", "live")
    n_at = meta["live"]["tokens"] if tag == "live" else meta["checkpoints"][int(tag[2:])]["tokens"]
    pos = int(os.environ.get("SUSHI_POS") or min((c["pos"] for c in smeta["ssm"]), key=lambda p: abs(p - n_at)))
    sa, _ = se.load(os.path.join(entry, f"s{pos:07d}.safetensors"))
    print(f"GDN: strata {tag} @{n_at} vs sushi checkpoint @{pos} (exact only if equal)")
    g = np.fromfile(os.path.join(dump, f"{tag}_gdn.f32"), dtype=np.float32)
    rec, conv = N_GDN_STATE * N_V_HEADS * N_GDN_STATE, CONV_CH * (D_CONV - 1)
    rows = g.reshape(-1, rec + conv)
    gdn_layers = sorted(int(k[1:].split(".")[0]) for k in sa if k.endswith(".ssm"))
    print(f"  strata gdn rows {rows.shape[0]}, sushi gdn layers {len(gdn_layers)}")
    for i in (0, 1, len(gdn_layers) - 1):
        L = gdn_layers[i]
        ss = np.array(sa[f"l{L}.ssm"].astype(mx.float32))[0]          # [48,128,128]
        sc = np.array(sa[f"l{L}.conv"].astype(mx.float32))[0]         # [3,10240]
        for order in ("rec,conv", "conv,rec"):
            r = rows[i, :rec] if order == "rec,conv" else rows[i, conv:]
            cv = rows[i, rec:] if order == "rec,conv" else rows[i, :conv]
            r3 = r.reshape(N_GDN_STATE, N_V_HEADS, N_GDN_STATE)
            cands = {"(a,h,b)->h,a,b": r3.transpose(1, 0, 2), "(a,h,b)->h,b,a": r3.transpose(1, 2, 0),
                     "(h,a,b)": r.reshape(N_V_HEADS, N_GDN_STATE, N_GDN_STATE),
                     "(h,a,b)^T": r.reshape(N_V_HEADS, N_GDN_STATE, N_GDN_STATE).transpose(0, 2, 1)}
            best = max(cands, key=lambda k: cos(cands[k], ss))
            ccands = {"(ch,t)->t,ch": cv.reshape(CONV_CH, D_CONV - 1).T, "(t,ch)": cv.reshape(D_CONV - 1, CONV_CH)}
            cbest = max(ccands, key=lambda k: cos(ccands[k], sc))
            print(f"  l{L} [{order}] ssm best {best} cos {cos(cands[best], ss):.4f} | "
                  f"conv best {cbest} cos {cos(ccands[cbest], sc):.4f}")

    # PLE conv state and indexer tails
    ple = np.fromfile(os.path.join(dump, f"{tag}_ple.f32"), dtype=np.float32)
    sp = np.array(sa["l1.aux"].astype(mx.float32))[0]                 # [9,10240]
    print(f"PLE: strata {ple.size} floats, sushi {sp.shape}; cos as [9,10240] {cos(ple.reshape(sp.shape), sp):.4f}"
          if ple.size == sp.size else f"PLE: strata {ple.size} floats vs sushi {sp.size}: sizes differ")
    print("indexer: block_pos", np.fromfile(os.path.join(dump, f"{tag}_block_pos.i32"), dtype=np.int32)[:4],
          "tails floats", np.fromfile(os.path.join(dump, f"{tag}_tails.f32"), dtype=np.float32).size,
          "dead floats", np.fromfile(os.path.join(dump, f"{tag}_dead.f32"), dtype=np.float32).size)
    print("checkpoints:", meta["checkpoints"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
