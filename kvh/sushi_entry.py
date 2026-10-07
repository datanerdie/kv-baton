"""EXPERIMENTAL (2026-10-07): read and write Sushi 1.1.1 prefix-cache entries (manifest v8, qwen4_exp).

Run with MLX available, e.g. `uv run --with mlx --with numpy python kvh/sushi_entry.py ...`: Sushi writes
its 8-bit KV with MLX's affine quantizer (group 64), so MLX is the exact reference for the packing.

Entry layout (`$HOME/.sushi/kv-cache/<fingerprint>/e<id>/`):
  meta.json   v8 manifest            tokens.bin  u32 LE token ids
  c%06d       per 1024-token chunk: l{L}.k/.v U32 [1,2,T,64] + .ks/.kb/.vs/.vb BF16 [1,2,T,4] (12 attention layers)
  s%07d       SSM checkpoint at a token position: l{i}.conv BF16 [1,3,10240], l{i}.ssm BF16 [1,48,128,128]
              (36 GDN layers), l1.aux BF16 [1,9,10240] (PLE conv state), l1.ple U32 [9], and when pos % 4 != 0
              l{L}.aux BF16 [1,pos%4,128] (the indexer's pending keys)
  qsa         l{L}.pooled BF16 [1,pos//4,128] + the same aux tensors, metadata qsa_rows = newest checkpoint

  sushi_entry.py roundtrip ENTRY_DIR     re-quantize the dequantized KV and check it reproduces Sushi's bytes
  sushi_entry.py summary ENTRY_DIR       shapes and value ranges
"""
import json
import os
import sys

import mlx.core as mx
import numpy as np

GROUP, BITS = 64, 8
ATTN_LAYERS = list(range(3, 48, 4))           # l3, l7, ... l47


def load(path):
    arrays, meta = mx.load(path, return_metadata=True)
    return arrays, meta


def chunk_files(entry):
    return sorted(f for f in os.listdir(entry) if f.startswith("c") and f.endswith(".safetensors"))


def dequant_kv(arrays, layer, which):
    q = arrays[f"l{layer}.{which}"]
    s = arrays[f"l{layer}.{which}s"]
    b = arrays[f"l{layer}.{which}b"]
    return mx.dequantize(q, s, b, group_size=GROUP, bits=BITS)          # [1,2,T,256]


def quant_kv(x):
    q, s, b = mx.quantize(x, group_size=GROUP, bits=BITS)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def kv_full(entry, layer, which):
    """All chunks of one layer's K or V, dequantized, as float32 numpy [2, T, 256]."""
    parts = []
    for f in chunk_files(entry):
        a, _ = load(os.path.join(entry, f))
        parts.append(np.array(dequant_kv(a, layer, which).astype(mx.float32))[0])
    return np.concatenate(parts, axis=1)


def roundtrip(entry):
    bad = 0
    for f in chunk_files(entry)[:2]:
        a, _ = load(os.path.join(entry, f))
        for L in ATTN_LAYERS:
            for w in ("k", "v"):
                x = dequant_kv(a, L, w)
                q, s, b = quant_kv(x)
                same_q = bool(mx.array_equal(q, a[f"l{L}.{w}"]).item())
                same_s = bool(mx.array_equal(s, a[f"l{L}.{w}s"]).item())
                same_b = bool(mx.array_equal(b, a[f"l{L}.{w}b"]).item())
                if not (same_q and same_s and same_b):
                    bad += 1
                    dq = np.abs(np.array(q, dtype=np.int64) - np.array(a[f"l{L}.{w}"], dtype=np.int64))
                    print(f"{f} l{L}.{w}: codes equal {same_q} ({(dq != 0).mean():.4%} words differ), "
                          f"scales {same_s}, biases {same_b}")
    print("roundtrip:", "byte-identical" if bad == 0 else f"{bad} tensors differ")


def summary(entry):
    meta = json.load(open(os.path.join(entry, "meta.json")))
    print({k: meta[k] for k in ("kv_len", "tokens", "chunk_tokens")}, "ssm", [s["pos"] for s in meta.get("ssm", [])])
    k = kv_full(entry, 3, "k")
    print("l3.k", k.shape, "absmax", float(np.abs(k).max()), "mean|x|", float(np.abs(k).mean()))
    # RoPE touches only the first 64 of 256 dims (partial_rotary_factor 0.25): compare dims 0..63 vs 64..255
    print("  per-dim std, rotated part", float(k[..., :64].std()), "unrotated", float(k[..., 64:].std()))
    last = sorted(f for f in os.listdir(entry) if f.startswith("s"))[-1]
    a, m = load(os.path.join(entry, last))
    print(last, m.get("qsa_rows"), {n: (str(a[n].dtype), a[n].shape) for n in ("l0.conv", "l0.ssm", "l1.aux", "l1.ple")})
    print("  l1.ple", np.array(a["l1.ple"]).tolist())


if __name__ == "__main__":
    {"roundtrip": roundtrip, "summary": summary}[sys.argv[1]](sys.argv[2])
