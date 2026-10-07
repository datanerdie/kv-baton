"""THROWAWAY spike client (2026-10-07): greedy 256-token decode with top-5 logprobs.

  client.py PORT ARM OUTDIR        -> OUTDIR/ARM.jsonl, one line per prompt
  client.py compare OUTDIR         -> table vs arm r0 (4bpw cold)

Prompts are cut from /Users/Shared/longctx/doc-400k.txt (Gutenberg, ~4.1 chars/token).
"""
import json
import math
import sys
import time
import urllib.request

DOC = "/Users/Shared/longctx/doc-400k.txt"
SIZES = {"p8k": 33_000, "p32k": 131_000, "p100k": 411_000}
Q = ("\n\nIn about 200 words, summarise what happens in the last part of the text above, "
     "naming the people involved.")


def prompts():
    doc = open(DOC).read()
    return {k: doc[:n] + Q for k, n in SIZES.items()}


def run(port, arm, out):
    with open(f"{out}/{arm}.jsonl", "w") as f:
        for name, text in prompts().items():
            body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 256,
                    "temperature": 0, "logprobs": True, "top_logprobs": 5,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            dt = time.time() - t0
            ch = r["choices"][0]
            lp = [{"t": c["token"], "lp": c["logprob"],
                   "top": {t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}}
                  for c in (ch.get("logprobs") or {}).get("content") or []]
            u = r.get("usage", {})
            rec = {"prompt": name, "wall_s": round(dt, 1), "prompt_tokens": u.get("prompt_tokens"),
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "text": ch["message"]["content"], "lp": lp}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{arm} {name}: prompt {rec['prompt_tokens']} cached {rec['cached']} wall {dt:.1f}s "
                  f"gen {len(lp)}", flush=True)


def kl_top5(a, b):
    """KL(a||b) over a's top-5, b's missing entries floored at b's 5th logprob."""
    floor = min(b.values()) if b else -30.0
    s = 0.0
    for t, la in a.items():
        s += math.exp(la) * (la - b.get(t, floor))
    return s


def compare(out):
    load = lambda arm: {r["prompt"]: r for r in map(json.loads, open(f"{out}/{arm}.jsonl"))}
    ref = load("r0")
    print(f"{'arm':6} {'prompt':6} {'cached':>8} {'wall':>6} {'same_prefix':>11} {'top1_agree':>10} "
          f"{'mean|dlp|':>9} {'meanKL5':>8}")
    for arm in ("r1", "f", "x", "s"):
        try:
            cur = load(arm)
        except FileNotFoundError:
            continue
        for name, r in cur.items():
            a, b = ref[name]["lp"], r["lp"]
            n = min(len(a), len(b))
            same = next((i for i in range(n) if a[i]["t"] != b[i]["t"]), n)
            # positions before the first divergence share their context, so they compare like for like
            dl = [abs(a[i]["lp"] - b[i]["lp"]) for i in range(same)]
            kl = [kl_top5(a[i]["top"], b[i]["top"]) for i in range(same)]
            agree = sum(1 for i in range(same)
                        if a[i]["top"] and b[i]["top"]
                        and max(a[i]["top"], key=a[i]["top"].get) == max(b[i]["top"], key=b[i]["top"].get))
            print(f"{arm:6} {name:6} {r['cached']!s:>8} {r['wall_s']:>6} {same:>5}/{n:<5} "
                  f"{agree:>5}/{same:<4} {sum(dl)/max(len(dl),1):>9.4f} {sum(kl)/max(len(kl),1):>8.5f}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
