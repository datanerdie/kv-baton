"""EXPERIMENTAL (2026-10-07): does Sushi answer as well from a Strata-prefilled state as from its own cold prefill?

Nine prompts over one ~100K-token haystack with eight needles (/Users/Shared/longctx100k/doc-100k-cut.txt, built by
qwen/longctx_build.py): q0 lists every locker code, q1..q8 ask for one city's code. Greedy, thinking off.

  needle_test.py strata PORT OUT_DIR          on the GPU box: prefill each prompt on Strata, save sessions/q<i>.bin
  needle_test.py sushi PORT ARM OUT_DIR       on the Mac: ask each prompt, write OUT_DIR/ARM.jsonl (answers + logprobs)
  needle_test.py score OUT_DIR                score every ARM.jsonl and compare arm "s" with arm "cold"
"""
import json
import math
import os
import sys
import time
import urllib.request

DOC = os.environ.get("NEEDLE_DOC", "/Users/Shared/longctx100k/doc-100k-cut.txt")
NEEDLES = os.environ.get("NEEDLE_JSON", "/Users/Shared/longctx100k/needles.json")


def prompts():
    doc, needles = open(DOC).read(), json.load(open(NEEDLES))
    out = [("q0", doc + "\n\nList every locker access code that appears in the text above, one per line, formatted "
                        "as '<City>: <code>'. Include only codes that actually appear in the text.")]
    for i, n in enumerate(needles, 1):
        out.append((f"q{i}", doc + f"\n\nWhat is the access code for the {n['city']} locker? "
                                   "Answer with the code only."))
    return out, needles


def post(port, path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def strata(port, out):
    ps, _ = prompts()
    for name, text in ps:
        t0 = time.time()
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": text}],
                 "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        dt = time.time() - t0
        s = post(port, "/slots/0?action=save", {"filename": f"{name}.bin"})
        u = r["usage"]
        print(f"{name}: prompt {u['prompt_tokens']} cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens')}"
              f" wall {dt:.1f}s save {s['timings']['save_ms']:.0f} ms", flush=True)


def sushi(port, arm, out):
    ps, _ = prompts()
    with open(os.path.join(out, f"{arm}.jsonl"), "w") as f:
        for name, text in ps:
            t0 = time.time()
            r = post(port, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": text}],
                     "max_tokens": 300, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                     "chat_template_kwargs": {"enable_thinking": False}})
            ch, u = r["choices"][0], r.get("usage", {})
            lp = [{"t": c["token"], "lp": c["logprob"], "top": {t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}}
                  for c in (ch.get("logprobs") or {}).get("content") or []]
            rec = {"q": name, "wall_s": round(time.time() - t0, 1), "prompt_tokens": u.get("prompt_tokens"),
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "text": ch["message"]["content"], "lp": lp}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{arm} {name}: cached {rec['cached']}/{rec['prompt_tokens']} wall {rec['wall_s']}s "
                  f"-> {rec['text'].strip()[:80]!r}", flush=True)


def kl_top5(a, b):
    floor = min(b.values()) if b else -30.0
    return sum(math.exp(la) * (la - b.get(t, floor)) for t, la in a.items())


def score(out):
    _, needles = prompts()
    arms = sorted(f[:-6] for f in os.listdir(out) if f.endswith(".jsonl"))
    res = {}
    for arm in arms:
        rows = {r["q"]: r for r in map(json.loads, open(os.path.join(out, f"{arm}.jsonl")))}
        q0 = rows["q0"]["text"]
        found = [n["city"] for n in needles if n["code"] in q0]
        singles = [rows[f"q{i}"]["text"].strip() for i in range(1, len(needles) + 1)]
        exact = [n["city"] for n, a in zip(needles, singles) if n["code"] in a]
        res[arm] = rows
        print(f"{arm:6} q0 listed {len(found)}/8 (missing {[n['city'] for n in needles if n['city'] not in found]}) | "
              f"q1-q8 exact {len(exact)}/8 (wrong {[ (n['city'], a[:20]) for n, a in zip(needles, singles) if n['code'] not in a]})")
    if "s" in res and "cold" in res:
        for q in sorted(res["cold"], key=lambda k: int(k[1:])):
            a, b = res["cold"][q]["lp"], res["s"][q]["lp"]
            n = min(len(a), len(b))
            same = next((i for i in range(n) if a[i]["t"] != b[i]["t"]), n)
            kl = [kl_top5(a[i]["top"], b[i]["top"]) for i in range(same)]
            print(f"  {q}: identical first {same}/{n} tokens, mean KL5 {sum(kl) / max(len(kl), 1):.5f}, "
                  f"same text {res['cold'][q]['text'].strip() == res['s'][q]['text'].strip()}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "strata":
        strata(sys.argv[2], sys.argv[3])
    elif mode == "sushi":
        sushi(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        score(sys.argv[2])
