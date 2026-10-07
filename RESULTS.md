# Strata prefill on the GPU box -> Sushi decode on the Mac (2026-10-07)

Experimental. A prompt is prefilled by Strata v0.1.40.1 (UD-IQ4_XS, YaRN 4, int8 KV) on the GPU box's RTX 3090s, the session file is dumped (`strata_dump.cpp`), converted into a Sushi 1.1.1 cache entry (`strata_to_sushi.py`) and restored by Sushi-4bpw on the Mac, which then decodes. Nothing in production uses it.

## Quality, 100K-token haystack with 8 needles (`needle_test.py`)

| | q0: list all 8 | q1-q8: one each | replies identical to Sushi cold |
|---|---|---|---|
| Sushi-4bpw cold | 8/8 | 8/8 | - |
| handoff, Strata on one 3090 | 8/8 | 8/8 | 9/9, top-5 KL 0.00005-0.0008 |
| handoff, Strata on two 3090s (peer tier) | 8/8 | 8/8 | 9/9 |

Open-ended text drifts more (a 100K summary prompt: top-5 KL 0.040 before the first differing token, coherent and on-scene), so retrieval is preserved and free-text wording is not bit-stable.

## Speed, cold 100K prompt

| step | one 3090 | two 3090s (`--peer-device 1 --peer-reserve-mib 4608`) |
|---|---|---|
| Strata prefill | 68.1 s (1,476 t/s) | **44.0 s (2,283 t/s)** |
| save session (1.7 GB) | 1.0 s | 1.2 s |
| dump | 2.9 s | 2.9 s |
| copy GPU box -> Mac (761 MB/s) | 2.4 s | 2.4 s |
| convert | 5.6 s | 5.6 s |
| Sushi restore + first token | ~1 s | ~1 s |
| **total** | **~81 s** | **~57 s** |
| Sushi-4bpw cold prefill | 152.4 s | 152.4 s |

## Two GPUs needs a local Strata patch

Upstream Strata refuses session files with `--peer-device` (generate.cpp). `strata-peer-session.patch` lets a save through when `STRATA_ALLOW_PEER_SESSION=1`; it lives on branch `kvh-peer-session` in the GPU box `~/strata` and the binary runs from `~/kvh/strata-peer` (setup's `engine/strata` untouched). With the peer attached, the saved checkpoint state is bit-identical to the single-GPU save and the attention KV 99.99 % bit-identical. Without `--peer-reserve-mib` the peer's cache fills the card and its prompt buffers do not fit, so it does no prefill work (68.6 s).

## Known limits

The handoff proxy renders the prompt itself, so it can only hand off requests whose render matches what Sushi tokenises. Checked live 2026-10-07: with `chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "medium"}` Sushi prepends a system message ("Reasoning effort is set to low. Keep your thinking brief and focused, ...") that the pack chat template alone does not produce, so the proxy render differs from Sushi from token 1 onward and a handed-off cache entry would never be reused. Thinking requests with `reasoning_effort` therefore must pass through to Sushi untouched until the proxy reproduces that server-side system message. Text content arrays are joined with "\n" by Sushi (the renderer does the same); that case matches token for token.

## Production path (2026-10-07, through the gateway)

The same pipeline, automated (`handoff_proxy.py`), measured through the local gateway as `qwen38-flash-bigdoc` with the 100K needle document (100,526 tokens, a fresh prefix each run):

| run | first answer | needles | proxy steps (s) | Sushi restored |
|---|---|---|---|---|
| Document A | 74.0 s | 8/8 | prefill 43.7, save 1.2, dump+copy 7.1, convert 7.7, restart 11.7 | 100,519 of 100,526 |
| Document B | 72.7 s | 8/8 | prefill 43.8, save 1.0, dump+copy 7.1, convert 5.5, restart 11.7 | 100,519 of 100,526 |

A follow-up on `qwen38-flash` with the same document answered in 1.9 s (8/8) from the injected entry. During run B, short chats sent every 3 s were held across the restart (up to 11.1 s) and all answered. A KV round-trip and tool-calling check afterwards: KV round trip ok, tools 28/28. Sushi runs in its own process group after a handoff restart (it does not die with the proxy's launchd job).

## Without the Sushi restart (2026-10-07, patched Sushi in production)

Sushi 1.1.1 + `sushi-kvh-import.patch`: the converted entry is moved into the cache folder and `POST /v1/kvh/import` indexes it at the next lookup. Through the gateway, streaming, a fresh 100K needle document: first token **59.1 s** (was 72–74 s with the restart, ~152 s Sushi alone), 8/8; proxy steps prefill 43.7, save 1.0, dump+copy 7.2, convert 6.0, import 0.0. The same document on `qwen38-flash` afterwards: first token 0.3 s. `the KV round-trip and tool-calling check` on the patched build: both checks passed. `the restart command restart` itself is now 9.2 s (polling every 0.2 s), used only as the fallback.

## Streamed conversion (2026-10-07)

The handoff pipes `ssh gpu-box cat <session>` straight into `strata_to_sushi.py -`, which parses the session file as it arrives and quantises each KV layer on arrival (no dump on the GPU box, no separate copy). On a real 32K session the streamed and dump-folder paths give byte-identical entries. Live through the gateway, fresh 100K needle document: proxy steps prefill 43.6 s, save 1.0 s, **stream 3.5 s** (was dump+copy 7.2 + convert 6.0), import 0.0 s; first token **49.2 s** (59.1 s before, 72–74 s with the restart, ~152 s Sushi alone), 8/8.
