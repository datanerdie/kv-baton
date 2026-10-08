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

The handoff proxy renders the prompt itself, so it can only hand off requests whose render matches what Sushi tokenises. Checked live 2026-10-07: with `chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "medium"}` Sushi's prompt starts with a system message ("Reasoning effort is set to low. Keep your thinking brief and focused, ...") that the proxy render does not have, so the two differ from token 1 onward and a handed-off cache entry would never be reused. Thinking requests therefore pass through to Sushi untouched. Text content arrays are joined with "\n" by Sushi (the renderer does the same); that case matches token for token.

**Correction (2026-10-07, later):** the message is not added by Sushi. The pack chat template writes it itself (medium = no message, low = a short one, xhigh = a long one; with no kwargs the template defaults to thinking on at xhigh). Sushi 1.1.1 reads `reasoning_effort` only as a top-level request field (`server.zig` `parseReasoningEffort`) and drops it from `chat_template_kwargs`, so `enable_thinking: true` there renders as low whatever effort was asked; top-level low / medium / xhigh render exactly as the template does (prompt_tokens 53 / 27 / 65 against the local render on a short probe), and "high" is refused with a 400. With no kwargs at all Sushi runs thinking off (29 tokens) while the template default is on (65). Thinking handoff is therefore a rendering fix (render with the effort Sushi will actually use), not a missing server-side message. The proxy's token check compares Strata with the proxy's render, not with Sushi, so a render that disagrees with Sushi is not caught: the entry is imported and simply never matched. The gateway's `qwen38-flash-think` alias had the same problem (medium asked, low run since 2026-10-06); it now sends `reasoning_effort: medium` at the top level too.

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

## Thinking requests hand off (2026-10-07, evening)

`render.py` now resolves thinking the way Sushi 1.1.1 does (`resolveEnableThinking`, `parseReasoningEffort`, `qwen38EffortFor`): `reasoning_effort` counts only at the top level (off / low / medium / xhigh / minimal / none; anything else is Sushi's 400, so the request passes through), `enable_thinking` comes from the top level or else `chat_template_kwargs`, thinking on without an effort word is low, and a request naming neither is thinking off. Strata follows the template instead (it honours the kwargs effort and defaults to thinking on at xhigh), so the proxy sends Strata the resolved `chat_template_kwargs` explicitly and drops the top-level fields; Sushi still gets the client's request unchanged. Before any prefill the proxy also sends the rendered text to Sushi's `/tokenize` (0.1 s at 100K) and passes through on any difference: until now nothing compared the render with Sushi itself. Still passed through: tools, images, a `reasoning` object, assistant messages carrying `reasoning_content`, and a trailing assistant message. `THINK_HANDOFF = False` in `handoff_proxy.py` is the kill switch.

Checked against live Sushi before going live: 12 request shapes (silent, kwargs on/off, kwargs effort, top-level low/medium/xhigh/minimal/none, off + kwargs on, top-level `enable_thinking`, `preserve_thinking`; multi-turn with U+202F) — Sushi's chat prompt_tokens equal `/tokenize` of the render in every case.

Live, 100K needle document (q0, all eight codes), straight to the proxy, fresh prefix each run:

| run | first token | total | needles | Sushi restored |
|---|---|---|---|---|
| handoff, top-level `reasoning_effort: medium` | 49.7 s | 55.1 s | 8/8 | 100,551 / 100,556 |
| handoff, kwargs `enable_thinking: true` (Sushi: low) | 50.1 s | 58.0 s | 8/8 | 100,578 / 100,583 |
| Sushi cold, medium | 153.1 s | 160.8 s | 8/8 | 0 |

Proxy steps were prefill 43.7–44.1 s, save 1.0, stream 3.9–4.0, import 0.0. All three answers are identical; the thinking text differs in length (902 / 1,611 chars for the medium handoff / cold), the same near-tie drift seen in free text before.

## Tool conversations hand off (2026-10-07, late)

Sushi renders tools straight from the request (the OpenAI `{"type": "function", "function": ...}` objects as sent) and parses JSON-string call arguments into mappings; a 10K agent conversation (two tools with nested schemas and non-ASCII descriptions, an assistant tool call, a 40K-character tool result) was token-identical to Sushi's own cache entry with thinking off, medium and kwargs-on, and Sushi flags such entries `has_tools: true`, which is part of its cache key. Strata does not render tools the same way: `serve/frontend.py` unwraps each tool to its function object before the template, so its prompt was 9–10 tokens short on every tools request, and it validates tool names before rendering, so no reshaping of the request gets it to Sushi's text. Local Strata patch `strata-kvh-prompt-ids.patch` (the GPU box branch `kvh-peer-session`, server-side Python only): an optional `kvh_prompt_ids` on `/v1/chat/completions` replaces the template render. The proxy now sends it on every handoff, with the ids that Sushi's `/tokenize` has just confirmed, so Strata's own rendering (tool unwrapping, effort-at-end, literal-tag and empty-turn handling) no longer matters. `render.py` allows tools, tool messages and a trailing tool result; `tool_choice` "none" drops the tools (as Sushi does), "required" or a named function pass through (Sushi adds its own instruction). The proxy's cache lookup only counts entries with the same `has_tools` flag, and the handoff marks the converted entry.

Live, an agent conversation whose `read_file` result is the 100K needle document (system, user request, assistant tool call, tool result; fresh file name each run), then a follow-up question:

| run | turn 1 first token | turn 1 | follow-up |
|---|---|---|---|
| handoff, thinking off | 50.6 s | another `read_file` call | 0.7 s, correct |
| Sushi cold, thinking off | 152.8 s | another `read_file` call | 0.7 s, correct |
| handoff via `qwen38-flash-bigdoc-think` (medium) | 49.9 s | 8/8 codes | 1.3 s, correct |

With thinking off the model calls `read_file` again in both arms, so that is the model on this conversation, not the handoff. Sushi restored 100,864 of 100,871 and 100,864 of 100,869 tokens after the two handoffs, and both follow-ups were served from the `has_tools` entry. A plain (no tools) 100K handoff through the new ids path: 50.1 s first token, 8/8.

## Overnight 2026-10-07/08: Sushi 1.2.0, quality runs, automatic handoff

**Sushi 1.2.0** (kvh build, in production since 2026-10-07 22:30). Cache format, lookup and the import path are unchanged; `render.py` follows the changes that matter: "minimal" is now a 400, `response_format` and `ignore_eos` pass through, a missing tool `description`/`parameters` is filled in as Sushi fills it (this one predates 1.2.0: a tool without a description never handed off before), and the proxy refuses to hand off when Sushi reports a default effort other than off. NLL on the quality fixtures is identical to 1.1.1 to the last digit (0.280633139 / 0.121696140).

**club-3090 8-pack** (`quality-test.sh --full`, pass@1 with pass@3 in brackets, 150 cases per arm):

| run | mixed | thinking on | thinking wall |
|---|---|---|---|
| Sushi 1.1.1, low effort (2026-10-07 morning) | 128 (133) | 128 (136) | 41 min |
| Sushi 1.2.0, low | 123 (127) | 130 (134) | 39 min |
| Sushi 1.2.0, xhigh (model card default) | 125 (126) | stopped: 99/110 on 7 packs vs 102 at low; cli-40 ~250 s per case | — |
| Sushi 1.2.0, medium | — | **131 (137)**, 0 token-limit or server errors | 40 min |
| Sushi 1.2.0, low, **every request handed off** | 130 (131) | 128 (136) | 48 min |

Sushi runs thinking at low when a request names no effort, so every earlier Sushi number was low effort. The only mixed-arm move from 1.1.1 to 1.2.0 is cli-40 (greedy, 1,024-token budget, single shot) 32 -> 27; with NLL identical this is near-tie variance in decode. Forced handoff: 752 handoffs, none failed, 665 decoded from Strata's state (the other 87 were short repeats Sushi served from its own cache, which it prefers unless the disk entry is >= 256 tokens longer); scores within noise of Sushi alone, so decoding from a Strata-made state costs nothing measurable across the eight packs.

**When a handoff pays** (first token, through a test proxy that always hands off, vs Sushi alone, thinking off):

| prompt | handoff | Sushi alone |
|---|---|---|
| ~5K | 5.1 s | 7.3 s |
| ~10K | 7.3 s | 13.9 s |
| ~20K | 11.2 s | 27.7 s |
| ~32K | 16.0 s | 45.3 s |

A handoff costs ~2.9 s plus Strata's prefill at ~2,300 t/s against Sushi's ~700 t/s: break-even near 3,000 new tokens. **Decode after a handoff** (32K prompt, 600 greedy tokens, four pairs): 53.2 / 57.8 / 57.4 / 52.0 t/s vs cold 58.0 / 57.1 / 57.6 / 55.2, about 3 % slower on average and no slow path of the kind seen elsewhere with mismatched KV dtypes; a likely cause is the MTP head's prompt state, which the converter does not carry over yet (Strata's session has it).

**Automatic handoff (live 2026-10-08 04:09):** the proxy now treats every chat request on the normal aliases as a candidate and hands off at 5,000 new tokens (`KVH_AUTO_MIN_NEW_TOKENS`); smaller requests are relayed with no log line or cache scan, and the one-handoff lock no longer holds other requests. The bigdoc aliases use the same threshold. Before each handoff the proxy now also checks Sushi's version (it must be one the render was checked against) and that Strata serves the expected model and engine, and each converted entry carries `kvh.json` (versions, model, render and converter hashes). Through the gateway: an 8,211-token prompt on `qwen38-flash` 6.0 s to first token, 7,731 tokens on `qwen38-flash-think` (medium) 5.6 s, a short chat 0.3 s; `the KV round-trip and tool-calling check` passed.

**Converter:** reads the session stream on its own thread while converting: 100K transfer + convert 3.3–3.5 s -> 2.3–2.7 s, output byte-identical. Transfers to the GPU box now use a Thunderbolt link when it is up (raw 12.7 vs 9.4 Gb/s on 10GbE; ssh caps both near 1 GB/s).

## Strata at 512K (2026-10-08)

Strata now runs with `--max-context 524288` (KV streaming with `--kv-resident 32768`: only a 32K window per attention layer stays in VRAM, the full int8 KV sits in pinned RAM, ~7 GB at 512K); the proxy reads Strata's limit from its `/v1/status`. 400K-token needle document (`doc-400k.txt`, 4 needles at ~51K / 153K / 306K / 388K), through the gateway's normal `qwen38-flash` alias (automatic handoff), thinking off, greedy:

| run | first token | needles |
|---|---|---|
| handoff | **226 s** (Strata prefill 210 s at ~1,900 t/s, save 3.2 s, stream 10.7 s) | 4/4 |
| Sushi alone | 631 s | 4/4 |

Same answer, 2.8x faster to first token. Not pursued: delta handoffs (Strata already reuses its cached prefix when a request extends the previous one: 30K after 20K prefilled 13.6K tokens in 6.6 s, so only ~1–2 s of transfer per 100K would remain to save); energy-efficient Ethernet (disabled on the GPU box's side, so the link never uses it). MTP draft head after a handoff: see the next section.

## MTP draft head after a handoff (2026-10-08)

A converted entry carries no MTP history, so Sushi's draft head starts blind after a restore. Strata does save its drafter's KV (the 13th session layer, int8, no indexer); against Sushi's own entry for identical tokens it maps row for row (V cosine 0.973) once K is shifted by +1 position (half-split pairs, YaRN x4: 0.944 vs 0.877 unshifted on the fastest pairs), and Sushi's head positions count from the spec's base. Sushi's head is a QSA layer whose indexer state Strata never computes, but with at most ~2,048 history rows it attends densely, so an experimental tail spec (last W rows, zero indexer state; branch `kvh-mtp-tail`) is adopted by Sushi ("MTP head restored (1024 tokens from base ...)").

Paired test, six 32K passages, 600 greedy tokens each:

| | draft acceptance | decode t/s |
|---|---|---|
| handoff (no MTP history) | 50.5 % | 56.2 |
| handoff + 1,024-row tail | 53.6 % | 55.8 |
| Sushi's own prefill | 54.0 % | 57.4 |

Paired differences: no-history vs Sushi's own decode -1.2 ± 1.4 t/s (se), tail vs no-history acceptance +3.1 ± 2.2 points but decode -0.4 ± 0.3 t/s. The decode cost of the missing history is about 2 % and not distinguishable from zero, and the tail does not recover it (Sushi's adaptive MTP evens out the extra accepted drafts), so it is not shipped. The earlier "~3 % slower" (four pairs) was within this noise.


## Agent reasoning history hands off (2026-10-08)

Until now the render refused any request whose history carried reasoning. Coding agents running with thinking on commonly send `reasoning_content` back on every assistant turn, so in such a session only the first request could hand off. Measured with a read-only agent session: 3 of 4 agent requests passed through, among them the turns that had just read two files (~12K and ~7K new tokens). Thinking-off sessions send no reasoning and were unaffected (handoffs at 9,082 and 5,978 new tokens).

Sushi hands that reasoning to the template (`server.zig` `messageReasoningFromObj`): `reasoning_content`, else `reasoning`, non-empty strings only, assistant turns only; the Qwen template trims it and, with `preserve_thinking` unset, keeps it on every turn, not only after the last user message. `render.py` now does the same. Live, in a two-turn agent session with thinking on: the render matched Sushi's own cache 22,735 tokens deep across an assistant turn with reasoning, and a 32,598-token request with 9,184 new tokens handed off (prefill 16.1 s) and was restored by Sushi at 32,593 tokens. The proxy now also logs why a request of 2,000 tokens or more passed through (`KVH_LOG_SKIPS_FROM_TOKENS`), to size the 3–5K range below the handoff threshold.

## Sushi 1.2.1 (2026-10-08)

**Sushi 1.2.1** (kvh build, in production since 2026-10-08 13:48). The import patch applies with line offsets only (regenerated); the MLX pins and release dylibs are byte-identical to 1.2.0. Cache format, fingerprint and entry layout are unchanged; the new root lock and private second-process root do not touch a single Sushi, and the startup sweep of index-less entries cannot catch an import, which is staged outside the root and renamed in. Thinking, effort, reasoning history and tool-definition fill are byte-identical in the source. The one render change: a history tool call whose arguments are not a JSON object (empty, null, an array, a scalar, malformed text) is now embedded as `{}` (before, the template raised and Sushi fell back to a generic format, dropping the call), so `render.py` now does the same instead of passing such requests through. NLL identical to 1.2.0 to the last digit (0.280633139 / 0.121696140); the KV round-trip and tool-calling check passed: tool calls 28/28, 31 of 8,740 tokens reprocessed after a restart.

Live 100K handoffs through the gateway (fresh nonce, needle 1): plain 51.2 s, thinking 49.8 s, tools 49.7 s first token, all three answers correct. The tools case carries two history calls with empty and malformed arguments; Sushi restored 100,853 cached tokens against a restore point of 100,852, so the `{}` render matches token for token.
