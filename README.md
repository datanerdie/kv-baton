# kv-baton

NVIDIA GPUs prefill, an Apple Silicon Mac decodes: the prompt's KV state is handed over like a relay baton.

A Mac decodes Qwen3.8-Flash-Next quickly but is slow to read a long new prompt. Two RTX 3090s read it about 3.5x faster. This project lets the GPUs read (prefill) a big document, converts their prompt state into a cache entry for the Mac's inference server, and lets the Mac answer from that entry as if it had read the document itself.

At 100K tokens the first token arrives in **49 s instead of ~152 s**, with the same answers: on a needle test the replies were token-identical to the Mac reading the document alone.

**Status: experimental, single-user, built for one home setup.** It needs a small local patch to each of the two inference engines (included). Nothing here is supported by either upstream project.

## The pieces

| | machine | role |
|---|---|---|
| [Sushi](https://github.com/beamivalice/sushi) 1.2.0 (Zig + MLX) | M4 Max Mac Studio, 128 GB | serves Qwen3.8-Flash-Next `Sushi-4bpw` and does all decoding; prefill ~650 t/s |
| [Strata](https://github.com/Niko1221/Strata) v0.1.40.1 (C++/CUDA) | Linux box with 2x RTX 3090 | prefill only, unsloth `UD-IQ4_XS` GGUF, `--kv int8`; 2,283 t/s across both cards |
| `kvh/handoff_proxy.py` | Mac | OpenAI-compatible proxy in front of Sushi; decides per request whether to hand off |
| `kvh/strata_to_sushi.py` | Mac | turns a Strata session file into a Sushi prefix-cache entry |
| LiteLLM gateway (optional) | Mac | exposes the model aliases; all of them go through the proxy |

```
client --> LiteLLM :4001 --> handoff proxy :8002 --> Sushi :8000 (Mac)   <- every request ends here
                                   |
                                   | >= 5,000 uncached tokens (any alias)
                                   v
                     ssh -L 18080 --> Strata :8080 (NVIDIA box)
```

## What happens on a big document

1. The proxy renders the request with the model's chat template and works out how much of it Sushi already has in its disk prefix cache. Thinking and effort are resolved the way Sushi does it, and the rendered text is checked against Sushi's own `/tokenize`. Fewer than 5,000 new tokens, images, a `tool_choice` that forces a call, a render that differs from Sushi's tokens, or the NVIDIA box unreachable: the request goes straight to Sushi unchanged.
2. Otherwise it sends the request to Strata with `max_tokens: 1` (prefill only) and the checked token ids as `kvh_prompt_ids` (local Strata patch: Strata prefills exactly those tokens instead of rendering the template itself), and asks Strata to save its slot to a session file.
3. `ssh <gpu-host> cat <session>` is piped straight into `strata_to_sushi.py -`, which parses the session file as it arrives and writes a Sushi cache entry:
   - the 12 attention layers' int8 KV is dequantised and requantised to Sushi's 8-bit affine format (group 64), layer by layer;
   - the 36 Gated DeltaNet layers' conv and recurrent state is converted, including Strata's different v-head order and `[k, h, v]` layout;
   - the per-layer embedding conv tail, the indexer's pending keys and pooled rows are carried over;
   - `tokens.bin` and a v8 `meta.json` are written.
4. The proxy checks that the entry's token ids equal its own render, moves the entry into Sushi's cache folder and calls `POST /v1/kvh/import` (the Sushi patch) so the running server indexes it.
5. The original request is relayed to Sushi, which restores the entry (all but the last 7 tokens) and starts answering.

While this runs, a streaming client gets SSE comment lines (`: handoff <step>`) every 10 s so it does not time out. Any failure at any step falls back to sending the request to Sushi untouched: slower, never wrong. Follow-up questions about the same document hit Sushi's cache directly (0.3 s to first token).

## Results (100K-token needle document, 8 needles)

| path | first token | needles |
|---|---|---|
| Sushi alone, cold | ~152 s | 8/8 |
| handoff, with a Sushi restart to load the entry | 72-74 s | 8/8 |
| handoff, `/v1/kvh/import` (no restart) | 59.1 s | 8/8 |
| **handoff, streamed conversion (current)** | **49.2 s** | **8/8** |

Current step times: Strata prefill 43.6 s, save 1.0 s, stream + convert 3.5 s, import 0.0 s. Details, the single-GPU numbers and the quality comparison (top-5 KL 0.00005-0.0008 against Sushi cold) are in [RESULTS.md](RESULTS.md).

## Files

| file | what |
|---|---|
| `kvh/handoff_proxy.py` | the proxy (stdlib HTTP server, SSE relay, keep-alives, pass-through on any doubt) |
| `kvh/handoff.py` | one handoff as a sequence of injectable steps; `real_steps()` wires them to ssh, the converter and Sushi |
| `kvh/cache_index.py` | reads Sushi's disk cache: best restorable prefix, next free entry id |
| `kvh/render.py` | the chat template render the proxy compares against (transformers) |
| `kvh/gate.py` | in-flight request counter, used only for the stock-Sushi restart fallback |
| `kvh/strata_to_sushi.py` | the converter (session stream or dump folder -> Sushi entry; needs `mlx`, `numpy`) |
| `kvh/strata_dump.cpp` | dumps a Strata session file to raw arrays (the older two-step path; useful for inspection) |
| `kvh/sushi_entry.py`, `kvh/compare_states.py` | read a Sushi entry; compare converted state against Sushi's own |
| `kvh/needle_test.py`, `kvh/client.py` | the needle test and the benchmark client |
| `kvh/sushi-kvh-import.patch`, `kvh/build-sushi-kvh.sh` | Sushi patch adding `POST /v1/kvh/import {"id": N}` (for 1.2.0; the 1.1.1 version is in the git history), and a build script that needs no Xcode |
| `kvh/strata-peer-session.patch` | Strata patch: allow session files with `--peer-device` when `STRATA_ALLOW_PEER_SESSION=1` |
| `kvh/strata-kvh-prompt-ids.patch` | Strata patch: an optional `kvh_prompt_ids` on `/v1/chat/completions` replaces the template render (Python server only) |
| `kvh/gpu-box/start-strata.sh` | starts Strata on the GPU box with the patched binary |
| `kvh/launchd/*.plist` | macOS agents for the proxy and the ssh tunnel (replace `/Users/YOU`) |
| `examples/` | the Strata server config and the LiteLLM aliases |
| `kvh/tests/` | 136 unit tests |

## Setting it up

This was built for one specific pair of machines; expect to adapt paths.

**GPU box.** Install Strata v0.1.40.1 and apply `kvh/strata-kvh-prompt-ids.patch` (required: without it Strata renders tools differently from Sushi and tool conversations never hand off; plain requests still work). Build its engine with `kvh/strata-peer-session.patch` applied for two GPUs (one GPU works with the stock engine at 1,476 t/s). Pack `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ4_XS` with Strata's tools and write a server config like `examples/strata-peer.json`. YaRN factor 4 and `--kv int8` are required, since the converter assumes them. Start it with `kvh/gpu-box/start-strata.sh`.

**Mac.** Install Sushi 1.2.0 (`brew install beamivalice/tap/sushi`, or unpack the release tarball and point `SUSHI_REL_LIB` at its `lib`) with the `Qwen3.8-Flash-Next-Sushi-4bpw` pack, then build the patched server with `kvh/build-sushi-kvh.sh` (`SUSHI_VERSION` picks the release, default 1.2.0) and run that binary instead of Homebrew's. Do not start it with `--think`: it changes the thinking defaults the proxy copies, so the proxy then hands nothing off. Stock Sushi also works: the proxy then restarts Sushi to make it load the entry (~10 s more, and other requests wait during the restart). Then:

```sh
uv run --with pytest --with transformers --with jinja2 pytest kvh/tests -q      # 136 passed
cp kvh/launchd/*.plist ~/Library/LaunchAgents/      # after editing paths and the ssh host
launchctl load ~/Library/LaunchAgents/local.kvh-tunnel.plist ~/Library/LaunchAgents/local.kvh-proxy.plist
```

Point clients (or the gateway, see `examples/litellm.yaml`) at `http://127.0.0.1:8002/v1`. Every chat request is a handoff candidate: with at least 5,000 tokens Sushi has not cached (`KVH_AUTO_MIN_NEW_TOKENS`) it is prefilled on the GPUs, anything smaller is relayed unchanged without further work. Measured break-even is about 3,000 tokens (first token, handoff vs Mac alone: 5K 5.1 vs 7.3 s, 10K 7.3 vs 13.9 s, 20K 11.2 vs 27.7 s). Models named `qwen38-flash-bigdoc*` are always candidates at the same threshold.

**Configuration** (environment of the proxy; defaults in brackets):

| variable | meaning |
|---|---|
| `KVH_STRATA_HOST` [`gpu-box`] | ssh host of the GPU box; sessions are read from `~/kvh/sessions` there |
| `KVH_STRATA_URL` [`http://127.0.0.1:18080`] | Strata through the tunnel |
| `KVH_SUSHI_URL` [`http://127.0.0.1:8000`] | Sushi |
| `KVH_PROXY_PORT` [`8002`] | the proxy's port (loopback only) |
| `KVH_SUSHI_MODEL_DIR` [`~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw`] | tokenizer and chat template used for the render |
| `KVH_SUSHI_CACHE_ROOT` [`~/.sushi/kv-cache/526f67face43a3d8`] | Sushi's cache folder for this pack (fallback when the log does not name it) |
| `KVH_SUSHI_LOG` [`../sushi-server.log`] | Sushi's log, read for the `[disk-cache] scanned ... at <dir>` line |
| `KVH_SUSHI_RESTART` [`~/.local/bin/sushi-restart`] | a command that restarts Sushi (needed only with stock Sushi) |
| `KVH_UV` [`~/.local/bin/uv`] | uv, used to run the converter with `mlx` and `numpy` |

The proxy only hands off while the expected model (`Qwen3.8-Flash-Next-Sushi-4bpw` at 1,048,576 context) is what Sushi reports; anything else is passed through.

## Limits

- **Thinking follows Sushi's rules, not the template's** (1.1.1 and 1.2.0 without `--think`). Sushi reads `reasoning_effort` only as a top-level field (inside `chat_template_kwargs` it is ignored), runs thinking-on without an effort word at low, and a request naming neither with thinking off; 1.2.0 refuses "minimal". `kvh/render.py` copies these rules, so re-check them on every Sushi upgrade: the `/tokenize` check covers the tokenizer, not the template. Requests with a `reasoning` object, `response_format` (Sushi writes a schema instruction into the prompt), or earlier assistant turns carrying `reasoning_content` pass through.
- **Tools hand off, images do not.** Tools render as Sushi renders them (the request's tool objects as sent, a missing `description` or `parameters` filled in as Sushi fills it, JSON-string call arguments parsed); a conversation may end with a tool result. `tool_choice` "none" drops the tools as Sushi does; "required" or a named function pass through, because Sushi adds its own instruction for those. Sushi's cache key includes `has_tools`: the proxy's lookup filters on it and the handoff marks the entry.
- **Two token checks.** Before the prefill the rendered text goes to Sushi's `/tokenize` (0.1 s at 100K); after it, the token ids in Strata's session are compared with the render. Strata prefills the ids it is given, so its own template no longer matters; Sushi's template side is covered by the thinking rules above and was checked token for token against Sushi's cache (plain, thinking and a 10K tool conversation).
- **Up to 131,072 tokens**, Strata's context in this config. Longer prompts go to Sushi alone.
- **Exact token agreement is required.** The proxy normalises the two differences found in practice: content arrays are joined with `"\n"`, as Sushi does, and U+202F is replaced by a space, because Sushi's tokenizer splits "°C" after it differently. Any remaining mismatch is caught and passed through.
- **One handoff at a time** (the proxy runs them one after another). Strata needs the GPUs to itself, so stop anything else using them before starting it.
- Free-text answers to open questions are not bit-identical to Sushi cold (top-5 KL ~0.04 on a 100K summary prompt), though they are coherent; retrieval matched exactly in every test.
- The converter is written against Sushi's cache format v8 (unchanged from 1.1.1 to 1.2.0) and Strata's STRSESS v1 session format. Either upstream can change these at any time.

## Licence

MIT, see [LICENSE](LICENSE). The two patches modify [Strata](https://github.com/Niko1221/Strata) (MIT) and [Sushi](https://github.com/beamivalice/sushi) (MIT / Apache-2.0) and are offered under those projects' licences. No model weights or model-derived data are included; Qwen3.8-Flash-Next and its quantisations are under their own licences.
