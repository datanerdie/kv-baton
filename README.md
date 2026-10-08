# kv-baton

NVIDIA GPUs prefill, an Apple Silicon Mac decodes: the prompt's KV state is handed over like a relay baton.

A Mac with lots of unified memory decodes a big model quickly but reads a long new prompt slowly. Two RTX 3090s read it about 3.5x faster. kv-baton lets the GPUs read (prefill) the prompt, converts their state into a cache entry for the Mac's inference server, and lets the Mac answer from that entry as if it had read the prompt itself. It is per request and automatic: a proxy in front of the Mac hands off only prompts big enough to gain from it, and everything else goes straight to the Mac.

| new prompt | first token, with handoff | Mac alone | |
|---|---|---|---|
| 10K tokens | 7.3 s | 13.9 s | 1.9x |
| 32K tokens | 16.0 s | 45.3 s | 2.8x |
| 100K tokens | ~49 s | ~153 s | 3.1x |
| 400K tokens | 226 s | 631 s | 2.8x |

The answers are the same: needle tests find every needle both ways, and a 150-case quality suite scores the same with every request handed off as on the Mac alone.

The model is Qwen3.8-Flash-Next, a hybrid MoE (Gated DeltaNet layers plus 12 sparse-attention layers), served by two different engines with different quantisations: [Strata](https://github.com/Niko1221/Strata) on CUDA and [Sushi](https://github.com/beamivalice/sushi) (Zig + MLX) on the Mac. Moving its state between them means converting attention KV, recurrent and convolution state, and the attention indexer's pooled keys from one engine's format into the other's.

**Status: experimental, single-user, built for one home setup.** It needs a small local patch to each engine (included). Nothing here is supported by either upstream project.

## Performance

Measured on the setup below, Sushi 1.2.0 and Strata v0.1.40.1, through the proxy, thinking off and greedy unless noted. Details and history are in [RESULTS.md](RESULTS.md).

| | machine | role |
|---|---|---|
| Sushi 1.2.0, `Qwen3.8-Flash-Next-Sushi-4bpw` | M4 Max Mac Studio, 128 GB | serves every request, does all decoding |
| Strata v0.1.40.1, unsloth `UD-IQ4_XS` GGUF, `--kv int8`, 512K context | Linux box, 2x RTX 3090 (24 GB each) | prefill only |
| the link between them | 10GbE, or a Thunderbolt cable | ssh streams ~0.9–1.0 GB/s either way |

**Time to first token on a new prompt**

| prompt | with handoff | Mac alone | speed-up |
|---|---|---|---|
| ~5K tokens | 5.1 s | 7.3 s | 1.4x |
| ~10K | 7.3 s | 13.9 s | 1.9x |
| ~20K | 11.2 s | 27.7 s | 2.5x |
| ~32K | 16.0 s | 45.3 s | 2.8x |
| ~100K | ~49 s | ~153 s | 3.1x |
| ~400K | 226 s | 631 s | 2.8x |

A handoff costs about 2.9 s of fixed overhead plus Strata's prefill, so it breaks even near 3,000 new tokens; the proxy hands off from 5,000. Short chats are relayed straight to the Mac (first token ~0.3 s), and follow-up questions about a document already in the Mac's cache answer in 0.3–1.9 s.

**Prefill throughput** (tokens per second)

| prompt | Strata, 2x 3090 | Sushi, M4 Max | handoff, end to end (tokens / first-token time) |
|---|---|---|---|
| ~10K | ~2,200 | ~720 | ~1,370 |
| ~32K | ~2,300 | ~720 | ~2,040 |
| ~100K | ~2,300 | ~660 | ~2,050 |
| ~400K | ~1,900 | ~630 | ~1,770 |

**Where a 100K handoff's time goes:** Strata prefill 43.6 s, saving the session 1.0 s, streaming and converting it 2.6 s, import into Sushi under 0.1 s, then Sushi restores the entry (~1 s) and starts answering.

**Decode** stays on the Mac: about 57 tokens/s at 32K after Sushi's own prefill and about 56 after a handoff, a difference within noise (paired six-passage test). The one known cost is that the MTP draft head starts without history after a restore (RESULTS.md has the investigation).

**Quality**

| test | Mac alone | with handoff |
|---|---|---|
| 100K needle document, 8 needles | 8/8 | 8/8, replies token-identical |
| 400K needle document, 4 needles | 4/4 | 4/4 |
| club-3090 quality suite, 8 packs, 150 cases, mixed arm (pass@1) | 123 | 130 (every request handed off) |
| same, thinking on (pass@1) | 130 | 128 (every request handed off) |

The quality-suite run handed off all 752 requests (665 were decoded from Strata's state; Sushi served the rest, short repeats, from its own cache); both arms are within the suite's run-to-run noise. Open-ended free text is not bit-identical to the Mac reading the prompt itself (top-5 KL ~0.04 on a 100K summary prompt), but stays coherent.

## How it works

```
client --> LiteLLM :4001 --> handoff proxy :8002 --> Sushi :8000 (Mac)   <- every request ends here
                                   |
                                   | >= 5,000 tokens Sushi has not cached
                                   v
                     ssh -L 18080 --> Strata :8080 (GPU box)
```

1. The proxy renders the request with the model's chat template, resolving thinking and effort the way Sushi does, and checks the result against Sushi's own `/tokenize`. It works out how much of the prompt Sushi already has in its disk prefix cache. With fewer than 5,000 new tokens, or anything it cannot reproduce exactly (images, a forced `tool_choice`, `response_format`, a render that differs from Sushi's tokens), it relays the request to Sushi untouched.
2. It checks that Sushi runs a version the render was validated against and that Strata serves the expected model, engine and context.
3. It sends Strata the checked token ids as `kvh_prompt_ids` with `max_tokens: 1` (prefill only; a local Strata patch makes Strata prefill exactly those ids instead of rendering the template itself) and asks Strata to save its slot to a session file.
4. `ssh <gpu-host> cat <session>` is piped into `strata_to_sushi.py -`, which reads the stream on one thread while it converts on another and writes a Sushi cache entry:
   - the 12 attention layers' int8 KV is dequantised and requantised to Sushi's 8-bit affine format (group 64), layer by layer;
   - the 36 Gated DeltaNet layers' conv and recurrent state is converted, including Strata's different v-head order and `[k, h, v]` layout;
   - the per-layer embedding conv tail, the indexer's pending keys and pooled rows are carried over;
   - `tokens.bin`, a v8 `meta.json` and a `kvh.json` stamp (versions, model, render and converter hashes) are written.
5. The proxy checks the entry's token ids against its own render, moves the entry into Sushi's cache folder and calls `POST /v1/kvh/import` (the Sushi patch), so the running server indexes it.
6. The original request is relayed to Sushi, which restores the entry (all but the last 7 tokens) and answers.

While a handoff runs, a streaming client gets SSE comment lines (`: handoff <step>`) every 10 s so it does not time out, and other requests are never held behind it. Any failure at any step falls back to sending the request to Sushi untouched: slower, never wrong.

## Files

| file | what |
|---|---|
| `kvh/handoff_proxy.py` | the proxy (stdlib HTTP server, SSE relay, keep-alives, pass-through on any doubt) |
| `kvh/handoff.py` | one handoff as a sequence of injectable steps; `real_steps()` wires them to ssh, the converter and Sushi |
| `kvh/render.py` | the chat template render, following Sushi's thinking, effort and tool rules |
| `kvh/cache_index.py` | reads Sushi's disk cache: best restorable prefix, next free entry id |
| `kvh/gate.py` | in-flight request counter, used only for the stock-Sushi restart fallback |
| `kvh/strata_to_sushi.py` | the converter (session stream or dump folder -> Sushi entry; needs `mlx`, `numpy`) |
| `kvh/strata_dump.cpp` | dumps a Strata session file to raw arrays (the older two-step path; useful for inspection) |
| `kvh/sushi_entry.py`, `kvh/compare_states.py` | read a Sushi entry; compare converted state against Sushi's own |
| `kvh/needle_test.py`, `kvh/client.py` | the needle test and the benchmark client |
| `kvh/sushi-kvh-import.patch`, `kvh/build-sushi-kvh.sh` | Sushi patch adding `POST /v1/kvh/import {"id": N}` (for 1.2.0; the 1.1.1 version is in the git history), and a build script that needs no Xcode |
| `kvh/strata-kvh-prompt-ids.patch` | Strata patch: an optional `kvh_prompt_ids` on `/v1/chat/completions` replaces the template render (Python server only) |
| `kvh/strata-peer-session.patch` | Strata patch: allow session files with `--peer-device` when `STRATA_ALLOW_PEER_SESSION=1` (two GPUs) |
| `kvh/gpu-box/start-strata.sh` | starts Strata on the GPU box with the patched binary |
| `kvh/launchd/*.plist` | macOS agents for the proxy and the ssh tunnel (replace `/Users/YOU`) |
| `examples/` | the Strata server config and the LiteLLM aliases |
| `kvh/tests/` | 160 unit tests |

## Setting it up

This was built for one specific pair of machines; expect to adapt paths.

**GPU box.** Install Strata v0.1.40.1 and apply `kvh/strata-kvh-prompt-ids.patch` (required: without it Strata renders tools differently from Sushi and tool conversations never hand off; plain requests still work). For two GPUs, build its engine with `kvh/strata-peer-session.patch` applied (one GPU works with the stock engine, at ~1,500 tokens/s). Pack `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ4_XS` with Strata's tools and write a server config like `examples/strata-peer.json`. YaRN factor 4 and `--kv int8` are required, since the converter assumes them. `--max-context 524288` with `--kv-resident 32768` keeps only a 32K window per attention layer in VRAM and the full KV in pinned RAM (~7 GB at 512K). Start it with `kvh/gpu-box/start-strata.sh`.

**Mac.** Install Sushi 1.2.0 (`brew install beamivalice/tap/sushi`, or unpack the release tarball and point `SUSHI_REL_LIB` at its `lib`) with the `Qwen3.8-Flash-Next-Sushi-4bpw` pack, then build the patched server with `kvh/build-sushi-kvh.sh` (`SUSHI_VERSION` picks the release, default 1.2.0) and run that binary instead of Homebrew's. Do not start it with `--think`: it changes the thinking defaults the proxy copies, so the proxy then hands nothing off. Stock Sushi also works: the proxy then restarts Sushi to make it load the entry (~10 s more, and other requests wait during the restart). Then:

```sh
uv run --with pytest --with transformers --with jinja2 pytest kvh/tests -q      # 162 passed, 2 skipped
cp kvh/launchd/*.plist ~/Library/LaunchAgents/      # after editing paths and the ssh host
launchctl load ~/Library/LaunchAgents/local.kvh-tunnel.plist ~/Library/LaunchAgents/local.kvh-proxy.plist
```

Point clients (or a gateway, see `examples/litellm.yaml`) at `http://127.0.0.1:8002/v1`. Every chat request is a handoff candidate; models named `qwen38-flash-bigdoc*` are always candidates at the same threshold.

**Configuration** (environment of the proxy; defaults in brackets):

| variable | meaning |
|---|---|
| `KVH_AUTO_MIN_NEW_TOKENS` [`5000`] | hand off from this many tokens Sushi has not cached (break-even is ~3,000) |
| `KVH_STRATA_HOST` [`gpu-box`] | ssh host of the GPU box; sessions are read from `~/kvh/sessions` there |
| `KVH_STRATA_URL` [`http://127.0.0.1:18080`] | Strata through the tunnel |
| `KVH_STRATA_MODEL` [`qwen3.8-flash-next-unsloth-ud-iq4_xs`], `KVH_STRATA_ENGINE` [`0.1.40`] | what Strata's `/v1/status` must report before a handoff |
| `KVH_SUSHI_URL` [`http://127.0.0.1:8000`] | Sushi |
| `KVH_PROXY_PORT` [`8002`] | the proxy's port (loopback only) |
| `KVH_SUSHI_MODEL_DIR` [`~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw`] | tokenizer and chat template used for the render |
| `KVH_SUSHI_CACHE_ROOT` [`~/.sushi/kv-cache/526f67face43a3d8`] | Sushi's cache folder for this pack (fallback when the log does not name it) |
| `KVH_SUSHI_LOG` [`../sushi-server.log`] | Sushi's log, read for its version and the `[disk-cache] scanned ... at <dir>` line |
| `KVH_SUSHI_RESTART` [`~/.local/bin/sushi-restart`] | a command that restarts Sushi (needed only with stock Sushi) |
| `KVH_UV` [`~/.local/bin/uv`] | uv, used to run the converter with `mlx` and `numpy` |
| `KVH_FORCE_ALL`, `KVH_MIN_NEW_TOKENS`, `KVH_INCOMING`, `KVH_PROXY_LOG` | for a separate test instance that hands off every request |

The proxy only hands off while the expected model (`Qwen3.8-Flash-Next-Sushi-4bpw` at 1,048,576 context) is what Sushi reports, and only to a Sushi version whose rendering rules it was checked against (1.1.1, 1.2.0); anything else is passed through.

## Limits

- **Thinking follows Sushi's rules, not the template's** (1.1.1 and 1.2.0 without `--think`). Sushi reads `reasoning_effort` only as a top-level field (inside `chat_template_kwargs` it is ignored), runs thinking-on without an effort word at low, and a request naming neither with thinking off; 1.2.0 refuses "minimal". `kvh/render.py` copies these rules, so re-check them on every Sushi upgrade: the `/tokenize` check covers the tokenizer, not the template. Requests with a top-level `reasoning` object or `response_format` (Sushi writes a schema instruction into the prompt) pass through. Reasoning that an agent sends back on earlier assistant turns (`reasoning_content`, or `reasoning` as a string) is rendered the way Sushi hands it to the template, which keeps it on every turn while `preserve_thinking` is unset.
- **Tools hand off, images do not.** Tools render as Sushi renders them (the request's tool objects as sent, a missing `description` or `parameters` filled in as Sushi fills it, JSON-string call arguments parsed); a conversation may end with a tool result. `tool_choice` "none" drops the tools as Sushi does; "required" or a named function pass through, because Sushi adds its own instruction for those. Sushi's cache key includes `has_tools`: the proxy's lookup filters on it and the handoff marks the entry.
- **Exact token agreement is required, and checked twice.** Before the prefill the rendered text goes to Sushi's `/tokenize` (0.1 s at 100K); after it, the token ids in Strata's session are compared with the render. Requests reach Sushi unchanged. Any mismatch is caught and passed through.
- **Up to Strata's context**, 524,288 tokens in the example config. The proxy reads the limit from Strata's `/v1/status`; longer prompts go to Sushi alone.
- **One handoff at a time.** A second big prompt waits for the first; Strata already reuses its cached prefix when a request extends the previous one. Strata needs the GPUs to itself, so stop anything else using them before starting it.
- **Sushi prefers its own cache** unless the converted entry is at least 256 tokens longer than the prefix it already holds in RAM, so very short or repeated prompts are answered from Sushi's own state.
- The converter is written against Sushi's cache format v8 (unchanged from 1.1.1 to 1.2.0) and Strata's STRSESS v1 session format. Either upstream can change these at any time.

## Licence

MIT, see [LICENSE](LICENSE). The two patches modify [Strata](https://github.com/Niko1221/Strata) (MIT) and [Sushi](https://github.com/beamivalice/sushi) (MIT / Apache-2.0) and are offered under those projects' licences. No model weights or model-derived data are included; Qwen3.8-Flash-Next and its quantisations are under their own licences.
