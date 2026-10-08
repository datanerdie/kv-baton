"""Proxy in front of Sushi (:8000) on 127.0.0.1:8002. Every request is relayed unchanged, except chat
requests for `qwen38-flash-bigdoc*` with >= min_new_tokens uncached tokens: those are prefilled by Strata on the GPU box
first (handoff.run), then relayed, and Sushi answers from the injected cache entry. Design:
README.md and RESULTS.md at the repository root
"""
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import http.client
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handoff  # noqa: E402
from cache_index import best_restore, find_cache_root  # noqa: E402
from gate import Gate  # noqa: E402

# Thinking requests hand off since render.py follows Sushi's own thinking/effort rules (2026-10-07); False is the
# kill switch that passes every thinking request through again.
THINK_HANDOFF = True
BIGDOC_PREFIX = "qwen38-flash-bigdoc"
# A handoff restarts Sushi, so it is only done while the production Sushi pack is what is running (never on another
# engine, never to start a server the user stopped on purpose).
EXPECTED_SUSHI_MODEL = "Qwen3.8-Flash-Next-Sushi-4bpw"
EXPECTED_SUSHI_CTX = 1048576
LOG = os.path.expanduser(os.environ.get("KVH_PROXY_LOG") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "handoff_proxy.log"))
_log_lock = threading.Lock()


def log(msg):
    with _log_lock, open(LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


# Automatic handoff (2026-10-08): chat requests on the normal aliases are candidates too. Threshold measured the same
# night (first token, handoff vs Sushi alone): 5K 5.1 vs 7.3 s, 10K 7.3 vs 13.9 s, 20K 11.2 vs 27.7 s; a handoff costs
# ~2.9 s plus Strata's prefill at ~2,300 t/s against Sushi's ~700 t/s, so it breaks even near 3,000 new tokens.
AUTO_HANDOFF = True
AUTO_MIN_NEW_TOKENS = int(os.environ.get("KVH_AUTO_MIN_NEW_TOKENS", "5000"))
# render.py copies Sushi's rendering rules; they were checked token for token on these versions only.
VALIDATED_SUSHI_VERSIONS = ("1.1.1", "1.2.0")
# The converter assumes this Strata model (UD-IQ4_XS, YaRN 4, int8 KV) and engine (STRSESS v1).
EXPECTED_STRATA = {"model": os.environ.get("KVH_STRATA_MODEL", "qwen3.8-flash-next-unsloth-ud-iq4_xs"),
                   "engine": os.environ.get("KVH_STRATA_ENGINE", "0.1.40")}
SUSHI_VERSION_RE = re.compile(r"^sushi (\d+\.\d+\.\d+)", re.M)


def sushi_version(log_path):
    """The version Sushi printed when it started (`sushi 1.2.0 (MLX ...)` near the top of its log), or None."""
    try:
        with open(log_path, errors="replace") as f:
            m = SUSHI_VERSION_RE.search(f.read(8192))
    except OSError:
        return None
    return m.group(1) if m else None


def strata_status(url=handoff.STRATA_URL):
    with urllib.request.urlopen(url + "/v1/status", timeout=5) as r:
        return json.load(r)


def file_sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


# Debug capture, off by default: while CAPTURE_DIR/ON exists, each bigdoc request body is saved (mode 0600) as
# CAPTURE_DIR/last-bigdoc.json, overwriting the previous one. `touch` the flag to switch on, delete it to switch off.
CAPTURE_DIR = os.path.expanduser("~/.sushi/kvh-capture")


def capture(raw):
    try:
        if not os.path.exists(os.path.join(CAPTURE_DIR, "ON")):
            return
        path = os.path.join(CAPTURE_DIR, "last-bigdoc.json")
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        log(f"capture: saved the bigdoc request ({len(raw)} bytes) to {path}")
    except OSError as e:
        log(f"capture failed: {e}")


def clean_incoming(incoming):
    """Remove `dump-*` and `e<N>` folders left in `incoming` by a crashed handoff; touches nothing else."""
    removed = []
    try:
        names = sorted(os.listdir(incoming))
    except OSError:
        return removed
    for name in names:
        path = os.path.join(incoming, name)
        if (name.startswith("dump-") or re.fullmatch(r"e\d+", name)) and os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
    return removed


def make_server(port, upstream, renderer, cfg, steps, keepalive_s=10.0, think_handoff=THINK_HANDOFF, cache_root_fn=None,
                force_all=False, auto_handoff=False, auto_min_new_tokens=AUTO_MIN_NEW_TOKENS, sushi_version_fn=None,
                strata_status_fn=None):
    """renderer None: pass-through only (no handoffs). cache_root_fn: returns the current cache root, per request.
    force_all (test instances only): every chat completion is a candidate at cfg.min_new_tokens, whatever its model.
    auto_handoff: chat completions on other models than qwen38-flash-bigdoc* are candidates at auto_min_new_tokens;
    below it they are relayed without a log line, a cache lookup or any other work beyond the render.
    sushi_version_fn / strata_status_fn: identity checks before a handoff (None skips that check, for tests)."""
    gate, one_handoff = Gate(), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, status, payload, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _chunk(self, data):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def _fail(self, status, msg, headers_sent):
            payload = json.dumps({"error": msg}).encode()
            if headers_sent:
                self._chunk(b"data: " + payload + b"\n\n")
                self._chunk(b"")
            else:
                self._send(status, payload)

        def _relay(self, raw, streaming, headers_sent, expect_cached=None):
            """Forward the request to Sushi and stream its answer back; Sushi counts as busy meanwhile.
            After a handoff, `expect_cached` is the restore point: the usage's cached_tokens is logged against it."""
            gate.enter()
            try:
                req = urllib.request.Request(upstream + self.path, data=raw, headers={"Content-Type": "application/json"})
                try:
                    r = urllib.request.urlopen(req, timeout=14400)
                except urllib.error.HTTPError as e:
                    body = e.read()
                    if headers_sent:
                        self._fail(e.code, body.decode(errors="replace"), True)
                    else:
                        self._send(e.code, body)
                    return
                except OSError as e:
                    log(f"relay: Sushi unreachable: {e}")
                    self._fail(502, f"Sushi unreachable: {e}", headers_sent)
                    return
                with r:
                    if not streaming:
                        try:
                            data = r.read()
                        except (OSError, http.client.HTTPException) as e:
                            log(f"relay: upstream read failed: {type(e).__name__}: {e}")
                            self._fail(502, f"Sushi reply broke off: {e}", headers_sent)
                            return
                        if expect_cached is not None:
                            self._check_cached(data, expect_cached)
                        self._send(r.status, data, r.headers.get("Content-Type", "application/json"))
                        return
                    if not headers_sent:
                        self._sse_headers(r.status)
                    lines = iter(r)
                    while True:
                        try:
                            line = next(lines)
                        except StopIteration:
                            break
                        except (OSError, http.client.HTTPException) as e:    # upstream broke; client writes are below
                            log(f"relay: upstream stream failed: {type(e).__name__}: {e}")
                            self._chunk(b"data: " + json.dumps({"error": f"Sushi stream broke off: {e}"}).encode() + b"\n\n")
                            break
                        if expect_cached is not None and b'"usage"' in line and line.startswith(b"data:"):
                            self._check_cached(line[5:], expect_cached)
                        self._chunk(line)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            finally:
                gate.leave()

        def _check_cached(self, data, expect):
            try:                               # logging only: never let it cost the client its reply
                u = json.loads(data).get("usage") or {}
                got = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                if got is not None:
                    log(f"handoff: Sushi restored {got} cached tokens (restore point {expect})"
                        + ("" if got >= expect else " -> HANDOFF DID NOT TAKE"))
            except Exception as e:
                log(f"bigdoc: could not read cached_tokens ({type(e).__name__}: {e})")

        def _sse_headers(self, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def do_GET(self):
            try:
                with urllib.request.urlopen(upstream + self.path, timeout=30) as r:
                    self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}).encode())

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw)
            except ValueError:
                return self._relay(raw, False, False)
            if not isinstance(body, dict):
                return self._relay(raw, False, False)
            streaming = bool(body.get("stream"))
            model = str(body.get("model") or "")
            bigdoc = force_all or model.startswith(BIGDOC_PREFIX)
            if not self.path.endswith("/chat/completions") or not (bigdoc or auto_handoff):
                return self._relay(raw, streaming, False)
            if bigdoc:
                capture(raw)
            kwargs = body.get("chat_template_kwargs")
            if kwargs is not None and not isinstance(kwargs, dict):
                if bigdoc:
                    log("bigdoc: chat_template_kwargs is not an object -> pass through")
                return self._relay(raw, streaming, False)
            threshold = cfg.min_new_tokens if bigdoc else auto_min_new_tokens
            return self._bigdoc(raw, body, streaming, threshold, "bigdoc" if bigdoc else "auto")

        def _sushi_is_production(self):
            """(ok, reason): the upstream answers /v1/models with the production pack at the production ctx."""
            try:
                with urllib.request.urlopen(upstream + "/v1/models", timeout=5) as r:
                    m = json.load(r)["data"][0]
                got = (m.get("id"), m.get("context_length"))
            except Exception as e:
                return False, f"/v1/models failed ({type(e).__name__}: {e})"
            if got != (EXPECTED_SUSHI_MODEL, EXPECTED_SUSHI_CTX):
                return False, f"upstream is {got[0]!r} ctx {got[1]!r}, not {EXPECTED_SUSHI_MODEL!r} ctx {EXPECTED_SUSHI_CTX}"
            effort = m.get("default_reasoning_effort")
            if effort not in (None, "off"):    # sushi --think X changes the defaults render.py follows (1.2.0)
                return False, f"Sushi's default reasoning effort is {effort!r} (--think), which the render does not model"
            return True, ""

        def _sushi_tokenizes_alike(self, prepared):
            """(ok, reason): Sushi's own tokenizer gives the render's ids for the rendered text. The Strata check after the
            prefill compares Strata with the render only; this is the one check against Sushi, made before any work."""
            try:
                req = urllib.request.Request(upstream + "/tokenize", data=json.dumps({"content": prepared.text}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    got = json.load(r)["tokens"]
            except Exception as e:
                return False, f"/tokenize failed ({type(e).__name__}: {e})"
            if got != prepared.ids:
                first = next((i for i, (a, b) in enumerate(zip(got, prepared.ids)) if a != b), min(len(got), len(prepared.ids)))
                return False, f"Sushi tokenizes {len(got)}, render {len(prepared.ids)}, first difference at {first}"
            return True, ""

        def _identity(self):
            """(ok, reason, stamp): Sushi runs a version the render was checked against and Strata serves the model and
            engine the converter assumes; stamp records them in the converted entry (kvh.json)."""
            stamp = {}
            if sushi_version_fn is not None:
                v = sushi_version_fn()
                if v not in VALIDATED_SUSHI_VERSIONS:
                    return False, f"Sushi version {v!r} is not one the render was checked against {VALIDATED_SUSHI_VERSIONS}", None
                stamp["sushi_version"] = v
            if strata_status_fn is not None:
                try:
                    st = strata_status_fn()
                except Exception as e:
                    return False, f"Strata /v1/status failed ({type(e).__name__}: {e})", None
                got = {"model": st.get("model"), "engine": st.get("engine")}
                if got != EXPECTED_STRATA:
                    return False, f"Strata serves {got}, the converter expects {EXPECTED_STRATA}", None
                stamp["strata"] = got
                ctx = (st.get("context") or {}).get("max_positions") or st.get("cache_max_tokens")
                if isinstance(ctx, int) and ctx > 0:
                    stamp["strata_ctx"] = ctx                # what Strata was started with (--max-context)
            return True, "", stamp

        def _bigdoc(self, raw, body, streaming, threshold, label):
            quiet = label == "auto"                      # a normal chat: no log line unless it is big enough
            if renderer is None:
                if not quiet:
                    log(f"{label}: no renderer (pass-through-only mode) -> pass through")
                return self._relay(raw, streaming, False)
            try:
                prepared = renderer.prepare(body)
            except Exception as e:
                if not quiet:
                    log(f"{label}: render failed ({type(e).__name__}: {e}) -> pass through")
                return self._relay(raw, streaming, False)
            if prepared is None:
                if not quiet:
                    log(f"{label}: not a request Sushi would render as modelled -> pass through")
                return self._relay(raw, streaming, False)
            ids = prepared.ids
            if len(ids) < threshold:                     # cannot have threshold new tokens: no cache lookup needed
                if not quiet:
                    log(f"{label}: prompt {len(ids)} < {threshold} -> pass through")
                return self._relay(raw, streaming, False)
            if prepared.thinking and not think_handoff:
                log(f"{label}: thinking on (THINK_HANDOFF off) -> pass through")
                return self._relay(raw, streaming, False)
            try:
                root = cache_root_fn() if cache_root_fn else cfg.cache_root
                # Strata's context is checked after the identity check, against what Strata itself reports
                rcfg = dataclasses.replace(cfg, cache_root=root, min_new_tokens=threshold, strata_max_ctx=1 << 62)
                restorable = best_restore(root, ids, prepared.has_tools)
                go, why = handoff.decide(len(ids), restorable, rcfg)
            except Exception as e:
                log(f"{label}: cache lookup failed ({type(e).__name__}: {e}) -> pass through")
                return self._relay(raw, streaming, False)
            log(f"{label}: prompt {len(ids)}, restorable {restorable}: {'HANDOFF' if go else 'pass'} ({why})")
            if not go:
                return self._relay(raw, streaming, False)
            ok, reason = self._sushi_is_production()
            if ok:
                ok, reason, stamp = self._identity()
            if ok:
                strata_ctx = stamp.get("strata_ctx", cfg.strata_max_ctx)
                if len(ids) >= strata_ctx:
                    ok, reason = False, f"prompt {len(ids)} >= Strata's context {strata_ctx}"
            if ok:
                ok, reason = self._sushi_tokenizes_alike(prepared)
            if not ok:
                log(f"{label}: not handing off, {reason} -> pass through")
                return self._relay(raw, streaming, False)
            effort = prepared.strata_body["chat_template_kwargs"]
            log(f"{label}: Sushi tokenizer agrees; thinking {'on, ' + effort['reasoning_effort'] if prepared.thinking else 'off'}"
                f"{', tools' if prepared.has_tools else ''}")
            stamp.update(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), prompt_tokens=len(ids), has_tools=prepared.has_tools,
                         template_kwargs=effort, render=getattr(renderer, "identity", None),
                         converter=file_sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strata_to_sushi.py")))
            headers_sent = False
            state = {"step": "waiting for another handoff", "done": False, "err": None, "t": None, "skip": None}

            def work():
                try:
                    with one_handoff:                    # one handoff at a time; other requests are never held here
                        again = best_restore(root, ids, prepared.has_tools)
                        go2, why2 = handoff.decide(len(ids), again, rcfg)
                        if not go2:                      # a handoff that finished while we waited covers this prompt
                            state["skip"] = f"restorable {again} now ({why2})"
                            return
                        state["t"] = handoff.run(prepared.strata_body, ids, rcfg, steps, gate,
                                                 lambda s: state.update(step=s), has_tools=prepared.has_tools, stamp=stamp)
                except Exception as e:
                    state["err"] = e
                finally:
                    state["done"] = True

            th = threading.Thread(target=work, daemon=True)
            th.start()
            alive = True
            if streaming:
                try:
                    self._sse_headers()
                    headers_sent = True
                except (BrokenPipeError, ConnectionResetError):
                    alive = False
            while not state["done"]:
                th.join(keepalive_s)
                if streaming and alive and not state["done"]:
                    try:
                        self._chunk(f": handoff {state['step']}\n\n".encode())
                    except (BrokenPipeError, ConnectionResetError):
                        alive = False          # the handoff still runs to the end; only the relay is skipped
            err = state["err"]
            if isinstance(err, handoff.SushiDown):
                log(f"{label}: SUSHI DOWN: {err}")
                msg = json.dumps({"error": f"Sushi did not come back after the handoff restart: {err}"}).encode()
                if headers_sent and alive:
                    self._chunk(b"data: " + msg + b"\n\n"); self._chunk(b"")
                elif alive:
                    self._send(503, msg)
                return
            expect = None
            if err is not None:
                log(f"{label}: handoff failed, passing through: {err}")
            elif state["skip"]:
                log(f"{label}: no handoff after the wait, {state['skip']} -> pass through")
            else:
                log(f"{label}: handoff done {state['t']}")
                expect = len(ids) - 8      # Strata's checkpoint sits ~7 tokens before the prompt end
            if alive:
                self._relay(raw, streaming, headers_sent, expect_cached=expect)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    # Sushi's cache folder for this model pack (the fingerprint is per pack), and the log it reports it in
    default_root = os.path.expanduser(os.environ.get("KVH_SUSHI_CACHE_ROOT", "~/.sushi/kv-cache/526f67face43a3d8"))
    sushi_log = os.path.expanduser(os.environ.get(
        "KVH_SUSHI_LOG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sushi-server.log")))
    port = int(os.environ.get("KVH_PROXY_PORT", "8002"))
    root = find_cache_root(sushi_log, default_root)
    # A test instance (KVH_FORCE_ALL=1) needs its own port, log and incoming folder (same disk as the cache root: the
    # entry is renamed into it), so it never cleans or races the production proxy's.
    incoming = os.path.expanduser(os.environ.get("KVH_INCOMING", "~/.sushi/kvh-incoming"))
    force_all = os.environ.get("KVH_FORCE_ALL") == "1"
    os.makedirs(incoming, exist_ok=True)
    removed = clean_incoming(incoming)
    if removed:
        log(f"removed leftovers from a previous crash in {incoming}: {', '.join(removed)}")
    try:
        from render import Renderer
        renderer = Renderer()
    except Exception as e:
        log(f"Renderer failed ({type(e).__name__}: {e}); running pass-through only, no handoffs")
        renderer = None
    cfg = handoff.Config(cache_root=root, incoming=incoming, min_new_tokens=AUTO_MIN_NEW_TOKENS)   # bigdoc alias: same
    if os.environ.get("KVH_MIN_NEW_TOKENS"):
        cfg = dataclasses.replace(cfg, min_new_tokens=int(os.environ["KVH_MIN_NEW_TOKENS"]))
    srv = make_server(port, handoff.SUSHI_URL, renderer, cfg, handoff.real_steps(),
                      cache_root_fn=lambda: find_cache_root(sushi_log, default_root), force_all=force_all,
                      auto_handoff=AUTO_HANDOFF and not force_all, sushi_version_fn=lambda: sushi_version(sushi_log),
                      strata_status_fn=strata_status)
    log(f"proxy up on 127.0.0.1:{port}, cache root {root}, Sushi {sushi_version(sushi_log)}" +
        (f" - TEST INSTANCE: every chat request is a candidate, min_new_tokens {cfg.min_new_tokens}" if force_all else
         f", automatic handoff at {AUTO_MIN_NEW_TOKENS} new tokens" if AUTO_HANDOFF else ""))
    srv.serve_forever()


if __name__ == "__main__":
    main()
