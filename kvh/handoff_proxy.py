"""Proxy in front of Sushi (:8000) on 127.0.0.1:8002. Every request is relayed unchanged, except chat
requests for `qwen38-flash-bigdoc*` with >= min_new_tokens uncached tokens: those are prefilled by Strata on the GPU box
first (handoff.run), then relayed, and Sushi answers from the injected cache entry. Design:
README.md and RESULTS.md at the repository root
"""
import dataclasses
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

# False: Sushi puts a server-side reasoning-effort system message in front of a thinking request that the local
# render cannot reproduce, so the restore point would not match; thinking requests pass through.
THINK_HANDOFF = False
BIGDOC_PREFIX = "qwen38-flash-bigdoc"
# A handoff restarts Sushi, so it is only done while the production Sushi pack is what is running (never on another
# engine, never to start a server the user stopped on purpose).
EXPECTED_SUSHI_MODEL = "Qwen3.8-Flash-Next-Sushi-4bpw"
EXPECTED_SUSHI_CTX = 1048576
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handoff_proxy.log")
_log_lock = threading.Lock()


def log(msg):
    with _log_lock, open(LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


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


def normalise_for_handoff(body):
    """(body, changed): the version of a bigdoc request that Strata, the render and Sushi all tokenise alike.

    Measured 2026-10-07 on a chat client's request: Sushi's tokenizer splits "°C" after U+202F (narrow no-break space)
    where the reference tokenizer keeps one token, and that client sends an attached file and the question as separate
    text parts, which Sushi joins with "\\n". So text-only content arrays become one "\\n"-joined string and U+202F
    becomes a plain space. Content with any non-text part (an image) is left alone.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return body, False
    out, changed = [], False
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        new = c
        if isinstance(new, list) and new and all(isinstance(p, dict) and p.get("type") == "text" for p in new):
            new = "\n".join(p.get("text") or "" for p in new)
        if isinstance(new, str):
            new = new.replace(" ", " ")
        if new is not c and new != c:
            m, changed = dict(m, content=new), True
        out.append(m)
    return (dict(body, messages=out), True) if changed else (body, False)


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


def make_server(port, upstream, renderer, cfg, steps, keepalive_s=10.0, think_handoff=THINK_HANDOFF, cache_root_fn=None):
    """renderer None: pass-through only (no handoffs). cache_root_fn: returns the current cache root, per request."""
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
                    log(f"bigdoc: Sushi restored {got} cached tokens (restore point {expect})"
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
            if not self.path.endswith("/chat/completions") or not model.startswith(BIGDOC_PREFIX):
                return self._relay(raw, streaming, False)
            capture(raw)
            body, changed = normalise_for_handoff(body)
            if changed:
                raw = json.dumps(body).encode()
                log("bigdoc: request normalised (text parts joined with \\n, U+202F -> space)")
            kwargs = body.get("chat_template_kwargs")
            if kwargs is not None and not isinstance(kwargs, dict):
                log("bigdoc: chat_template_kwargs is not an object -> pass through")
                return self._relay(raw, streaming, False)
            kwargs = kwargs or {}
            if "reasoning_effort" in body or "reasoning" in body:
                log("bigdoc: top-level reasoning_effort/reasoning set (Sushi injects a system message the render cannot match) -> pass through")
                return self._relay(raw, streaming, False)
            if "reasoning_effort" in kwargs:
                log("bigdoc: reasoning_effort set (Sushi injects a system message the render cannot match) -> pass through")
                return self._relay(raw, streaming, False)
            if kwargs.get("enable_thinking") and not think_handoff:
                log("bigdoc: thinking on (THINK_HANDOFF off) -> pass through")
                return self._relay(raw, streaming, False)
            with one_handoff:
                return self._bigdoc(raw, body, streaming)

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
            return True, ""

        def _bigdoc(self, raw, body, streaming):
            if renderer is None:
                log("bigdoc: no renderer (pass-through-only mode) -> pass through")
                return self._relay(raw, streaming, False)
            try:
                ids = renderer.render_ids(body)
            except Exception as e:
                log(f"bigdoc: render failed ({type(e).__name__}: {e}) -> pass through")
                return self._relay(raw, streaming, False)
            if ids is None:
                log("bigdoc: not a plain-text request -> pass through")
                return self._relay(raw, streaming, False)
            try:
                root = cache_root_fn() if cache_root_fn else cfg.cache_root
                rcfg = dataclasses.replace(cfg, cache_root=root)
                restorable = best_restore(root, ids)
                go, why = handoff.decide(len(ids), restorable, rcfg)
            except Exception as e:
                log(f"bigdoc: cache lookup failed ({type(e).__name__}: {e}) -> pass through")
                return self._relay(raw, streaming, False)
            log(f"bigdoc: prompt {len(ids)}, restorable {restorable}: {'HANDOFF' if go else 'pass'} ({why})")
            if not go:
                return self._relay(raw, streaming, False)
            ok, reason = self._sushi_is_production()
            if not ok:
                log(f"bigdoc: not handing off, {reason} -> pass through")
                return self._relay(raw, streaming, False)
            headers_sent = False
            state = {"step": "start", "done": False, "err": None, "t": None}

            def work():
                try:
                    state["t"] = handoff.run(body, ids, rcfg, steps, gate, lambda s: state.update(step=s))
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
                log(f"bigdoc: SUSHI DOWN: {err}")
                msg = json.dumps({"error": f"Sushi did not come back after the handoff restart: {err}"}).encode()
                if headers_sent and alive:
                    self._chunk(b"data: " + msg + b"\n\n"); self._chunk(b"")
                elif alive:
                    self._send(503, msg)
                return
            expect = None
            if err is not None:
                log(f"bigdoc: handoff failed, passing through: {err}")
            else:
                log(f"bigdoc: handoff done {state['t']}")
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
    incoming = os.path.expanduser("~/.sushi/kvh-incoming")
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
    cfg = handoff.Config(cache_root=root, incoming=incoming)
    srv = make_server(port, handoff.SUSHI_URL, renderer, cfg, handoff.real_steps(),
                      cache_root_fn=lambda: find_cache_root(sushi_log, default_root))
    log(f"proxy up on 127.0.0.1:{port}, cache root {root}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
