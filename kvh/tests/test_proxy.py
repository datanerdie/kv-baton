import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import handoff
import handoff_proxy
from gate import Gate


class FakeSushi:
    """Upstream stand-in: answers chat completions (JSON or SSE) and /health, records request bodies."""
    def __init__(self):
        self.bodies, self.delay = [], 0.0
        self.models = {"data": [{"id": handoff_proxy.EXPECTED_SUSHI_MODEL, "context_length": handoff_proxy.EXPECTED_SUSHI_CTX}]}
        self.models_status = 200                      # GET /v1/models: configurable document and status
        self.reply, self.break_stream = None, False   # reply: raw non-streaming bytes; break_stream: die mid-SSE
        outer = self
        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def do_GET(self):
                b, status = b'{"status":"ok"}', 200
                if self.path == "/v1/models":
                    b, status = json.dumps(outer.models).encode(), outer.models_status
                self.send_response(status); self.send_header("Content-Length", str(len(b))); self.end_headers()
                self.wfile.write(b)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body); time.sleep(outer.delay)
                if body.get("stream"):
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                    if outer.break_stream:
                        line = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line)); self.wfile.flush()
                        time.sleep(0.2)                   # let the proxy read the line, then reset the connection (RST)
                        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                        self.close_connection = True
                        return
                    for line in (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n', b"\n", b"data: [DONE]\n", b"\n"):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    b = outer.reply or json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"


class FakeRenderer:
    def __init__(self, n): self.n = n
    def render_ids(self, body): return list(range(self.n))


def steps(log, n, slow=0.0, fail=None):
    def mk(name, ret=None):
        def f(*a):
            log.append(name); time.sleep(slow)
            if fail == name:
                raise (handoff.SushiDown if name == "restart" else RuntimeError)(name)
            return ret
        return f
    def dump(session, d):
        import os
        from array import array
        log.append("dump"); os.makedirs(d); array("i", range(n)).tofile(open(os.path.join(d, "live_ids.i32"), "wb"))
    def convert(d, out):
        import os
        log.append("convert"); os.makedirs(out)
    return handoff.Steps(mk("prefill", n), mk("save"), dump, convert, mk("restart"))


@pytest.fixture
def setup(tmp_path):
    root, inc = tmp_path / "root", tmp_path / "inc"
    root.mkdir(); inc.mkdir()
    cfg = handoff.Config(cache_root=str(root), incoming=str(inc), min_new_tokens=100, strata_max_ctx=100000)
    up = FakeSushi()
    def start(n_tokens, st, keepalive_s=10.0, **kw):
        srv = handoff_proxy.make_server(0, up.url, FakeRenderer(n_tokens), cfg, st, keepalive_s=keepalive_s, **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{srv.server_address[1]}", srv
    return up, start


def post(url, body, raw=False, timeout=30):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if raw else json.loads(data)


def test_normal_alias_passes_through_untouched(setup):
    up, start = setup
    log = []; url, _ = start(5000, steps(log, 5000))
    b = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "x"}], "temperature": 0.7}
    assert post(url, b)["choices"][0]["message"]["content"] == "hi"
    assert up.bodies == [b] and log == []


def test_streaming_pass_through(setup):
    up, start = setup
    url, _ = start(10, steps([], 10))
    raw = post(url, {"model": "m", "messages": [], "stream": True}, raw=True)
    assert b'"content":"hi"' in raw and b"[DONE]" in raw


def test_bigdoc_small_prompt_passes_through(setup):
    up, start = setup
    log = []; url, _ = start(50, steps(log, 50))              # 50 new tokens < 100
    post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]})
    assert log == [] and len(up.bodies) == 1


def test_bigdoc_big_prompt_hands_off_then_relays(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    out = post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]})
    assert log == ["prefill", "save", "dump", "convert", "restart"]
    assert out["choices"][0]["message"]["content"] == "hi" and len(up.bodies) == 1


def test_handoff_failure_still_answers(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500, fail="save"))
    out = post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]})
    assert out["choices"][0]["message"]["content"] == "hi" and "restart" not in log


def test_streaming_keepalive_during_handoff(setup):
    up, start = setup
    url, _ = start(500, steps([], 500, slow=0.15), keepalive_s=0.05)
    raw = post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}], "stream": True}, raw=True)
    assert raw.index(b": handoff") < raw.index(b"data:")


def test_sushi_down_answers_503(setup):
    up, start = setup
    url, _ = start(500, steps([], 500, fail="restart"))
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]})
    assert e.value.code == 503


def test_sushi_unreachable_answers_502_not_hang(tmp_path):
    cfg = handoff.Config(cache_root=str(tmp_path), incoming=str(tmp_path), min_new_tokens=100)
    srv = handoff_proxy.make_server(0, "http://127.0.0.1:9", FakeRenderer(10), cfg, steps([], 10))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"http://127.0.0.1:{srv.server_address[1]}", {"model": "m", "messages": []}, timeout=10)
    assert e.value.code == 502 and b"Sushi unreachable" in e.value.read()


def test_client_disconnect_mid_handoff_still_restarts(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500, slow=0.2))
    with pytest.raises(Exception):
        post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]}, timeout=0.3)
    time.sleep(1.5)
    assert log[-1] == "restart"


def test_second_bigdoc_waits_for_the_first(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500, slow=0.1))
    b = {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]}
    t = threading.Thread(target=post, args=(url, b)); t.start()
    time.sleep(0.05)
    post(url, b); t.join()
    assert log.count("restart") <= 2 and log[:5] == ["prefill", "save", "dump", "convert", "restart"]


BIG = {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]}


def test_bigdoc_thinking_on_passes_through(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, dict(BIG, chat_template_kwargs={"enable_thinking": True}))
    assert log == [] and len(up.bodies) == 1


def test_bigdoc_reasoning_effort_passes_through_even_with_thinking_off(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, dict(BIG, chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "high"}))
    assert log == [] and len(up.bodies) == 1


def test_bigdoc_thinking_off_still_hands_off(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, dict(BIG, chat_template_kwargs={"enable_thinking": False}))
    assert log == ["prefill", "save", "dump", "convert", "restart"]


def test_bigdoc_non_dict_chat_template_kwargs_passes_through(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    out = post(url, dict(BIG, chat_template_kwargs="enable_thinking"))
    assert out["choices"][0]["message"]["content"] == "hi" and log == [] and len(up.bodies) == 1


def test_renderer_exception_passes_through(setup, tmp_path):
    up, _ = setup
    class Boom:
        def render_ids(self, body): raise ValueError("template exploded")
    cfg = handoff.Config(cache_root=str(tmp_path), incoming=str(tmp_path), min_new_tokens=100)
    log = []
    srv = handoff_proxy.make_server(0, up.url, Boom(), cfg, steps(log, 500))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = post(f"http://127.0.0.1:{srv.server_address[1]}", BIG)
    assert out["choices"][0]["message"]["content"] == "hi" and log == [] and len(up.bodies) == 1


def test_missing_cache_root_passes_through_and_logs(setup, tmp_path, monkeypatch):
    up, _ = setup
    logfile = tmp_path / "handoff_proxy-test.log"      # the autouse fixture in conftest points LOG here
    cfg = handoff.Config(cache_root=str(tmp_path / "no-such-dir"), incoming=str(tmp_path), min_new_tokens=100)
    log = []
    srv = handoff_proxy.make_server(0, up.url, FakeRenderer(500), cfg, steps(log, 500))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = post(f"http://127.0.0.1:{srv.server_address[1]}", BIG)
    assert out["choices"][0]["message"]["content"] == "hi" and log == [] and len(up.bodies) == 1
    assert "cache lookup failed" in logfile.read_text()


@pytest.mark.parametrize("reply", [
    b'[1, 2]',
    b'{"usage": {"prompt_tokens_details": {"cached_tokens": "x"}}, "choices": [{"message": {"content": "hi"}}]}',
])
def test_unreadable_usage_after_handoff_still_returns_reply(setup, reply):
    up, start = setup
    up.reply = reply
    log = []; url, _ = start(500, steps(log, 500))
    raw = post(url, BIG, raw=True)
    assert raw == reply and log[-1] == "restart"


def test_upstream_stream_breaking_midway_ends_cleanly_with_error_event(setup):
    up, start = setup
    up.break_stream = True
    url, _ = start(10, steps([], 10))
    raw = post(url, {"model": "m", "messages": [], "stream": True}, raw=True, timeout=10)
    assert b'"content":"hi"' in raw and b'"error"' in raw



def _models(model, ctx):
    return {"data": [{"id": model, "context_length": ctx}]}


@pytest.mark.parametrize("models, status, reason", [
    (_models("Qwen3.8-Flash-Next-other", 1048576), 200, "not 'Qwen3.8-Flash-Next-Sushi-4bpw'"),
    (_models("Qwen3.8-Flash-Next-Sushi-4bpw", 262144), 200, "ctx 262144"),
    ({"error": "nope"}, 500, "/v1/models failed"),
    ({"data": []}, 200, "/v1/models failed"),
])
def test_bigdoc_passes_through_unless_production_sushi(setup, tmp_path, models, status, reason):
    up, start = setup
    up.models, up.models_status = models, status
    log = []; url, _ = start(500, steps(log, 500))
    out = post(url, BIG)
    assert out["choices"][0]["message"]["content"] == "hi" and log == [] and len(up.bodies) == 1
    assert reason in (tmp_path / "handoff_proxy-test.log").read_text()


def test_bigdoc_hands_off_when_models_match(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, BIG)
    assert log == ["prefill", "save", "dump", "convert", "restart"]


@pytest.mark.parametrize("extra", [{"reasoning_effort": "high"}, {"reasoning": {"effort": "high"}}])
def test_bigdoc_top_level_reasoning_passes_through(setup, extra):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, dict(BIG, **extra))
    assert log == [] and len(up.bodies) == 1


def test_renderer_none_is_pass_through_only(setup):
    up, _ = setup
    cfg = handoff.Config(cache_root="/nonexistent", incoming="/nonexistent", min_new_tokens=100)
    log = []
    srv = handoff_proxy.make_server(0, up.url, None, cfg, steps(log, 500))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = post(f"http://127.0.0.1:{srv.server_address[1]}", BIG)
    assert out["choices"][0]["message"]["content"] == "hi" and log == [] and len(up.bodies) == 1


def test_cache_root_fn_is_read_per_request(setup, tmp_path):
    from conftest import make_entry
    up, start = setup
    roots = []
    def root_fn():
        roots.append(1)
        return str(tmp_path / "root2")
    (tmp_path / "root2").mkdir()
    make_entry(tmp_path / "root2", 1, list(range(500)), [450])     # restorable 450 -> 50 new tokens < 100: pass
    log = []; url, _ = start(500, steps(log, 500), cache_root_fn=root_fn)
    post(url, BIG); post(url, BIG)
    assert log == [] and len(roots) == 2 and len(up.bodies) == 2


def test_cache_root_fn_root_reaches_handoff_run(setup, tmp_path, monkeypatch):
    up, start = setup
    other = tmp_path / "other"; other.mkdir()
    seen = []
    real_run = handoff.run
    def spy(body, ids, cfg, st, gate, progress):
        seen.append(cfg.cache_root); return real_run(body, ids, cfg, st, gate, progress)
    monkeypatch.setattr(handoff, "run", spy)
    log = []; url, _ = start(500, steps(log, 500), cache_root_fn=lambda: str(other))
    post(url, BIG)
    assert seen == [str(other)] and log[-1] == "restart"


def test_clean_incoming_removes_only_leftovers(tmp_path):
    for n in ("dump-abc", "e1000", "e7"):
        (tmp_path / n).mkdir(); (tmp_path / n / "f").write_text("x")
    for n in ("keep", "entries", "e12x"):
        (tmp_path / n).mkdir()
    (tmp_path / "dump-file").write_text("x")
    assert handoff_proxy.clean_incoming(str(tmp_path)) == ["dump-abc", "e1000", "e7"]
    assert sorted(os.listdir(tmp_path)) == ["dump-file", "e12x", "entries", "keep"]
    assert handoff_proxy.clean_incoming(str(tmp_path / "missing")) == []


def _capture_dir(tmp_path, monkeypatch, on):
    cap = tmp_path / "capture"
    cap.mkdir()
    if on:
        (cap / "ON").touch()
    monkeypatch.setattr(handoff_proxy, "CAPTURE_DIR", str(cap))
    return cap


def test_capture_saves_bigdoc_body_privately_when_switched_on(setup, tmp_path, monkeypatch):
    cap = _capture_dir(tmp_path, monkeypatch, on=True)
    up, start = setup
    url, _ = start(50, steps([], 50))
    b = {"model": "qwen38-flash-bigdoc", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": " x\n"}]}
    post(url, b)
    saved = cap / "last-bigdoc.json"
    assert json.loads(saved.read_text()) == b
    assert saved.stat().st_mode & 0o777 == 0o600


def test_capture_off_without_the_switch(setup, tmp_path, monkeypatch):
    cap = _capture_dir(tmp_path, monkeypatch, on=False)
    up, start = setup
    url, _ = start(50, steps([], 50))
    post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": "x"}]})
    assert not (cap / "last-bigdoc.json").exists()


def test_capture_ignores_normal_aliases(setup, tmp_path, monkeypatch):
    cap = _capture_dir(tmp_path, monkeypatch, on=True)
    up, start = setup
    url, _ = start(50, steps([], 50))
    post(url, {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "x"}]})
    assert not (cap / "last-bigdoc.json").exists()


def test_bigdoc_request_is_normalised_the_same_for_strata_and_sushi(setup):
    up, start = setup
    seen = []
    st = steps([], 500)
    st.strata_prefill = lambda b: (seen.append(b), 500)[1]
    url, _ = start(500, st)
    b = {"model": "qwen38-flash-bigdoc", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "[file name]: a.txt\n60 °C"}, {"type": "text", "text": "Read and summarise"}]}]}
    post(url, b)
    want = "[file name]: a.txt\n60 °C\nRead and summarise"
    assert seen and seen[0]["messages"][0]["content"] == want
    assert up.bodies[-1]["messages"][0]["content"] == want


def test_small_bigdoc_request_reaches_sushi_normalised_too(setup):
    up, start = setup
    url, _ = start(50, steps([], 50))
    post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": [{"type": "text", "text": "a b"}]}]})
    assert up.bodies[-1]["messages"][0]["content"] == "a b"


def test_normal_alias_keeps_content_arrays_and_narrow_spaces(setup):
    up, start = setup
    url, _ = start(50, steps([], 50))
    b = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": [{"type": "text", "text": "a b"}]}]}
    post(url, b)
    assert up.bodies[-1] == b


def test_image_parts_are_left_alone(setup):
    up, start = setup
    url, _ = start(50, steps([], 50))
    content = [{"type": "text", "text": "a b"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    post(url, {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": content}]})
    assert up.bodies[-1]["messages"][0]["content"] == content
