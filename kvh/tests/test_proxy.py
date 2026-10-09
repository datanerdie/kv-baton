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
from render import THINKING_FIELDS, Prepared, sushi_template_kwargs


class FakeSushi:
    """Upstream stand-in: answers chat completions (JSON or SSE) and /health, records request bodies."""
    def __init__(self):
        self.bodies, self.delay = [], 0.0
        self.models = {"data": [{"id": handoff_proxy.EXPECTED_SUSHI_MODEL, "context_length": handoff_proxy.EXPECTED_SUSHI_CTX}]}
        self.models_status = 200                      # GET /v1/models: configurable document and status
        self.reply, self.break_stream = None, False   # reply: raw non-streaming bytes; break_stream: die mid-SSE
        self.no_done = False                          # end the SSE stream cleanly but without data: [DONE]
        self.tokenize = None                          # /tokenize: None = echo the render's ids back, else this reply
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
                if self.path == "/tokenize":
                    if outer.tokenize == 404:
                        self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                    toks = outer.tokenize if outer.tokenize is not None else [int(x) for x in body["content"].split()]
                    b = json.dumps({"tokens": toks}).encode()
                    self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
                    self.wfile.write(b); return
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
                    lines = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n', b"\n", b"data: [DONE]\n", b"\n")
                    for line in lines[:2] if outer.no_done else lines:
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
    """n token ids, the real thinking resolution; the text is the ids, so FakeSushi's /tokenize can echo them."""
    def __init__(self, n): self.n = n
    def prepare_why(self, body):
        kw = sushi_template_kwargs(body)
        if kw is None:
            return None, "thinking settings"
        ids = list(range(body.get("_n", self.n)))        # a test can size one request with "_n"
        sbody = {k: v for k, v in body.items() if k not in THINKING_FIELDS}
        sbody["chat_template_kwargs"] = dict(kw)
        sbody["kvh_prompt_ids"] = ids
        return Prepared(ids, " ".join(map(str, ids)), sbody, kw["enable_thinking"], body.get("tools") is not None), None


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


def test_force_all_makes_any_model_a_candidate(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500), force_all=True)
    post(url, {"model": "Qwen3.8-Flash-Next-Sushi-4bpw", "messages": [{"role": "user", "content": "x"}]})
    assert log == ["prefill", "save", "dump", "convert", "restart"]


def test_without_force_all_other_models_pass_through(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, {"model": "Qwen3.8-Flash-Next-Sushi-4bpw", "messages": [{"role": "user", "content": "x"}]})
    assert log == [] and len(up.bodies) == 1


def test_bigdoc_thinking_on_hands_off(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, dict(BIG, chat_template_kwargs={"enable_thinking": True}))
    assert log == ["prefill", "save", "dump", "convert", "restart"]


def test_bigdoc_thinking_on_passes_through_with_the_kill_switch(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500), think_handoff=False)
    post(url, dict(BIG, chat_template_kwargs={"enable_thinking": True}))
    assert log == [] and len(up.bodies) == 1


def test_bigdoc_kwargs_reasoning_effort_is_ignored_like_sushi(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500))
    post(url, dict(BIG, chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "high"}))
    assert log == ["prefill", "save", "dump", "convert", "restart"]


def test_strata_gets_sushis_thinking_settings_spelled_out(setup):
    up, start = setup
    log, seen = [], {}
    st = steps(log, 500)
    st.strata_prefill = lambda b: (seen.update(b), log.append("prefill"), 500)[2]
    url, _ = start(500, st)
    b = dict(BIG, reasoning_effort="medium", chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh"})
    post(url, b)
    assert log[0] == "prefill" and "reasoning_effort" not in seen
    assert seen["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert up.bodies[-1]["reasoning_effort"] == "medium"          # Sushi still gets the client's request unchanged


def test_tools_request_hands_off_with_ids_and_the_tools_flag(setup, monkeypatch):
    up, start = setup
    log, seen, flags = [], {}, []
    real_run = handoff.run
    monkeypatch.setattr(handoff, "run", lambda *a, **kw: (flags.append(kw.get("has_tools")), real_run(*a, **kw))[1])
    st = steps(log, 500)
    st.strata_prefill = lambda b: (seen.update(b), log.append("prefill"), 500)[2]
    url, _ = start(500, st)
    post(url, dict(BIG, tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}]))
    assert log[0] == "prefill" and seen["kvh_prompt_ids"] == list(range(500)) and flags == [True]


@pytest.mark.parametrize("reply", [[1, 2, 3], 404])
def test_sushi_tokenizer_disagreeing_or_failing_passes_through(setup, reply):
    up, start = setup
    up.tokenize = reply
    log = []; url, _ = start(500, steps(log, 500))
    post(url, BIG)
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
        def prepare_why(self, body): raise ValueError("template exploded")
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


def test_upstream_stream_ending_without_done_gets_error_event(setup):
    # http.client can report a cut-off chunked stream as a clean end, so only the missing [DONE] shows it
    up, start = setup
    up.no_done = True
    url, _ = start(10, steps([], 10))
    raw = post(url, {"model": "m", "messages": [], "stream": True}, raw=True, timeout=10)
    assert b'"content":"hi"' in raw and b'"error"' in raw


def test_complete_stream_gets_no_error_event(setup):
    up, start = setup
    url, _ = start(10, steps([], 10))
    raw = post(url, {"model": "m", "messages": [], "stream": True}, raw=True, timeout=10)
    assert raw.rstrip().endswith(b"data: [DONE]") and b'"error"' not in raw



def _models(model, ctx):
    return {"data": [{"id": model, "context_length": ctx}]}


@pytest.mark.parametrize("models, status, reason", [
    (_models("Qwen3.8-Flash-Next-other", 1048576), 200, "not 'Qwen3.8-Flash-Next-Sushi-4bpw'"),
    (_models("Qwen3.8-Flash-Next-Sushi-4bpw", 262144), 200, "ctx 262144"),
    ({"error": "nope"}, 500, "/v1/models failed"),
    ({"data": []}, 200, "/v1/models failed"),
    ({"data": [{"id": "Qwen3.8-Flash-Next-Sushi-4bpw", "context_length": 1048576, "default_reasoning_effort": "low"}]},
     200, "default reasoning effort is 'low'"),
])
def test_bigdoc_passes_through_unless_production_sushi(setup, tmp_path, models, status, reason):
    up, start = setup
    up.models, up.models_status = models, status
    log = []; url, _ = start(500, steps(log, 500))
    out = post(url, BIG)
    assert out["choices"][0]["message"]["content"] == "hi" and log == [] and len(up.bodies) == 1
    assert reason in (tmp_path / "handoff_proxy-test.log").read_text()


@pytest.mark.parametrize("effort", [None, "off"])
def test_bigdoc_hands_off_when_models_match(setup, effort):
    up, start = setup
    if effort:
        up.models["data"][0]["default_reasoning_effort"] = effort      # what sushi 1.2.0 reports without --think
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
    def spy(body, ids, cfg, st, gate, progress, **kw):
        seen.append(cfg.cache_root); return real_run(body, ids, cfg, st, gate, progress, **kw)
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


def test_requests_reach_sushi_unchanged_and_strata_gets_token_ids(setup):
    """No rewriting any more (2026-10-08): Sushi joins text parts itself and 1.2.0 tokenises U+202F like the render."""
    up, start = setup
    seen = []
    st = steps([], 500)
    st.strata_prefill = lambda b: (seen.append(b), 500)[1]
    url, _ = start(500, st)
    b = {"model": "qwen38-flash-bigdoc", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "[file name]: a.txt\n60\u202f°C"}, {"type": "text", "text": "Read"}]}]}
    post(url, b)
    assert seen and seen[0]["kvh_prompt_ids"] == list(range(500))
    assert up.bodies[-1] == b


AUTO = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "x"}]}


def test_auto_handoff_on_a_normal_alias_above_its_threshold(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500), auto_handoff=True, auto_min_new_tokens=300)
    post(url, AUTO)
    assert log == ["prefill", "save", "dump", "convert", "restart"]


def test_auto_small_requests_pass_without_a_log_line(setup, tmp_path):
    up, start = setup
    log = []; url, _ = start(100, steps(log, 100), auto_handoff=True, auto_min_new_tokens=300)
    post(url, AUTO)
    text = (tmp_path / "handoff_proxy-test.log").read_text() if (tmp_path / "handoff_proxy-test.log").exists() else ""
    assert log == [] and len(up.bodies) == 1 and "auto:" not in text


def test_auto_skips_from_the_logging_size_on_are_logged(setup, tmp_path, monkeypatch):
    up, start = setup
    monkeypatch.setattr(handoff_proxy, "LOG_SKIPS_FROM_TOKENS", 50)
    log = []; url, _ = start(100, steps(log, 100), auto_handoff=True, auto_min_new_tokens=300)
    post(url, AUTO)                                                  # 100 tokens: under the threshold, over the size
    post(url, dict(AUTO, reasoning_effort="bogus"))                  # not modelled, but its body is under 4 x 50 bytes
    post(url, dict(AUTO, reasoning_effort="bogus", messages=[{"role": "user", "content": "x" * 300}]))
    text = (tmp_path / "handoff_proxy-test.log").read_text()
    assert log == [] and len(up.bodies) == 3
    assert "auto: prompt 100 < 300 -> pass through" in text
    assert text.count("not modelled") == 1 and "auto: not modelled (thinking settings)" in text


def test_without_auto_normal_aliases_are_not_candidates(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500), auto_min_new_tokens=300)
    post(url, AUTO)
    assert log == [] and len(up.bodies) == 1


def test_small_requests_are_not_held_while_a_handoff_runs(setup):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500, slow=0.4), auto_handoff=True, auto_min_new_tokens=300)
    t = threading.Thread(target=post, args=(url, BIG)); t.start()
    time.sleep(0.2)                                      # the handoff is in its first step now
    t0 = time.time()
    post(url, {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}], "_n": 10})
    took = time.time() - t0
    t.join()
    assert took < 0.4 and log[:2] == ["prefill", "save"]


@pytest.mark.parametrize("version, status, reason", [
    ("1.3.0", {"model": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "engine": "0.1.40"}, "Sushi version '1.3.0'"),
    (None, {"model": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "engine": "0.1.40"}, "Sushi version None"),
    ("1.2.0", {"model": "qwen3.8-flash-next-other", "engine": "0.1.40"}, "Strata serves"),
    ("1.2.0", {"model": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "engine": "0.1.42"}, "Strata serves"),
])
def test_identity_mismatch_passes_through(setup, tmp_path, version, status, reason):
    up, start = setup
    log = []; url, _ = start(500, steps(log, 500), sushi_version_fn=lambda: version, strata_status_fn=lambda: status)
    post(url, BIG)
    assert log == [] and len(up.bodies) == 1
    assert reason in (tmp_path / "handoff_proxy-test.log").read_text()


def test_identity_ok_hands_off_and_stamps_the_entry(setup, tmp_path):
    up, start = setup
    status = {"model": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "engine": "0.1.40", "loaded": True}
    log = []; url, _ = start(500, steps(log, 500), sushi_version_fn=lambda: "1.2.0", strata_status_fn=lambda: status)
    post(url, BIG)
    assert log[-1] == "restart"
    stamps = list((tmp_path / "root").glob("e*/kvh.json"))
    assert len(stamps) == 1
    st = json.loads(stamps[0].read_text())
    assert st["sushi_version"] == "1.2.0" and st["strata"] == {"model": status["model"], "engine": "0.1.40"}
    assert st["prompt_tokens"] == 500 and st["has_tools"] is False and "converter" in st


@pytest.mark.parametrize("engine", ["0.1.40", "0.1.41"])
def test_every_validated_strata_engine_hands_off(setup, tmp_path, engine):
    up, start = setup
    status = {"model": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "engine": engine, "loaded": True}
    log = []; url, _ = start(500, steps(log, 500), sushi_version_fn=lambda: "1.2.0", strata_status_fn=lambda: status)
    post(url, BIG)
    assert log[-1] == "restart"
    st = json.loads(next((tmp_path / "root").glob("e*/kvh.json")).read_text())
    assert st["strata"] == {"model": status["model"], "engine": engine}


def test_a_handoff_that_finished_during_the_wait_is_not_repeated(setup, tmp_path, monkeypatch):
    up, start = setup
    calls = []
    def fake_best_restore(root, ids, has_tools=False):
        calls.append(1)
        return 0 if len(calls) == 1 else len(ids) - 8     # the second look (after the lock) finds a fresh entry
    monkeypatch.setattr(handoff_proxy, "best_restore", fake_best_restore)
    log = []; url, _ = start(500, steps(log, 500))
    post(url, BIG)
    assert log == [] and len(up.bodies) == 1
    assert "no handoff after the wait" in (tmp_path / "handoff_proxy-test.log").read_text()


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


@pytest.mark.parametrize("ctx, handed_off", [(131072, False), (524288, True)])
def test_strata_context_comes_from_its_status(setup, tmp_path, ctx, handed_off):
    up, start = setup
    status = {"model": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "engine": "0.1.40", "context": {"max_positions": ctx}}
    log = []; url, srv = start(200_000, steps(log, 200_000), sushi_version_fn=lambda: "1.2.0", strata_status_fn=lambda: status)
    post(url, BIG)
    assert (log[:1] == ["prefill"]) is handed_off
    if not handed_off:
        assert "prompt 200000 >= Strata's context 131072" in (tmp_path / "handoff_proxy-test.log").read_text()
