import dataclasses
import json
import os
from array import array

import pytest

import handoff
from conftest import make_entry
from gate import Gate


def cfg(tmp_path):
    root, inc = tmp_path / "root", tmp_path / "incoming"
    root.mkdir(); inc.mkdir()
    return handoff.Config(cache_root=str(root), incoming=str(inc), min_new_tokens=100, strata_max_ctx=1000)


@pytest.mark.parametrize("n,restorable,want", [
    (500, 0, True),      # 500 new >= 100
    (500, 450, False),   # only 50 new
    (1000, 0, False),    # at Strata's limit
    (2000, 0, False),    # over it
])
def test_decide(tmp_path, n, restorable, want):
    assert handoff.decide(n, restorable, cfg(tmp_path))[0] is want


@pytest.mark.parametrize("n,restorable,want", [
    (900, 0, True),      # cold: all new
    (900, 600, True),    # 300 new = 33% >= 25%
    (900, 700, False),   # 200 new >= 100 but 22% < 25%
])
def test_decide_fraction(tmp_path, n, restorable, want):
    c = dataclasses.replace(cfg(tmp_path), min_new_fraction=0.25)
    assert handoff.decide(n, restorable, c)[0] is want


def fake_steps(log, prompt_ids, fail_at=None, restart=None):
    def step(name, fn):
        def run(*a):
            log.append(name)
            if fail_at == name:
                raise RuntimeError(f"{name} failed")
            return fn(*a)
        return run
    def dump(session, d):
        os.makedirs(d)
        array("i", prompt_ids).tofile(open(os.path.join(d, "live_ids.i32"), "wb"))
    def convert(d, out):
        os.makedirs(out); open(os.path.join(out, "meta.json"), "w").write("{}")
    return handoff.Steps(strata_prefill=step("prefill", lambda b: len(prompt_ids)),
                         strata_save=step("save", lambda n: None), fetch_dump=step("dump", dump),
                         convert=step("convert", convert), restart_sushi=step("restart", restart or (lambda: None)))


def test_run_happy_path(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    make_entry(c.cache_root, 5, [1], [])
    t = handoff.run({"model": "qwen38-flash-bigdoc", "messages": [], "stream": True}, ids, c,
                    fake_steps(log, ids), Gate(), lambda s: None)
    assert log == ["prefill", "save", "dump", "convert", "restart"]
    assert os.path.isdir(os.path.join(c.cache_root, "e1005"))     # renamed in before the restart
    assert os.listdir(c.incoming) == []                            # nothing left behind
    assert set(t) >= {"prefill", "save", "dump", "convert", "restart"}


@pytest.mark.parametrize("has_tools", [False, True])
def test_has_tools_reaches_the_entry(tmp_path, has_tools):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    make_entry(c.cache_root, 5, [1], [])
    handoff.run({"model": "qwen38-flash-bigdoc", "messages": []}, ids, c, fake_steps(log, ids), Gate(), lambda s: None,
                has_tools=has_tools)
    meta = json.load(open(os.path.join(c.cache_root, "e1005", "meta.json")))
    assert meta.get("has_tools", False) is has_tools


def test_strata_request_is_prefill_only(tmp_path):
    c, ids, seen = cfg(tmp_path), list(range(300)), {}
    s = fake_steps([], ids)
    s.strata_prefill = lambda b: (seen.update(b), len(ids))[1]
    handoff.run({"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True,
                 "stream_options": {"include_usage": True}, "max_tokens": 9000, "temperature": 0.7,
                 "max_completion_tokens": 8000, "n": 1, "n_predict": 100, "logprobs": 5, "top_logprobs": 3,
                 "chat_template_kwargs": {"enable_thinking": False}}, ids, c, s, Gate(), lambda m: None)
    assert seen["max_tokens"] == 1 and seen["temperature"] == 0 and seen["stream"] is False
    assert "stream_options" not in seen and seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert "max_completion_tokens" not in seen and "n" not in seen and "n_predict" not in seen
    assert "logprobs" not in seen and "top_logprobs" not in seen


@pytest.mark.parametrize("fail_at", ["prefill", "save", "dump", "convert"])
def test_failure_cleans_up_and_never_restarts(tmp_path, fail_at):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    with pytest.raises(handoff.HandoffError):
        handoff.run({"messages": []}, ids, c, fake_steps(log, ids, fail_at), Gate(), lambda s: None)
    assert "restart" not in log
    assert os.listdir(c.incoming) == [] and os.listdir(c.cache_root) == []


def test_token_mismatch_is_refused(tmp_path):
    c, log = cfg(tmp_path), []
    with pytest.raises(handoff.HandoffError, match="token"):
        handoff.run({"messages": []}, list(range(300)), c, fake_steps(log, list(range(1, 301))), Gate(),
                    lambda s: None)
    assert "restart" not in log and os.listdir(c.cache_root) == []


def test_busy_sushi_falls_back_without_restart(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    c.idle_wait_s = 0.1
    g = Gate(); g.enter()                       # a request stays in flight
    with pytest.raises(handoff.HandoffError, match="busy"):
        handoff.run({"messages": []}, ids, c, fake_steps(log, ids), g, lambda s: None)
    assert "restart" not in log and os.listdir(c.cache_root) == []


def test_sushi_not_coming_back_raises_sushi_down(tmp_path):
    c, ids = cfg(tmp_path), list(range(300))
    restart_count = [0]
    def dead():
        restart_count[0] += 1
        raise handoff.SushiDown("sushi did not come back")
    with pytest.raises(handoff.SushiDown):
        handoff.run({"messages": []}, ids, c, fake_steps([], ids, restart=dead), Gate(), lambda s: None)
    assert restart_count[0] == 2
    assert os.listdir(c.cache_root) == []


def test_sushi_down_first_succeeds_second(tmp_path):
    c, ids = cfg(tmp_path), list(range(300))
    restart_count = [0]
    def sushidown_then_success():
        restart_count[0] += 1
        if restart_count[0] == 1:
            raise handoff.SushiDown("first attempt failed")
    with pytest.raises(handoff.HandoffError, match="entry removed"):
        handoff.run({"messages": []}, ids, c, fake_steps([], ids, restart=sushidown_then_success), Gate(), lambda s: None)
    assert restart_count[0] == 2
    assert os.listdir(c.cache_root) == []


def test_restart_fails_once_then_succeeds(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    restart_count = [0]
    def flaky_restart():
        restart_count[0] += 1
        log.append("restart")
        if restart_count[0] == 1:
            raise RuntimeError("first attempt failed")
    with pytest.raises(handoff.HandoffError, match="entry removed"):
        handoff.run({"messages": []}, ids, c, fake_steps(log, ids, restart=flaky_restart), Gate(), lambda s: None)
    assert restart_count[0] == 2
    assert os.listdir(c.cache_root) == []


def test_restart_fails_twice(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    restart_count = [0]
    def always_fails():
        restart_count[0] += 1
        log.append("restart")
        raise RuntimeError(f"attempt {restart_count[0]} failed")
    with pytest.raises(handoff.SushiDown, match="failed twice"):
        handoff.run({"messages": []}, ids, c, fake_steps(log, ids, restart=always_fails), Gate(), lambda s: None)
    assert restart_count[0] == 2
    assert os.listdir(c.cache_root) == []


def test_non_sushidown_exception_from_restart_handled_same_as_failure(tmp_path):
    c, ids = cfg(tmp_path), list(range(300))
    restart_count = [0]
    def fails_with_runtime_error():
        restart_count[0] += 1
        raise RuntimeError("generic error")
    with pytest.raises(handoff.SushiDown, match="failed twice"):
        handoff.run({"messages": []}, ids, c, fake_steps([], ids, restart=fails_with_runtime_error), Gate(), lambda s: None)
    assert restart_count[0] == 2
    assert os.listdir(c.cache_root) == []


def test_missing_live_ids_file(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    def dump_without_ids(session, d):
        os.makedirs(d)  # create dir but don't write live_ids.i32
    s = fake_steps(log, ids)
    s.fetch_dump = lambda session, d: dump_without_ids(session, d)
    with pytest.raises(handoff.HandoffError):
        handoff.run({"messages": []}, ids, c, s, Gate(), lambda s: None)
    assert "restart" not in log
    assert os.listdir(c.incoming) == [] and os.listdir(c.cache_root) == []


def test_missing_cache_root_directory(tmp_path):
    c = handoff.Config(cache_root=str(tmp_path / "nonexistent"), incoming=str(tmp_path / "incoming"),
                       min_new_tokens=100, strata_max_ctx=1000)
    (tmp_path / "incoming").mkdir()
    ids = list(range(300))
    with pytest.raises(handoff.HandoffError):
        handoff.run({"messages": []}, ids, c, fake_steps([], ids), Gate(), lambda s: None)
    assert os.listdir(tmp_path / "incoming") == []


def _real_restart(monkeypatch, returncode, health=True):
    calls = []
    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return handoff.subprocess.CompletedProcess(cmd, returncode)
    monkeypatch.setattr(handoff.subprocess, "run", fake_run)
    monkeypatch.setattr(handoff, "_health", lambda t: health)
    return handoff.real_steps().restart_sushi, calls


def test_real_restart_runs_the_restart_command_in_its_own_session(monkeypatch):
    restart, calls = _real_restart(monkeypatch, 0)
    restart()
    assert calls[0][0] == handoff.RESTART_CMD.split() and calls[0][1]["start_new_session"] is True


def test_real_restart_nonzero_exit_is_sushi_down(monkeypatch):
    restart, _ = _real_restart(monkeypatch, 3)
    with pytest.raises(handoff.SushiDown, match="exited 3"):
        restart()


def test_runtime_import_replaces_the_restart(tmp_path):
    c, log, ids, seen = cfg(tmp_path), [], list(range(300)), []
    make_entry(c.cache_root, 5, [1], [])
    s = fake_steps(log, ids)
    s.import_entry = lambda eid: (seen.append(eid), True)[1]
    t = handoff.run({"messages": []}, ids, c, s, Gate(), lambda m: None)
    assert seen == [1005] and "restart" not in log
    assert os.path.isdir(os.path.join(c.cache_root, "e1005")) and os.listdir(c.incoming) == []
    assert "import" in t and "restart" not in t


@pytest.mark.parametrize("outcome", ["false", "raises"])
def test_failed_import_falls_back_to_the_restart(tmp_path, outcome):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    s = fake_steps(log, ids)
    def imp(eid):
        log.append("import")
        if outcome == "raises":
            raise RuntimeError("404 Unknown endpoint")
        return False
    s.import_entry = imp
    t = handoff.run({"messages": []}, ids, c, s, Gate(), lambda m: None)
    assert log[-2:] == ["import", "restart"] and "restart" in t
    assert os.path.isdir(os.path.join(c.cache_root, "e1000"))


def test_failed_import_and_busy_sushi_leave_nothing_behind(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    c.idle_wait_s = 0.1
    s = fake_steps(log, ids)
    s.import_entry = lambda eid: False
    g = Gate(); g.enter()
    with pytest.raises(handoff.HandoffError, match="busy"):
        handoff.run({"messages": []}, ids, c, s, g, lambda m: None)
    assert "restart" not in log and os.listdir(c.cache_root) == [] and os.listdir(c.incoming) == []


def test_real_import_entry_posts_the_id(monkeypatch):
    calls = []
    def fake_post(url, body, timeout):
        calls.append((url, body))
        return {"queued": body["id"]}
    monkeypatch.setattr(handoff, "_post", fake_post)
    assert handoff.real_steps().import_entry(1234) is True
    assert calls == [("http://127.0.0.1:8000/v1/kvh/import", {"id": 1234})]


def _stream_step(log, ids, fail=False):
    def stream(session, out):
        log.append("stream")
        if fail:
            raise RuntimeError("ssh: connection reset")
        os.makedirs(out)
        array("I", ids).tofile(open(os.path.join(out, "tokens.bin"), "wb"))
        open(os.path.join(out, "meta.json"), "w").write("{}")
    return stream


def test_streamed_conversion_replaces_dump_and_convert(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    s = fake_steps(log, ids)
    s.stream_convert = _stream_step(log, ids)
    t = handoff.run({"messages": []}, ids, c, s, Gate(), lambda m: None)
    assert log == ["prefill", "save", "stream", "restart"]
    assert "stream" in t and os.path.isdir(os.path.join(c.cache_root, "e1000"))


def test_streamed_entry_with_other_tokens_is_refused(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    s = fake_steps(log, ids)
    s.stream_convert = _stream_step(log, list(range(1, 301)))
    with pytest.raises(handoff.HandoffError, match="token"):
        handoff.run({"messages": []}, ids, c, s, Gate(), lambda m: None)
    assert "restart" not in log and os.listdir(c.cache_root) == [] and os.listdir(c.incoming) == []


def test_failed_stream_cleans_up(tmp_path):
    c, log, ids = cfg(tmp_path), [], list(range(300))
    s = fake_steps(log, ids)
    s.stream_convert = _stream_step(log, ids, fail=True)
    with pytest.raises(handoff.HandoffError, match="stream"):
        handoff.run({"messages": []}, ids, c, s, Gate(), lambda m: None)
    assert "restart" not in log and os.listdir(c.cache_root) == [] and os.listdir(c.incoming) == []
