import threading
import time

import pytest

from gate import Gate


def test_exclusive_runs_when_idle():
    g = Gate()
    assert g.exclusive(lambda: 42, 1.0) == (True, 42)


def test_exclusive_waits_for_inflight_then_runs():
    g = Gate()
    g.enter()
    threading.Timer(0.2, g.leave).start()
    t0 = time.monotonic()
    ran, _ = g.exclusive(lambda: None, 2.0)
    assert ran and time.monotonic() - t0 >= 0.15


def test_exclusive_gives_up_when_busy():
    g = Gate()
    g.enter()
    assert g.exclusive(lambda: 1, 0.2) == (False, None)
    g.enter(); g.leave(); g.leave()           # the gate is open again
    assert g.inflight == 0


def test_new_requests_wait_while_fn_runs():
    g = Gate()
    order = []
    def slow():
        time.sleep(0.3); order.append("restart done")
    t = threading.Thread(target=g.exclusive, args=(slow, 1.0)); t.start()
    time.sleep(0.05)
    g.enter(); order.append("request entered"); g.leave()
    t.join()
    assert order == ["restart done", "request entered"]


def test_waiting_does_not_block_new_requests():
    g = Gate()
    g.enter()                                   # a long generation is running
    t = threading.Thread(target=g.exclusive, args=(lambda: None, 0.5)); t.start()
    time.sleep(0.05)
    t0 = time.monotonic(); g.enter(); g.leave()  # a chat message during the wait is NOT held
    assert time.monotonic() - t0 < 0.1
    g.leave(); t.join()


def test_exception_reopens_gate():
    g = Gate()
    with pytest.raises(RuntimeError):
        g.exclusive(lambda: (_ for _ in ()).throw(RuntimeError("boom")), 1.0)
    g.enter(); g.leave()
