"""Counts requests in flight to Sushi and runs a Sushi restart only when none is.

While the restart waits for idle, new requests still go in (a busy chat is never held for minutes); once idle, the
gate closes, the restart runs, and requests that arrive meanwhile wait for it to finish.
"""
import threading
import time


class Gate:
    def __init__(self):
        self._c = threading.Condition()
        self._inflight = 0
        self._closed = False

    @property
    def inflight(self):
        with self._c:
            return self._inflight

    def enter(self):
        with self._c:
            while self._closed:
                self._c.wait()
            self._inflight += 1

    def leave(self):
        with self._c:
            self._inflight -= 1
            self._c.notify_all()

    def exclusive(self, fn, max_wait):
        deadline = time.monotonic() + max_wait
        with self._c:
            while self._inflight > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, None
                self._c.wait(left)
            self._closed = True
        try:
            return True, fn()
        finally:
            with self._c:
                self._closed = False
                self._c.notify_all()
