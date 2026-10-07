import io
import threading

import pytest

pytest.importorskip("mlx.core")
from strata_to_sushi import Prefetch  # noqa: E402


class Trickle(io.RawIOBase):
    """A pipe-like stream: short reads of a few bytes, then EOF (or an error)."""
    def __init__(self, data, step=3, fail_at=None):
        self.data, self.pos, self.step, self.fail_at = data, 0, step, fail_at
    def readable(self): return True
    def read(self, n=-1):
        if self.fail_at is not None and self.pos >= self.fail_at:
            raise OSError("connection reset")
        b = self.data[self.pos:self.pos + min(n, self.step)]
        self.pos += len(b)
        return b


def read_all(p, sizes):
    out = []
    for n in sizes:
        out.append(p.read(n))
    return out


def test_reads_return_the_stream_in_order_across_blocks():
    data = bytes(range(256)) * 40
    p = Prefetch(Trickle(data, step=7), block=50, depth=3)
    got = b"".join(iter(lambda: p.read(33), b""))
    assert got == data


def test_reads_never_exceed_n_and_eof_is_empty():
    p = Prefetch(Trickle(b"abcdefghij", step=4), block=4, depth=2)
    parts = read_all(p, [3] * 8)                 # a read stops at a block boundary, so more reads than 10 / 3
    assert all(len(x) <= 3 for x in parts) and b"".join(parts) == b"abcdefghij" and parts[-1] == b""
    assert p.read(3) == b""                      # and stays at the end


def test_a_reader_error_reaches_the_consumer():
    p = Prefetch(Trickle(b"x" * 100, step=10, fail_at=30), block=10, depth=2)
    with pytest.raises(OSError, match="connection reset"):
        while p.read(10):
            pass


def test_reading_runs_on_another_thread():
    seen = []
    class Spy(Trickle):
        def read(self, n=-1):
            seen.append(threading.current_thread() is not threading.main_thread())
            return super().read(n)
    p = Prefetch(Spy(b"y" * 20, step=5), block=5, depth=2)
    while p.read(8):
        pass
    assert seen and all(seen)
