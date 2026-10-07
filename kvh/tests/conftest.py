import json
import os
import sys
from array import array

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_entry(root, eid, tokens, ssm_positions):
    """A minimal Sushi v8 entry: meta.json + tokens.bin (enough for cache_index)."""
    d = os.path.join(root, f"e{eid}")
    os.makedirs(d)
    array("I", tokens).tofile(open(os.path.join(d, "tokens.bin"), "wb"))
    json.dump({"v": 8, "kv_len": len(tokens), "tokens": len(tokens),
               "ssm": [{"pos": p, "bytes": 1} for p in ssm_positions]}, open(os.path.join(d, "meta.json"), "w"))
    return d


@pytest.fixture(autouse=True)
def _private_proxy_log(tmp_path, monkeypatch):
    """Tests never write into the real kvh/handoff_proxy.log."""
    try:
        import handoff_proxy
    except ImportError:
        return
    monkeypatch.setattr(handoff_proxy, "LOG", str(tmp_path / "handoff_proxy-test.log"))
    monkeypatch.setattr(handoff_proxy, "CAPTURE_DIR", str(tmp_path / "no-capture"))   # never the real capture folder
