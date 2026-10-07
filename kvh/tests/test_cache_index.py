import json
import os

from cache_index import best_restore, common_prefix, find_cache_root, next_entry_id
from conftest import make_entry


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix(list(range(10000)), list(range(10000))) == 10000
    assert common_prefix(list(range(10000)), list(range(5000)) + [-1]) == 5000
    assert common_prefix([], [1]) == 0


def test_best_restore_picks_highest_checkpoint_inside_shared_prefix(tmp_path):
    prompt = list(range(100))
    make_entry(tmp_path, 1, list(range(60)) + [999] * 40, [20, 50, 70])   # shares 60: 50 usable, 70 not
    make_entry(tmp_path, 2, list(range(30)), [10, 29])                    # shares 30: 29 usable
    assert best_restore(str(tmp_path), prompt) == 50


def test_best_restore_never_returns_the_full_prompt(tmp_path):
    prompt = list(range(40))
    make_entry(tmp_path, 1, list(range(40)), [40])       # a checkpoint AT the prompt end cannot be restored
    make_entry(tmp_path, 2, list(range(40)), [33])
    assert best_restore(str(tmp_path), prompt) == 33


def test_best_restore_only_counts_entries_with_the_same_tools_flag(tmp_path):
    prompt = list(range(100))
    d = make_entry(tmp_path, 1, list(range(100)), [80])
    make_entry(tmp_path, 2, list(range(100)), [40])
    meta = json.load(open(os.path.join(d, "meta.json"))); meta["has_tools"] = True
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    assert best_restore(str(tmp_path), prompt) == 40
    assert best_restore(str(tmp_path), prompt, has_tools=True) == 80


def test_best_restore_ignores_junk(tmp_path):
    (tmp_path / "e7").mkdir()                              # no meta.json
    (tmp_path / "notes").mkdir()
    (tmp_path / "e8").mkdir()
    (tmp_path / "e8" / "meta.json").write_text("{broken")
    assert best_restore(str(tmp_path), [1, 2, 3]) == 0


def test_next_entry_id(tmp_path):
    assert next_entry_id(str(tmp_path)) == 1000
    make_entry(tmp_path, 41, [1], [])
    make_entry(tmp_path, 7, [1], [])
    assert next_entry_id(str(tmp_path)) == 1041


def test_find_cache_root(tmp_path):
    log = tmp_path / "sushi-server.log"
    log.write_text("x\n  [disk-cache] scanned 3 persisted entries (12.0 MB) at /a/b/526f\nmore\n"
                   "  [disk-cache] scanned 4 persisted entries (13.0 MB) at /a/b/c0de\n")
    assert find_cache_root(str(log), "/default") == "/a/b/c0de"
    assert find_cache_root(str(tmp_path / "missing.log"), "/default") == "/default"


def test_best_restore_skips_non_object_meta(tmp_path):
    (tmp_path / "e5").mkdir()
    (tmp_path / "e5" / "meta.json").write_text("[]")
    make_entry(tmp_path, 6, list(range(30)), [10, 29])
    assert best_restore(str(tmp_path), list(range(100))) == 29
