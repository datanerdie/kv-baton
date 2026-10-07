"""Read-only view of Sushi's disk prefix cache (manifest v8): which part of a prompt can be restored.

Sushi resumes at an SSM checkpoint: a prompt can restore from entry E at checkpoint position p when the first p tokens
of the prompt equal E's tokens and p is below the prompt length (Sushi always forwards at least one token).
"""
import json
import os
import re
from array import array

ENTRY_RE = re.compile(r"^e(\d+)$")
SCAN_RE = re.compile(r"\[disk-cache\] scanned \d+ persisted entries \([^)]*\) at (\S+)")


def find_cache_root(log_path, default):
    """The cache folder Sushi reported at its last start (`[disk-cache] scanned ... at <dir>`), else `default`."""
    try:
        with open(log_path, errors="replace") as f:
            found = SCAN_RE.findall(f.read())
    except OSError:
        return default
    return found[-1] if found else default


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n:
        j = min(i + 4096, n)
        if list(a[i:j]) == list(b[i:j]):
            i = j
            continue
        while a[i] == b[i]:
            i += 1
        return i
    return n


def _tokens(path):
    a = array("I")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def best_restore(root, prompt_ids):
    best = 0
    for name in os.listdir(root):
        if not ENTRY_RE.match(name):
            continue
        d = os.path.join(root, name)
        try:
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            cps = [int(s["pos"]) for s in meta.get("ssm") or []]
            if not cps or max(cps) <= best:
                continue
            shared = common_prefix(_tokens(os.path.join(d, "tokens.bin")), prompt_ids)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        usable = [p for p in cps if p <= shared and p < len(prompt_ids)]
        if usable:
            best = max(best, max(usable))
    return best


def next_entry_id(root):
    ids = [int(m.group(1)) for m in map(ENTRY_RE.match, os.listdir(root)) if m]
    return max(ids, default=0) + 1000
