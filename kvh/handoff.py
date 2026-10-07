"""One big-document handoff: Strata (GPU box) prefills, the state becomes a Sushi cache entry, Sushi restarts.

Every side effect is a field of `Steps`, so the order, the clean-up and the fallbacks are tested with fakes;
`real_steps()` wires them to the tunnel, ssh/rsync, the converter and the restart command.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
from array import array
from dataclasses import dataclass
from typing import Callable, Optional

from cache_index import next_entry_id

KVH = os.path.dirname(os.path.abspath(__file__))
# Defaults fit the author's setup; each can be overridden from the environment.
UV = os.path.expanduser(os.environ.get("KVH_UV", "~/.local/bin/uv"))
# Only used with stock Sushi (no /v1/kvh/import): the command that restarts Sushi so it rescans its cache.
RESTART_CMD = os.environ.get("KVH_SUSHI_RESTART", os.path.expanduser("~/.local/bin/sushi-restart"))
SUSHI_URL = os.environ.get("KVH_SUSHI_URL", "http://127.0.0.1:8000")
STRATA_URL = os.environ.get("KVH_STRATA_URL", "http://127.0.0.1:18080")    # Strata's port, tunnelled to the Mac
STRATA_HOST = os.environ.get("KVH_STRATA_HOST", "gpu-box")    # ssh alias of the NVIDIA box


class HandoffError(Exception):
    pass


class SushiDown(HandoffError):
    pass


@dataclass
class Config:
    cache_root: str
    incoming: str
    min_new_tokens: int = 30000
    strata_max_ctx: int = 131072
    idle_wait_s: float = 300.0


@dataclass
class Steps:
    strata_prefill: Callable[[dict], int]
    strata_save: Callable[[str], None]
    fetch_dump: Callable[[str, str], None]
    convert: Callable[[str, str], None]
    restart_sushi: Callable[[], None]
    # Patched Sushi only (kvh/sushi-kvh-import.patch): index the entry while Sushi runs. True = no restart
    # needed; False / an exception (stock Sushi answers 404) = fall back to restart_sushi.
    import_entry: Optional[Callable[[int], bool]] = None
    # Session file streamed from the GPU box straight into the converter (no dump on the GPU box, copy and convert overlap):
    # (session name, entry dir). When set, it replaces fetch_dump + convert.
    stream_convert: Optional[Callable[[str, str], None]] = None


def decide(n_prompt, restorable, cfg):
    new = n_prompt - restorable
    if n_prompt >= cfg.strata_max_ctx:
        return False, f"prompt {n_prompt} >= Strata's {cfg.strata_max_ctx}"
    if new < cfg.min_new_tokens:
        return False, f"only {new} new tokens (< {cfg.min_new_tokens})"
    return True, f"{new} new tokens"


def run(body, prompt_ids, cfg, steps, gate, progress):
    tag = uuid.uuid4().hex[:12]
    session, dump_dir = f"kvh-{tag}.bin", os.path.join(cfg.incoming, f"dump-{tag}")
    staged = None
    final = None
    t = {}

    def timed(name, fn, *a):
        progress(name)
        t0 = time.monotonic()
        try:
            return fn(*a)
        except HandoffError:
            raise
        except Exception as e:
            raise HandoffError(f"{name}: {e}") from e
        finally:
            t[name] = round(time.monotonic() - t0, 2)

    sbody = {k: v for k, v in body.items() if k not in ("stream_options", "max_completion_tokens", "n", "n_predict", "logprobs", "top_logprobs")}
    sbody.update(max_tokens=1, temperature=0, stream=False)
    try:
        eid = next_entry_id(cfg.cache_root)
        staged, final = os.path.join(cfg.incoming, f"e{eid}"), os.path.join(cfg.cache_root, f"e{eid}")

        n = timed("prefill", steps.strata_prefill, sbody)
        if n != len(prompt_ids):
            raise HandoffError(f"token count differs: Strata {n}, Sushi render {len(prompt_ids)}")
        timed("save", steps.strata_save, session)
        if steps.stream_convert is not None:
            timed("stream", steps.stream_convert, session, staged)
            got = array("I")
            with open(os.path.join(staged, "tokens.bin"), "rb") as f:
                got.frombytes(f.read())
            if list(got) != list(prompt_ids):
                raise HandoffError("token ids differ between Strata and Sushi's render")
        else:
            timed("dump", steps.fetch_dump, session, dump_dir)
            got = array("i")
            with open(os.path.join(dump_dir, "live_ids.i32"), "rb") as f:
                got.frombytes(f.read())
            if list(got) != list(prompt_ids):
                raise HandoffError("token ids differ between Strata and Sushi's render")
            timed("convert", steps.convert, dump_dir, staged)

        if steps.import_entry is not None:
            os.rename(staged, final)
            progress("import")
            t0 = time.monotonic()
            try:
                imported = bool(steps.import_entry(eid))
            except Exception:
                imported = False
            t["import"] = round(time.monotonic() - t0, 2)
            if imported:
                return t

        def swap_and_restart():
            if not os.path.exists(final):
                os.rename(staged, final)
            try:
                steps.restart_sushi()
            except Exception as first_error:
                shutil.rmtree(final, ignore_errors=True)
                try:
                    steps.restart_sushi()
                except SushiDown as e:
                    raise SushiDown(f"restart failed twice: {e}") from first_error
                except Exception as second_error:
                    raise SushiDown(f"restart failed twice: {second_error}") from first_error
                raise HandoffError("Sushi failed to restart with the injected entry; entry removed, Sushi back without it")

        progress("waiting for Sushi to be idle")
        t0 = time.monotonic()
        try:
            ran, _ = gate.exclusive(swap_and_restart, cfg.idle_wait_s)
        except SushiDown:
            shutil.rmtree(final, ignore_errors=True)
            raise
        t["restart"] = round(time.monotonic() - t0, 2)
        if not ran:
            shutil.rmtree(final, ignore_errors=True)    # moved in for the import attempt; never used
            raise HandoffError(f"Sushi busy for {cfg.idle_wait_s:.0f} s")
        return t
    except HandoffError:
        raise
    except Exception as e:
        raise HandoffError(f"handoff: {e}") from e
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _health(timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SUSHI_URL}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def real_steps(tunnel=STRATA_URL, gpu_host=STRATA_HOST):
    def prefill(body):
        return int(_post(f"{tunnel}/v1/chat/completions", body, 600)["usage"]["prompt_tokens"])    # 100K ~ 44 s, cap 131K

    def save(session):
        _post(f"{tunnel}/slots/0?action=save", {"filename": session}, 600)

    def fetch_dump(session, local):
        name = session[:-4]
        try:
            remote = (f"cd ~/kvh && mkdir -p dumps/{name} && "
                      f"./strata_dump sessions/{session} dumps/{name} >/dev/null")
            subprocess.run(["ssh", "-o", "BatchMode=yes", gpu_host, remote], check=True, timeout=600)
            subprocess.run(["rsync", "-a", f"{gpu_host}:kvh/dumps/{name}/", local + "/"], check=True, timeout=900)
        finally:
            try:
                subprocess.run(["ssh", "-o", "BatchMode=yes", gpu_host, f"rm -rf ~/kvh/dumps/{name} ~/kvh/sessions/{session}"], timeout=120)
            except Exception:
                pass

    def convert(dump, out):
        subprocess.run([UV, "run", "-q", "--with", "mlx", "--with", "numpy", "python",
                        os.path.join(KVH, "strata_to_sushi.py"), dump, out], check=True, timeout=900)

    def restart():
        try:
            # own session: the restart command, and the Sushi it starts, must not die with the proxy's process group
            r = subprocess.run(RESTART_CMD.split(), timeout=600, start_new_session=True, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise SushiDown(f"{RESTART_CMD} timeout") from e
        if r.returncode != 0:
            raise SushiDown(f"{RESTART_CMD} exited {r.returncode}")
        if _health(300):
            return
        raise SushiDown(f"Sushi not healthy after {RESTART_CMD}")

    def import_entry(eid):
        # 404 on stock Sushi (no /v1/kvh/import) raises -> run() falls back to the restart
        return "queued" in _post(f"{SUSHI_URL}/v1/kvh/import", {"id": eid}, 10)

    def stream_convert(session, out):
        cat = subprocess.Popen(["ssh", "-o", "BatchMode=yes", gpu_host, f"cat ~/kvh/sessions/{session}"],
                               stdout=subprocess.PIPE, start_new_session=True)
        try:
            conv = subprocess.run([UV, "run", "-q", "--with", "mlx", "--with", "numpy", "python",
                                   os.path.join(KVH, "strata_to_sushi.py"), "-", out],
                                  stdin=cat.stdout, timeout=900, capture_output=True, start_new_session=True)
            cat.stdout.close()
            if cat.wait(timeout=60) != 0:
                raise RuntimeError(f"ssh cat exited {cat.returncode}")
            if conv.returncode != 0:
                raise RuntimeError(f"converter exited {conv.returncode}: {conv.stderr.decode(errors='replace')[-300:]}")
        finally:
            if cat.poll() is None:
                cat.kill()
            try:
                subprocess.run(["ssh", "-o", "BatchMode=yes", gpu_host, f"rm -f ~/kvh/sessions/{session}"], timeout=120)
            except Exception:
                pass

    return Steps(prefill, save, fetch_dump, convert, restart, import_entry, stream_convert)
