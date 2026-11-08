"""Resource-contention guard, built 2026-11-08 after the 223-item real diary pipeline died at
215/223 with exit code 1 and no captured traceback (its stdout had been redirected to a file that
no longer exists -- see lucy_west_resource_guard_and_crash_resilience_20261108.md). Investigated
for outside interference first (Windows Application/System event logs, OOM-killer events, crash
dumps, disk health) -- found NONE, which points at an ordinary unhandled Python exception rather
than something external killing the process. The most likely real cause: several extra heavy
`load_or_init()` diagnostic scripts were run concurrently with the live pipeline that same window,
competing for RAM on a 31.7GB machine. Two real, separate fixes, both here:

1. `mark_running()` / `is_running()` / `pid_info()`: a tiny PID+timestamp lockfile so any OTHER
   script can tell whether the real production pipeline is active right now, instead of guessing.
2. `yield_to_production(min_free_gb)`: meant to be called by any AUXILIARY/diagnostic script
   (not the production pipeline itself) before or during heavy work. If the production lockfile
   says it's running AND free RAM is below the floor, this SELF-PAUSES the calling process
   (sleeps, checking periodically) until conditions clear. Deliberately self-pause, not
   other-process-suspend: suspending another process by PID (psutil.Process(pid).suspend() is
   real and available) was considered and rejected as the default here -- misidentifying which
   process is safe to freeze is a real risk (it could suspend something that mattered), while a
   process pausing ONLY ITSELF can never make that mistake. `force_suspend_other(pid)` is provided
   below for the rare case a human explicitly wants to pause a SPECIFIC other process by its known
   PID -- never auto-invoked.
"""
import json
import time
from pathlib import Path

import psutil

LOCKFILE = Path(__file__).resolve().parent / "checkpoints" / ".production_pipeline.lock"


def mark_running(label: str) -> None:
    """Call once, right after a long-running production script starts. Overwrites any stale
    lockfile from a previous run (a crashed process never cleans up after itself, so staleness is
    expected and handled by checking the PID is actually alive in is_running(), not by trusting
    the file's mere existence)."""
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text(json.dumps({"pid": psutil.Process().pid, "label": label,
                                     "started": time.time()}), encoding="utf-8")


def mark_stopped() -> None:
    """Call on clean exit. A crash skips this -- is_running() handles that case by checking PID
    liveness, not file presence, so a crashed run's stale lockfile is harmless, not a false
    positive."""
    LOCKFILE.unlink(missing_ok=True)


def pid_info():
    if not LOCKFILE.is_file():
        return None
    try:
        return json.loads(LOCKFILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_running() -> bool:
    """True only if the lockfile exists AND the PID it names is a live process right now -- a
    stale lockfile from a crashed run must never look like "still running" forever."""
    info = pid_info()
    if not info:
        return False
    return psutil.pid_exists(info["pid"])


def yield_to_production(min_free_gb: float = 4.0, check_every_s: float = 5.0,
                         max_wait_s: float = 600.0) -> float:
    """For AUXILIARY scripts only -- call before/during heavy work (a full checkpoint load, a
    many-rep training loop). If the real production pipeline is running AND free RAM is below
    min_free_gb, sleeps (checking periodically) until either condition clears, up to max_wait_s.
    Returns the actual seconds waited (0.0 if no wait was needed). Never touches any other
    process -- this only ever pauses the process that calls it."""
    waited = 0.0
    while is_running() and psutil.virtual_memory().available / 1e9 < min_free_gb:
        if waited >= max_wait_s:
            print(f"[resource_guard] waited {waited:.0f}s for the production pipeline to free "
                  f"RAM, giving up -- proceeding anyway, may contend for memory.", flush=True)
            break
        print(f"[resource_guard] production pipeline is running and free RAM is "
              f"{psutil.virtual_memory().available / 1e9:.1f}GB (< {min_free_gb}GB floor) -- "
              f"pausing this process {check_every_s:.0f}s before rechecking.", flush=True)
        time.sleep(check_every_s)
        waited += check_every_s
    return waited


def available_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def force_suspend_other(pid: int):
    """Explicit, human-directed pause of a SPECIFIC other process by PID. Never called
    automatically by anything in this module -- provided only because a real mechanism for it
    (psutil.Process.suspend()/.resume()) exists and Jonathan asked about pausing a process as a
    candidate fix; using it on an auto-detected/guessed PID is exactly the risk this module's
    default design (self-pause only) exists to avoid."""
    proc = psutil.Process(pid)
    proc.suspend()
    return proc


def resume_other(pid: int):
    psutil.Process(pid).resume()
