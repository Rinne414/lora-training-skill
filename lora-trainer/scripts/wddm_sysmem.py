#!/usr/bin/env python3
"""Warn when a Windows training run spills GPU memory into shared system memory.

On Windows the display driver (WDDM) lets a CUDA program keep allocating after the
card's own (dedicated) memory is full: the extra goes to shared system memory over
PCIe, nothing fails, and training silently becomes several times slower. Linux (and a
Windows card set to "Prefer No Sysmem Fallback") raises a clear out-of-memory error
instead.

This script only looks, it changes nothing. It reads the training process's counters
``\\GPU Process Memory(pid_<PID>_*)\\Shared Usage`` and ``Dedicated Usage`` through
PowerShell ``Get-Counter`` (no admin rights needed), asks ``nvidia-smi`` how full the
card is, and warns once when shared usage is above a threshold (default 1 GiB) while
the card is nearly full (default 90 %). When a counter cannot be read (not Windows,
localized counter names, no counters for that pid) it says "could not check" and exits 0.

    python wddm_sysmem.py --task-id <uuid> --state ./insights/wddm.json
    python wddm_sysmem.py --pid 12345
    python wddm_sysmem.py --match 20261027-032119.toml

The training process is found by its command line: the trainer starts python with
``--config_file <config_path>``, and the task's ``config_path`` comes from
``GET /api/tasks``. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:28000"
GIB = 1024**3
DEFAULT_SHARED_THRESHOLD_BYTES = GIB  # more than this much shared memory is not normal
DEFAULT_NEAR_FULL = 0.90
POWERSHELL_TIMEOUT_SECONDS = 30
INSTANCE = re.compile(r"^pid_(\d+)_")

PS_COUNTERS = (
    "$ErrorActionPreference='Stop';"
    "$c=Get-Counter -Counter '\\GPU Process Memory(*)\\Shared Usage',"
    "'\\GPU Process Memory(*)\\Dedicated Usage';"
    "$c.CounterSamples|ForEach-Object{[pscustomobject]@{path=$_.Path;instance=$_.InstanceName;"
    "value=[int64]$_.CookedValue}}|ConvertTo-Json -Compress"
)
PS_PROCESSES = (
    "$ErrorActionPreference='Stop';"
    "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\"|"
    "Select-Object ProcessId,ParentProcessId,Name,CommandLine|ConvertTo-Json -Compress"
)

MESSAGE = {
    "en": (
        "The training process is using {shared:.1f} GiB of shared system memory while the card is "
        "{full:.0%} full. Windows is spilling GPU memory into RAM instead of failing, so training "
        "is probably much slower than it should be."
    ),
    "zh": (
        "训练进程占用了 {shared:.1f} GiB 共享系统内存，而显卡已用 {full:.0%}。"
        "Windows 把显存溢出到内存而没有报错，训练很可能比正常慢得多。"
    ),
}
FIX = {
    "en": (
        "To get a clear out-of-memory error instead of slowness: NVIDIA Control Panel > Manage 3D "
        "settings > 'CUDA - Sysmem Fallback Policy' > 'Prefer No Sysmem Fallback' (global, or for the "
        "trainer's python.exe). Then the run fails fast when it does not fit and you can lower "
        "train_batch_size / resolution, turn on gradient_checkpointing or raise blocks_to_swap. "
        "Nothing was changed for you."
    ),
    "zh": (
        "想让它直接报显存不足而不是变慢：NVIDIA 控制面板 > 管理 3D 设置 > 「CUDA - 系统内存回退策略」"
        " > 「首选无系统内存回退」（全局，或只对训练器的 python.exe）。之后放不下会立刻报错，"
        "再调低 train_batch_size / resolution、开启 gradient_checkpointing 或提高 blocks_to_swap。"
        "本脚本没有改任何设置。"
    ),
}


class CheckUnavailable(Exception):
    """A counter or tool could not be read; the check reports 'could not check'."""


class SystemRunner:
    """Runs the real PowerShell and nvidia-smi commands."""

    @staticmethod
    def _run(args: list[str]) -> str:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argument lists, no shell
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=POWERSHELL_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as error:
            raise CheckUnavailable(f"{args[0]} not found") from error
        except subprocess.TimeoutExpired as error:
            raise CheckUnavailable(f"{args[0]} timed out") from error
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip().splitlines()
            raise CheckUnavailable(
                f"{args[0]} failed: {detail[0][:200] if detail else done.returncode}"
            )
        return done.stdout

    def _powershell(self, script: str) -> str:
        return self._run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])

    def counters(self) -> str:
        return self._powershell(PS_COUNTERS)

    def processes(self) -> str:
        return self._powershell(PS_PROCESSES)

    def gpu_memory(self) -> str:
        return self._run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_rows(text: str) -> list[dict]:
    """ConvertTo-Json gives an object for one item and a list for many."""
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except ValueError as error:
        raise CheckUnavailable(f"unreadable PowerShell output: {error}") from error
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def usage_by_pid(rows: list[dict]) -> dict[int, dict[str, int]]:
    """Shared and dedicated bytes per pid, summed over adapters."""
    usage: dict[int, dict[str, int]] = {}
    for row in rows:
        match = INSTANCE.match(str(row.get("instance", "")))
        counter = str(row.get("path", "")).rsplit("\\", 1)[-1].lower()
        key = {"shared usage": "shared", "dedicated usage": "dedicated"}.get(counter)
        if not match or key is None:
            continue
        entry = usage.setdefault(int(match.group(1)), {"shared": 0, "dedicated": 0})
        entry[key] += int(row.get("value") or 0)
    return usage


def find_pids(processes_json: str, needle: str, exclude: Iterable[int] = ()) -> list[int]:
    """Python processes whose command line contains ``needle`` (never this script itself)."""
    skip = set(exclude)
    found = []
    for proc in parse_rows(processes_json):
        command = str(proc.get("CommandLine") or "")
        pid = int(proc["ProcessId"])
        if needle and needle in command and pid not in skip:
            found.append(pid)
    return found


def parse_gpu_memory(text: str) -> tuple[Optional[float], Optional[float]]:
    """``(total_bytes, used_bytes)`` of the fullest card; None when unreadable."""
    best: Optional[tuple[float, float]] = None
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total, used = float(parts[0]) * 1024**2, float(parts[1]) * 1024**2
        except ValueError:
            continue
        if total > 0 and (best is None or used / total > best[1] / best[0]):
            best = (total, used)
    return best if best else (None, None)


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------


def evaluate(
    shared: float,
    dedicated: float,
    vram_total: Optional[float],
    vram_used: Optional[float],
    threshold_bytes: float = DEFAULT_SHARED_THRESHOLD_BYTES,
    near_full: float = DEFAULT_NEAR_FULL,
) -> dict[str, Any]:
    """Status: ok | warn | shared_high_vram_unknown."""
    if shared <= threshold_bytes:
        return {"status": "ok"}
    if vram_total is None or vram_used is None:
        return {
            "status": "shared_high_vram_unknown",
            "message_en": (
                f"The training process uses {shared / GIB:.1f} GiB of shared system memory. "
                "How full the card is could not be read, so this may or may not be the sysmem fallback."
            ),
            "message_zh": (
                f"训练进程占用了 {shared / GIB:.1f} GiB 共享系统内存。读不到显卡占用情况，"
                "无法判断是不是显存溢出到了内存。"
            ),
            "fix_en": FIX["en"],
            "fix_zh": FIX["zh"],
        }
    fullness = vram_used / vram_total
    if fullness < near_full:
        return {
            "status": "ok",
            "note_en": (
                f"Shared usage is {shared / GIB:.1f} GiB but the card is only {fullness:.0%} full, "
                "so this is not a memory spill."
            ),
            "note_zh": f"共享内存 {shared / GIB:.1f} GiB，但显卡只用了 {fullness:.0%}，不是显存溢出。",
        }
    return {
        "status": "warn",
        "message_en": MESSAGE["en"].format(shared=shared / GIB, full=fullness),
        "message_zh": MESSAGE["zh"].format(shared=shared / GIB, full=fullness),
        "fix_en": FIX["en"],
        "fix_zh": FIX["zh"],
    }


def _unavailable(reason: str, reason_zh: Optional[str] = None) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "message_en": f"Could not check shared GPU memory: {reason}.",
        "message_zh": f"无法检查共享显存：{reason_zh or reason}。",
        "announce": False,
    }


def _load_state(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def check(
    pids: list[int],
    runner: Any = None,
    state_path: Optional[Path] = None,
    platform: Optional[str] = None,
    threshold_bytes: float = DEFAULT_SHARED_THRESHOLD_BYTES,
    near_full: float = DEFAULT_NEAR_FULL,
) -> dict[str, Any]:
    """One read of the counters. ``announce`` is true only the first time a warning appears."""
    platform = sys.platform if platform is None else platform
    if platform != "win32":
        return _unavailable(
            "only Windows spills GPU memory into system memory this way",
            "只有 Windows 会这样把显存溢出到系统内存",
        )
    if not pids:
        return _unavailable("no training process found", "没有找到训练进程")
    runner = runner or SystemRunner()
    try:
        usage = usage_by_pid(parse_rows(runner.counters()))
    except CheckUnavailable as error:
        return _unavailable(str(error))
    mine = {pid: usage[pid] for pid in pids if pid in usage}
    if not mine:
        ids = ", ".join(map(str, pids))
        return _unavailable(f"no GPU counters for pid {ids}", f"进程 {ids} 没有 GPU 计数器")
    shared = sum(u["shared"] for u in mine.values())
    dedicated = sum(u["dedicated"] for u in mine.values())
    try:
        total, used = parse_gpu_memory(runner.gpu_memory())
    except CheckUnavailable:
        total = used = None
    verdict = evaluate(shared, dedicated, total, used, threshold_bytes, near_full)
    state = _load_state(state_path)
    warned = bool(state.get("warned"))
    verdict["announce"] = verdict["status"] in ("warn", "shared_high_vram_unknown") and not warned
    verdict.update(
        {
            "pids": sorted(mine),
            "shared_gib": round(shared / GIB, 2),
            "dedicated_gib": round(dedicated / GIB, 2),
            "vram_total_gib": round(total / GIB, 2) if total else None,
            "vram_used_gib": round(used / GIB, 2) if used else None,
        }
    )
    if state_path is not None:
        now_warned = verdict["status"] != "ok" and (warned or verdict["announce"])
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"warned": now_warned}), encoding="utf-8")
    return verdict


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def task_config_name(base_url: str, task_id: str) -> Optional[str]:
    """File name of the task's ``config_path`` from ``GET /api/tasks``; None when unknown."""
    try:
        with urllib.request.urlopen(  # noqa: S310 - local trusted service
            f"{base_url.rstrip('/')}/api/tasks", timeout=10
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    tasks = ((payload or {}).get("data") or {}).get("tasks") or []
    for task in tasks:
        if task.get("id") == task_id:
            path = (task.get("metadata") or {}).get("config_path")
            return re.split(r"[\\/]", str(path))[-1] if path else None
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--task-id", help="find the training process of this trainer task")
    parser.add_argument("--pid", type=int, action="append", help="check this process id")
    parser.add_argument("--match", help="find python processes whose command line contains this")
    parser.add_argument("--state", help="remember that the warning was already shown")
    parser.add_argument("--threshold-gb", type=float, default=1.0)
    parser.add_argument("--near-full", type=float, default=DEFAULT_NEAR_FULL)
    args = parser.parse_args(argv)
    if not (args.task_id or args.pid or args.match):
        parser.error("give --task-id, --pid or --match")

    runner = SystemRunner()
    pids = list(args.pid or [])
    needle = args.match
    if args.task_id and not needle:
        needle = task_config_name(args.base_url, args.task_id)
        if needle is None and not pids:
            print(
                json.dumps(
                    _unavailable("trainer does not know this task", "训练器不认识这个任务"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
    if needle and sys.platform == "win32":
        try:
            pids += find_pids(runner.processes(), needle, exclude={os.getpid(), os.getppid()})
        except CheckUnavailable as error:
            print(json.dumps(_unavailable(str(error)), ensure_ascii=False, indent=2))
            return 0
    result = check(
        sorted(set(pids)),
        runner=runner,
        state_path=Path(args.state) if args.state else None,
        threshold_bytes=args.threshold_gb * GIB,
        near_full=args.near_full,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
