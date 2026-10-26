#!/usr/bin/env python3
"""Ask the trainer what state it is in, and submit a run the way the newer trainers want.

The story-next fork of SD-Trainer (branch ``feat/engine-state`` and later) adds:

    GET  /api/engines                 per train type: ready / broken / ..., plain reason + hint
    POST /api/engines/diagnose        deeper checks for one train type and a config
    GET  /api/queue                   queue entries; ``restored`` after a restart
    POST /api/queue/start             start a restored (waiting) queue
    POST /api/run                     now answers ``queued`` + ``entry_id``, and
                                      ``output_name_renamed`` when the name was taken

Every call here degrades on an older trainer: a 404 means "this trainer has no such
route" and is reported as a note, never as a failure.

Commands (all print JSON):

    preflight --config cfg.json     engines + diagnose, before the confirm card is shown
    queue                           what the queue holds; whether the user must decide
    start-queue --confirmed         POST /api/queue/start (only after the user said yes)
    submit --config cfg.json --confirmed
                                    POST /api/run, then follow the queue entry to its task id

What this script never does: re-queue an interrupted entry, start a restored queue without
``--confirmed``, or call ``/api/plugins/anima-lora/repair`` (a full reinstall that can
re-download gigabytes; it has no command here on purpose). ``--confirmed`` is the user's
"yes"; pass it only after they gave it.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:28000"
REQUEST_TIMEOUT_SECONDS = 20
QUEUE_POLL_SECONDS = 1.0
DEFAULT_WAIT_SECONDS = 20
READY_STATES = ("ready",)
UNKNOWN_STATES = ("unknown",)
ACTIONS_NOTE = (
    "These actions are a full reinstall that may re-download several GB. This skill only "
    "describes them; the user decides, and nothing here calls the repair route."
)
OLDER = "this trainer has no {what} (older build); skipped"


class Unreachable(Exception):
    pass


def _call(base_url: str, path: str, body: Optional[dict] = None) -> tuple[int, Any]:
    """``(status, json)``; HTTP errors keep their status, network errors raise ``Unreachable``."""
    url = f"{base_url.rstrip('/')}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except OSError as error:
        raise Unreachable(str(error)) from error
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, None


def _data(status: int, payload: Any) -> Optional[dict]:
    if status == 200 and isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return None


def _view(issue: dict) -> dict:
    keys = ("code", "severity", "reason_zh", "reason_en", "hint_zh", "hint_en", "field")
    return {key: issue[key] for key in keys if key in issue}


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


def preflight(base_url: str, train_type: str, config: dict) -> dict[str, Any]:
    """Report the engine state and diagnose result for ``train_type`` and ``config``."""
    result: dict[str, Any] = {
        "train_type": train_type,
        "verdict": "unknown",
        "engine": None,
        "issues": [],
        "config_issues": [],
        "notes": [],
    }
    try:
        status, payload = _call(base_url, "/api/engines")
        data = _data(status, payload)
        if data is None:
            result["notes"].append(OLDER.format(what="engine report"))
        else:
            row = (data.get("by_train_type") or {}).get(train_type)
            if row is None:
                result["notes"].append(f"the trainer does not list train type {train_type}")
            else:
                result["engine"] = {
                    key: row.get(key, "")
                    for key in ("engine", "state", "reason_zh", "reason_en", "hint_zh", "hint_en")
                }
        status, payload = _call(
            base_url, "/api/engines/diagnose", {"train_type": train_type, "config": config}
        )
        diag = _data(status, payload)
        if diag is None:
            result["notes"].append(OLDER.format(what="diagnose route"))
        else:
            _merge_diagnose(result, diag)
    except Unreachable as error:
        result["notes"].append(f"trainer not reachable: {error}")
        return result
    result["verdict"] = _verdict(result)
    return result


def _merge_diagnose(result: dict, diag: dict) -> None:
    engines = diag.get("engines") or []
    engine = next(
        (e for e in engines if e.get("id") == (result.get("engine") or {}).get("engine")),
        engines[0] if engines else None,
    )
    if engine:
        result["issues"] = [_view(i) for i in engine.get("issues") or []]
        if result["engine"] is None:
            result["engine"] = {
                "engine": engine.get("id"),
                **{
                    k: engine.get(k, "")
                    for k in ("state", "reason_zh", "reason_en", "hint_zh", "hint_en")
                },
            }
        actions = engine.get("actions") or []
        if actions:
            result["actions_needing_the_user"] = [
                {"id": a.get("id"), "label_en": a.get("label_en"), "label_zh": a.get("label_zh")}
                for a in actions
            ]
            result["actions_note"] = ACTIONS_NOTE
    result["config_issues"] = [_view(i) for i in diag.get("config_issues") or []]
    result["diagnose_ok"] = diag.get("ok")


def _verdict(result: dict) -> str:
    engine = result["engine"]
    state = (engine or {}).get("state")
    everything = result["issues"] + result["config_issues"]
    if any(i.get("severity") == "error" for i in everything):
        return "problems"
    if state and state not in READY_STATES + UNKNOWN_STATES:
        return "problems"
    if everything:
        return "warnings"
    if engine is None or state in UNKNOWN_STATES:
        return "unknown"
    return "ok"


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------


def queue_summary(base_url: str) -> dict[str, Any]:
    """What the queue holds and whether the user has to decide something."""
    try:
        status, payload = _call(base_url, "/api/queue")
    except Unreachable as error:
        return {"available": False, "note": f"trainer not reachable: {error}"}
    data = _data(status, payload)
    if data is None:
        return {"available": False, "note": OLDER.format(what="queue")}
    entries = data.get("entries") or []
    restored = data.get("restored")
    interrupted = [
        {"id": e.get("id"), "name": e.get("name"), "error": e.get("error")}
        for e in entries
        if e.get("interrupted")
    ]
    waiting = bool(restored and restored.get("waiting"))
    return {
        "available": True,
        "active": data.get("active"),
        "busy": data.get("busy"),
        "running_task_id": data.get("running_task_id"),
        "halt_reason": data.get("halt_reason"),
        "counts": _counts(entries),
        "restored": restored,
        "ask_user_before_start": waiting,
        "interrupted": interrupted,
        "auto_requeue": False,
        "advice": _advice(waiting, interrupted),
    }


def _counts(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry.get("status"))] = counts.get(str(entry.get("status")), 0) + 1
    return counts


def _advice(waiting: bool, interrupted: list[dict]) -> list[str]:
    advice = []
    if waiting:
        advice.append(
            "The trainer restarted with entries still queued and is holding them. Tell the user "
            "and ask before starting the queue (start-queue --confirmed)."
        )
    if interrupted:
        advice.append(
            "These trainings were interrupted by the restart. They are not restarted "
            "automatically and this skill does not requeue them; offer it, and let the user "
            "press Requeue if they want it."
        )
    return advice


def start_queue(base_url: str, confirmed: bool) -> dict[str, Any]:
    """POST /api/queue/start, only with the user's yes."""
    if not confirmed:
        return {
            "status": "refused",
            "message": "ask the user first, then repeat with --confirmed",
        }
    try:
        status, payload = _call(base_url, "/api/queue/start", {})
    except Unreachable as error:
        return {"status": "unreachable", "message": str(error)}
    if status == 404:
        return {"status": "unavailable", "message": OLDER.format(what="queue")}
    if isinstance(payload, dict):
        return {
            "status": payload.get("status", f"http_{status}"),
            "message": payload.get("message"),
        }
    return {"status": f"http_{status}", "message": None}


# --------------------------------------------------------------------------
# Submit
# --------------------------------------------------------------------------


def submit(
    base_url: str,
    config: dict,
    confirmed: bool,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST /api/run, report the output name actually used and follow a queued entry."""
    requested = str(config.get("output_name") or "")
    if not confirmed:
        return {
            "status": "refused",
            "message": "show the confirm card and get the user's yes, then repeat with --confirmed",
        }
    try:
        status, payload = _call(base_url, "/api/run", config)
    except Unreachable as error:
        return {"status": "unreachable", "message": str(error), "task_id": None}
    payload = payload if isinstance(payload, dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    result: dict[str, Any] = {
        "status": payload.get("status", f"http_{status}"),
        "message": payload.get("message"),
        "queued": bool(data.get("queued")),
        "entry_id": data.get("entry_id"),
        "task_id": data.get("task_id"),
        "output_name": requested,
        "output_name_renamed": None,
    }
    if result["status"] != "success":
        return result
    renamed = data.get("output_name_renamed")
    if isinstance(renamed, dict) and renamed.get("to"):
        result["output_name"] = str(renamed["to"])
        result["output_name_renamed"] = {"from": renamed.get("from"), "to": renamed["to"]}
    result["use_this_name_for"] = (
        f"output_name {result['output_name']!r}: the LoRA file, task_insights.py --output-name, "
        "previews and the result sidecar"
    )
    if result["queued"] and not result["task_id"]:
        _follow_entry(base_url, result, wait_seconds, sleep)
    if result["task_id"]:
        result["log_tail_url"] = f"{base_url.rstrip('/')}/api/train/log/tail/{result['task_id']}"
    return result


def _follow_entry(base_url: str, result: dict, wait_seconds: float, sleep) -> None:
    """Poll the queue until the entry has a task id, or say it is waiting."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, payload = _call(base_url, "/api/queue")
        except Unreachable:
            return
        data = _data(status, payload) or {}
        entry = next(
            (e for e in data.get("entries") or [] if e.get("id") == result["entry_id"]), None
        )
        if entry:
            result["entry_status"] = entry.get("status")
            if entry.get("name"):
                result["queue_entry_name"] = entry["name"]
            if entry.get("task_id"):
                result["task_id"] = entry["task_id"]
                return
        if not data.get("active") or time.monotonic() >= deadline:
            result["waiting_for_queue_start"] = not data.get("active")
            if result["waiting_for_queue_start"]:
                result["note"] = (
                    "the queue is paused or held; the run starts when the user starts the queue "
                    "(ask first, then start-queue --confirmed)"
                )
            return
        sleep(QUEUE_POLL_SECONDS)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("the config must be a JSON object")
    return config


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="engine report + diagnose before the confirm card")
    pre.add_argument("--config", required=True, help="the flat config JSON for POST /api/run")
    pre.add_argument("--train-type", help="default: model_train_type in the config")
    sub.add_parser("queue", help="queue state; whether the user must decide")
    start = sub.add_parser("start-queue", help="start a restored queue (user said yes)")
    start.add_argument("--confirmed", action="store_true")
    run = sub.add_parser("submit", help="POST /api/run and follow the queue entry")
    run.add_argument("--config", required=True)
    run.add_argument("--confirmed", action="store_true")
    run.add_argument("--wait", type=float, default=DEFAULT_WAIT_SECONDS)
    args = parser.parse_args(argv)

    if args.command == "queue":
        result = queue_summary(args.base_url)
    elif args.command == "start-queue":
        result = start_queue(args.base_url, args.confirmed)
    else:
        try:
            config = _load_config(args.config)
        except (OSError, ValueError) as error:
            print(json.dumps({"status": "error", "message": f"cannot read config: {error}"}))
            return 2
        if args.command == "preflight":
            train_type = args.train_type or str(config.get("model_train_type") or "anima-lora")
            result = preflight(args.base_url, train_type, config)
        else:
            result = submit(args.base_url, config, args.confirmed, args.wait)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result.get("status") == "refused" else 0


if __name__ == "__main__":
    sys.exit(main())
