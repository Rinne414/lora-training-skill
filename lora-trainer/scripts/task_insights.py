#!/usr/bin/env python3
"""Read one training task's loss curve and preview images from SD-Trainer.

Newer trainers answer per-task routes, which are already filtered to that task's
own time window (so a re-run with the same output name never mixes with an
earlier one):

    GET /api/tasks/{task_id}/metrics            loss / lr series + stdout progress
    GET /api/tasks/{task_id}/previews           preview image list
    GET /api/tasks/{task_id}/previews/{name}    one image (?thumb=1 for a small JPEG)

Older trainers have none of these (HTTP 404 "Not Found"), and a task id only
lives as long as the trainer process (HTTP 404 "Unknown task_id"). In both cases
this script falls back to scanning ``<output_dir>/sample`` and the TensorBoard
``logging_dir`` itself. The fallback needs ``--since`` (the unix time the run was
launched) to keep an earlier same-name run out; without it everything under the
folders is returned and the result says so.

Standard library only. TensorBoard is optional and only used by the fallback.

    python task_insights.py --task-id <uuid> --out ./previews
    python task_insights.py --output-dir D:/out/zkz --output-name zkz \
        --logging-dir ./logs --since 1793037351 --out ./previews   # no task id
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:28000"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LOSS_TAGS = ("loss/average", "loss/current", "loss/epoch_average", "lr/unet")
SINCE_TOLERANCE_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 15


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


def _request(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> tuple[int, bytes]:
    """Return ``(status, body)``; HTTP errors keep their status, network errors raise."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _get_json(url: str) -> tuple[int, Any]:
    status, body = _request(url)
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, None


def _fallback_reason(status: int, payload: Any) -> Optional[str]:
    """None when the answer is usable, otherwise why the folder scan must take over."""
    if status == 200 and isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return None
    detail = str(payload.get("detail", "")) if isinstance(payload, dict) else ""
    if status == 404 and detail.startswith("Unknown task_id"):
        return "task_unknown_to_trainer"  # route exists; trainer restarted or pruned the task
    if status == 404:
        return "trainer_has_no_task_insights_api"  # older trainer
    return f"unexpected_response_http_{status}"


# --------------------------------------------------------------------------
# Trainer API
# --------------------------------------------------------------------------


def fetch_from_api(base_url: str, task_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Ask the trainer. Returns ``(result, None)`` or ``(None, why_fall_back)``."""
    root = f"{base_url.rstrip('/')}/api/tasks/{urllib.parse.quote(task_id, safe='')}"
    status, metrics = _get_json(f"{root}/metrics")
    reason = _fallback_reason(status, metrics)
    if reason:
        return None, reason
    status, previews = _get_json(f"{root}/previews")
    reason = _fallback_reason(status, previews)
    if reason:
        return None, reason
    m, p = metrics["data"], previews["data"]
    images = [
        {**image, "url": f"{base_url.rstrip('/')}{image['url']}"} for image in p.get("images", [])
    ]
    return {
        "source": "trainer_api",
        "tags": m.get("tags", {}),
        "progress": m.get("progress", {}),
        "loss_reason": m.get("reason_text") or m.get("reason"),
        "previews": images,
        "preview_enabled": p.get("preview_enabled"),
        "previews_reason": p.get("reason_text") or p.get("reason"),
    }, None


def download_previews(images: list[dict], out_dir: Path, thumb: bool = False) -> list[str]:
    """Save API previews into ``out_dir``; returns local paths (failures are skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for image in images:
        name = Path(str(image.get("name", ""))).name
        if not name:
            continue
        url = image["url"] + ("?thumb=1" if thumb else "")
        try:
            status, body = _request(url)
        except OSError:
            continue
        if status != 200:
            continue
        target = out_dir / (f"thumb_{name}.jpg" if thumb else name)
        target.write_bytes(body)
        saved.append(str(target))
    return saved


# --------------------------------------------------------------------------
# Folder-scan fallback (older trainers)
# --------------------------------------------------------------------------


def scan_previews(output_dir: Path, output_name: str, since: Optional[float]) -> list[Path]:
    """Preview files sd-scripts wrote to ``<output_dir>/sample`` during the run."""
    floor = since - SINCE_TOLERANCE_SECONDS if since else 0.0
    found = []
    for root, recursive in ((output_dir / "sample", True), (output_dir, False)):
        if not root.is_dir():
            continue
        for path in root.rglob("*") if recursive else root.iterdir():
            if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                continue
            if output_name and not path.name.startswith(output_name):
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= floor:
                found.append((mtime, path))
    return [path for _, path in sorted(found)]


def _run_dir_time(name: str) -> Optional[float]:
    match = re.search(r"(\d{14})", name)
    if not match:
        return None
    try:
        return time.mktime(time.strptime(match.group(1), "%Y%m%d%H%M%S"))
    except (ValueError, OverflowError):
        return None


def scan_loss(logging_dir: Path, since: Optional[float]) -> tuple[dict, Optional[str]]:
    """Loss series from the run that started at or after ``since`` (else the newest run)."""
    if not logging_dir.is_dir():
        return {}, "logging_dir_missing"
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}, "tensorboard_not_installed"
    runs: dict[Path, float] = {}
    for event_file in logging_dir.rglob("events.out.tfevents.*"):
        runs[event_file.parent] = max(runs.get(event_file.parent, 0.0), event_file.stat().st_mtime)
    if since:
        runs = {
            run: mtime for run, mtime in runs.items() if mtime >= since - SINCE_TOLERANCE_SECONDS
        }
    if not runs:
        return {}, "no_tensorboard_run_for_this_launch"
    stamped = [(run, _run_dir_time(run.name)) for run in runs]
    timed = [(run, ts) for run, ts in stamped if ts is not None]
    if timed:
        pick = min if since else max
        chosen = pick(timed, key=lambda item: item[1])[0]
    else:
        chosen = max(runs, key=runs.get)
    accumulator = event_accumulator.EventAccumulator(
        str(chosen), size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    series = {
        tag: [{"step": int(e.step), "value": float(e.value)} for e in accumulator.Scalars(tag)]
        for tag in LOSS_TAGS
        if tag in available
    }
    return (series, None) if series else ({}, "no_loss_tags_in_run")


def fetch_from_folders(
    output_dir: Optional[str], output_name: str, logging_dir: Optional[str], since: Optional[float]
) -> dict:
    previews: list[Path] = scan_previews(Path(output_dir), output_name, since) if output_dir else []
    tags, loss_reason = (
        scan_loss(Path(logging_dir), since) if logging_dir else ({}, "no_logging_dir_given")
    )
    notes = []
    if since is None:
        notes.append("no --since: an earlier run with the same output name may be mixed in")
    if not output_dir:
        notes.append("no --output-dir: previews were not scanned")
    return {
        "source": "folder_scan",
        "tags": tags,
        "progress": {},
        "loss_reason": loss_reason,
        "previews": [{"name": p.name, "path": str(p)} for p in previews],
        "preview_enabled": None,
        "previews_reason": None if previews else "no preview images found in the run window",
        "notes": notes,
    }


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarize_loss(tags: dict) -> dict:
    """Short, decision-ready numbers from ``loss/average`` (or ``loss/current``)."""
    tag = "loss/average" if tags.get("loss/average") else "loss/current"
    points = tags.get(tag) or []
    if not points:
        return {}
    values = [float(p["value"]) for p in points]
    finite = [v for v in values if math.isfinite(v)]
    summary = {
        "tag": tag,
        "points": len(values),
        "first": values[0],
        "last": values[-1],
        "has_nan_or_inf": len(finite) != len(values),
    }
    if finite:
        low = min(finite)
        summary["min"] = low
        summary["min_step"] = points[values.index(low)]["step"]
    epoch_avg = tags.get("loss/epoch_average") or []
    if epoch_avg:
        summary["epoch_average"] = [round(float(p["value"]), 6) for p in epoch_avg]
    return summary


def collect(
    base_url: str,
    task_id: Optional[str],
    output_dir: Optional[str] = None,
    output_name: str = "",
    logging_dir: Optional[str] = None,
    since: Optional[float] = None,
) -> dict:
    """Trainer API first, folder scan when the trainer cannot answer."""
    fallback_why = "no_task_id_given"
    if task_id:
        try:
            result, fallback_why = fetch_from_api(base_url, task_id)
        except OSError as error:  # trainer not reachable
            result, fallback_why = None, f"trainer_unreachable: {error}"
        if result is not None:
            result["loss_summary"] = summarize_loss(result["tags"])
            return result
    result = fetch_from_folders(output_dir, output_name, logging_dir, since)
    result["fallback_because"] = fallback_why
    result["loss_summary"] = summarize_loss(result["tags"])
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--task-id", help="task_id returned by POST /api/run")
    parser.add_argument("--output-dir", help="fallback: the run's output_dir")
    parser.add_argument("--output-name", default="", help="fallback: the run's output_name")
    parser.add_argument("--logging-dir", help="fallback: the run's logging_dir (TensorBoard)")
    parser.add_argument("--since", type=float, help="fallback: unix time the run was launched")
    parser.add_argument("--out", help="copy / download preview images into this folder")
    parser.add_argument("--thumb", action="store_true", help="download 384px JPEG thumbnails")
    args = parser.parse_args(argv)

    result = collect(
        args.base_url, args.task_id, args.output_dir, args.output_name, args.logging_dir, args.since
    )
    if args.out:
        out_dir = Path(args.out)
        if result["source"] == "trainer_api":
            result["saved_previews"] = download_previews(result["previews"], out_dir, args.thumb)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            result["saved_previews"] = [
                str(shutil.copy2(item["path"], out_dir / item["name"]))
                for item in result["previews"]
            ]
    if args.out and result["tags"]:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        loss_file = Path(args.out) / "loss.json"
        loss_file.write_text(json.dumps(result["tags"]), encoding="utf-8")
        result["loss_file"] = str(loss_file)
    result["tags"] = {tag: f"{len(points)} points" for tag, points in result["tags"].items()}
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
