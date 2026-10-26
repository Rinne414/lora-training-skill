"""Ask the trainer whether a dataset folder is in use before a script rewrites it.

Newer SD-Trainer builds (the story-next fork, branch ``feat/dataset-inuse-guard``;
upstream lora-scripts-next has a similar guard) answer

    GET /api/datasets/in-use?path=<folder>[&recursive=false]

and refuse writes to a folder that a queued or running training still reads
(HTTP 409, ``reason: dataset_in_use``; repeat the write with ``force=true`` to
override). A script that rewrites a dataset folder on disk bypasses that guard,
so it calls ``check_in_use`` first and stops unless the user said to go on.

The check never blocks on trouble: a trainer that is not running, an older
trainer without the route (404) or any network error gives state ``unknown`` and
the script carries on, with a one-line note.

Standard library only. As a command it checks folders and exits 3 when in use:

    python in_use_guard.py --path "D:/work/mychar/dataset/5_mychar"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = os.environ.get("SD_TRAINER_URL", "http://127.0.0.1:28000")
REQUEST_TIMEOUT_SECONDS = 5
EXIT_IN_USE = 3


def check_in_use(
    path: str,
    base_url: str = DEFAULT_BASE_URL,
    recursive: bool = True,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return ``{"state": "free"|"in_use"|"unknown", "tasks": [...], "why": str}``."""
    query = urllib.parse.urlencode({"path": str(path), "recursive": str(bool(recursive)).lower()})
    url = f"{base_url.rstrip('/')}/api/datasets/in-use?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local trusted service
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        why = (
            "this trainer has no in-use check (older build)"
            if error.code == 404
            else f"HTTP {error.code}"
        )
        return {"state": "unknown", "tasks": [], "why": why}
    except (OSError, ValueError) as error:  # URLError, timeouts, bad JSON
        return {"state": "unknown", "tasks": [], "why": f"trainer not reachable ({error})"}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or "in_use" not in data:
        return {"state": "unknown", "tasks": [], "why": "unexpected response from the trainer"}
    tasks = data.get("tasks") or []
    return {"state": "in_use" if data["in_use"] else "free", "tasks": tasks, "why": ""}


def describe_tasks(tasks: list[dict[str, Any]]) -> str:
    return ", ".join(f'"{t.get("name") or t.get("task_id")}" ({t.get("status")})' for t in tasks)


def guard_paths(
    paths: list[str],
    base_url: str = DEFAULT_BASE_URL,
    allow_in_use: bool = False,
) -> int:
    """0 = go ahead, ``EXIT_IN_USE`` = stop. Prints the reason either way."""
    for path in paths:
        result = check_in_use(path, base_url)
        if result["state"] == "unknown":
            print(
                f"note: could not check whether {path} is in use: {result['why']}", file=sys.stderr
            )
        elif result["state"] == "in_use":
            names = describe_tasks(result["tasks"])
            if allow_in_use:
                print(
                    f"warning: {path} is in use by {names}; continuing (--allow-in-use)",
                    file=sys.stderr,
                )
                continue
            print(
                f"STOP: {path} is used by a queued or running training task: {names}.\n"
                "Rewriting it now means captions changed mid-run are only partly picked up and cached "
                "latents / text-encoder outputs will not see the change.\n"
                "Ask the user. If they say go on, repeat with --allow-in-use.",
                file=sys.stderr,
            )
            return EXIT_IN_USE
    return 0


def add_guard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trainer-url", default=DEFAULT_BASE_URL, help="SD-Trainer base URL")
    parser.add_argument(
        "--allow-in-use",
        action="store_true",
        help="go on even if the trainer says a folder is used by a queued or running task",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Is a dataset folder used by a queued or running training?"
    )
    parser.add_argument(
        "--path", action="append", required=True, help="folder to check (repeatable)"
    )
    parser.add_argument("--trainer-url", default=DEFAULT_BASE_URL, help="SD-Trainer base URL")
    args = parser.parse_args()
    status = 0
    for path in args.path:
        result = check_in_use(path, args.trainer_url)
        print(json.dumps({"path": path, **result}, ensure_ascii=False))
        if result["state"] == "in_use":
            status = EXIT_IN_USE
    return status


if __name__ == "__main__":
    raise SystemExit(main())
