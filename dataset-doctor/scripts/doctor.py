"""Dataset-doctor orchestrator: run the image scan and the caption check, then
emit a single PASS / WARN / FAIL readiness verdict for LoRA training.

This is the gate that lora-trainer calls before launching a run. It does not
modify any files; it only reports.

Verdict mapping (by worst issue severity across both sub-reports):
    FAIL  -> a CRITICAL issue (no images, corrupt files) — do not train.
    WARN  -> a HIGH or MEDIUM issue — train only after explicit confirmation.
    PASS  -> only LOW/INFO issues or none — good to go.

Exit code: 0 for PASS/WARN, 2 for FAIL, so a hook or wrapper can gate on it.

Usage:
    python doctor.py <path> [--trigger zkz] [--epochs 10] [--batch-size 1]
                     [--target-reso 1024,1024] [--prefer-json]
                     [--no-recursive] [--json] [--report] [--output FILE]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# The trainer ships an embedded Python whose ._pth file suppresses the usual
# "script dir on sys.path" behaviour, so make sibling imports explicit.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C  # noqa: E402
import scan_dataset  # noqa: E402
import check_cache  # noqa: E402
import check_captions  # noqa: E402
import check_subjects  # noqa: E402

VERDICT_PASS = "PASS"
VERDICT_WARN = "WARN"
VERDICT_FAIL = "FAIL"

_VERDICT_EXIT = {VERDICT_PASS: 0, VERDICT_WARN: 0, VERDICT_FAIL: 2}
_SEVERITY_ORDER = {C.SEV_CRITICAL: 0, C.SEV_HIGH: 1, C.SEV_MEDIUM: 2, C.SEV_LOW: 3, C.SEV_INFO: 4}


def compute_verdict(scan_result: dict, caption_result: dict) -> dict:
    """Merge both sub-reports into a verdict + prioritised recommendations."""
    merged = list(scan_result.get("issues", [])) + list(caption_result.get("issues", []))
    counts = {sev: 0 for sev in (C.SEV_CRITICAL, C.SEV_HIGH, C.SEV_MEDIUM, C.SEV_LOW, C.SEV_INFO)}
    for issue in merged:
        counts[issue["severity"]] = counts.get(issue["severity"], 0) + 1

    if counts[C.SEV_CRITICAL] > 0:
        verdict = VERDICT_FAIL
    elif counts[C.SEV_HIGH] > 0 or counts[C.SEV_MEDIUM] > 0:
        verdict = VERDICT_WARN
    else:
        verdict = VERDICT_PASS

    merged_sorted = sorted(merged, key=lambda i: _SEVERITY_ORDER.get(i["severity"], 99))

    # Prioritised, de-duplicated remediation list (severity order, stable).
    seen: set[str] = set()
    recommendations: list[dict] = []
    for issue in merged_sorted:
        fix = issue.get("fix", "").strip()
        if not fix or fix in seen:
            continue
        seen.add(fix)
        recommendations.append(
            {"severity": issue["severity"], "code": issue["code"], "action": fix}
        )

    return {
        "verdict": verdict,
        "summary": {
            "issue_counts": counts,
            "total_issues": len(merged),
            "images": scan_result.get("totals", {}).get("images", 0),
            "effective_images": scan_result.get("effective_steps", {}).get(
                "total_effective_images", 0
            ),
            "total_steps": scan_result.get("effective_steps", {}).get("total_steps"),
            "missing_captions": caption_result.get("totals", {}).get("missing", 0),
        },
        "issues": merged_sorted,
        "recommendations": recommendations,
    }


def run_doctor(
    root: Path,
    recursive: bool = True,
    prefer_json: bool = False,
    trigger: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: int = 1,
    target_reso: tuple[int, int] = (1024, 1024),
) -> dict:
    scan_result = scan_dataset.analyze_dataset(
        root,
        recursive=recursive,
        epochs=epochs,
        batch_size=batch_size,
        target_reso=target_reso,
        prefer_json=prefer_json,
    )
    caption_result = check_captions.analyze_captions(
        root,
        recursive=recursive,
        prefer_json=prefer_json,
        trigger=trigger,
    )
    # Later checks live in their own modules; their issues join the sub-report they belong to.
    scan_result["issues"] = list(scan_result["issues"]) + check_cache.analyze_cache(root, recursive)
    caption_result["issues"] = list(caption_result["issues"]) + check_subjects.analyze_subjects(
        root, recursive, prefer_json
    )
    verdict = compute_verdict(scan_result, caption_result)
    return {
        "tool": "dataset_doctor",
        "root": str(root),
        "verdict": verdict["verdict"],
        "summary": verdict["summary"],
        "recommendations": verdict["recommendations"],
        "scan": scan_result,
        "captions": caption_result,
    }


def build_markdown(result: dict) -> str:
    badge = {VERDICT_PASS: "✅ PASS", VERDICT_WARN: "⚠️ WARN", VERDICT_FAIL: "⛔ FAIL"}
    s = result["summary"]
    lines = [
        f"# Dataset Doctor — {badge.get(result['verdict'], result['verdict'])}",
        "",
        f"`{result['root']}`",
        "",
    ]
    counts = s["issue_counts"]
    lines.append(
        f"- Images: **{s['images']}** · effective: **{s['effective_images']}**"
        + (f" · total steps: **{s['total_steps']}**" if s.get("total_steps") else "")
    )
    lines.append(
        f"- Issues — critical {counts[C.SEV_CRITICAL]}, high {counts[C.SEV_HIGH]}, "
        f"medium {counts[C.SEV_MEDIUM]}, low {counts[C.SEV_LOW]}, info {counts[C.SEV_INFO]}"
    )
    lines.append("")
    lines.append("## Recommended actions (priority order)")
    if not result["recommendations"]:
        lines.append("Nothing to fix — dataset looks ready. 🎉")
    for i, rec in enumerate(result["recommendations"], 1):
        lines.append(
            f"{i}. {C.severity_emoji(rec['severity'])} **{rec['code']}** — {rec['action']}"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(scan_dataset.build_markdown(result["scan"]))
    lines.append(check_captions.build_markdown(result["captions"]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset readiness gate for LoRA training.")
    parser.add_argument(
        "path", help="train_data_dir (parent of <repeats>_<concept>) or an image folder"
    )
    parser.add_argument("--trigger", default=None, help="expected trigger word / activation tag")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-reso", default="1024,1024", help="WxH, e.g. 1024,1024")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--prefer-json",
        action="store_true",
        help="accept experimental .json captions instead of requiring trainer-supported .txt",
    )
    parser.add_argument("--json", action="store_true", help="print combined JSON only")
    parser.add_argument("--report", action="store_true", help="print markdown only")
    parser.add_argument("--output", default=None, help="write combined JSON to this file")
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.is_dir():
        parser.error(f"path is not a directory: {root}")

    result = run_doctor(
        root,
        recursive=not args.no_recursive,
        prefer_json=args.prefer_json,
        trigger=args.trigger,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_reso=scan_dataset._parse_reso(args.target_reso),
    )

    out = Path(args.output) if args.output else None
    if args.json:
        print(C.dump_json(result, out))
    elif args.report:
        print(build_markdown(result))
        if out:
            C.dump_json(result, out)
    else:
        print(build_markdown(result))
        print()
        print(C.dump_json(result, out))

    return _VERDICT_EXIT.get(result["verdict"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
