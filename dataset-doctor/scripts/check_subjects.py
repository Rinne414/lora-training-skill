"""Find captions that contradict themselves about how many people are in the image.

Taggers (and hand edits) leave captions like ``1girl, 2girls`` or ``solo, multiple
girls``. The model then learns that one image is both "one person" and "several",
and the subject count stops steering anything. ``solo focus`` is a different tag
(one subject in focus among others) and is never touched.

A caption conflicts when it has

  * two different counts for the same kind (``1girl`` with ``2girls``, ``2boys``
    with ``3boys``), or ``1girl`` with ``multiple girls`` (also boys, others), or
  * ``solo`` together with more than one subject in total (``2girls``, ``multiple
    boys``, or ``1girl`` + ``1boy``).

``resolve_tags`` is the safe fix used by ``fix_dataset.py fix-subject-count``: keep the
higher count, drop ``solo`` when there are several subjects. Tags are matched with
underscores read as spaces (``multiple_girls`` == ``multiple girls``) but are written
back exactly as they were.

Usage:
    python check_subjects.py <path> [--prefer-json] [--no-recursive] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C  # noqa: E402
import check_captions as CC  # noqa: E402

KINDS = ("girl", "boy", "other")
NUMBERED = re.compile(r"^(\d+)\+?(girl|boy|other)s?$")
MULTIPLE = re.compile(r"^multiple (girl|boy|other)s$")
JSON_LIST_KEYS = tuple(sorted(CC.ANIMA_JSON_LIST_KEYS))
MAX_EXAMPLES = CC.MAX_EXAMPLES


def normalize(tag: str) -> str:
    return " ".join(tag.replace("_", " ").split()).lower()


def resolve_tags(tags: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """``(kept tags, [(removed tag, reason)])``; nothing removed means no conflict."""
    numbered: dict[str, list[tuple[int, int]]] = {kind: [] for kind in KINDS}
    multiple: dict[str, list[int]] = {kind: [] for kind in KINDS}
    solo: list[int] = []
    for index, tag in enumerate(tags):
        norm = normalize(tag)
        count = NUMBERED.match(norm)
        many = MULTIPLE.match(norm)
        if count:
            numbered[count.group(2)].append((index, int(count.group(1))))
        elif many:
            multiple[many.group(1)].append(index)
        elif norm == "solo":
            solo.append(index)

    removed: dict[int, str] = {}
    subjects = 0
    for kind in KINDS:
        top = max((n for _, n in numbered[kind]), default=0)
        top_index = next((i for i, n in numbered[kind] if n == top), None)
        for index, number in numbered[kind]:
            if number < top:
                removed[index] = f"'{tags[index]}' contradicts '{tags[top_index]}'"
        if multiple[kind] and 0 < top < 2:
            for index, _ in numbered[kind]:
                removed[index] = f"'{tags[index]}' contradicts '{tags[multiple[kind][0]]}'"
            top = 2
        subjects += top or (2 if multiple[kind] else 0)
    if subjects >= 2:
        for index in solo:
            removed[index] = f"'{tags[index]}' on an image with {subjects} subjects"

    kept = [tag for index, tag in enumerate(tags) if index not in removed]
    return kept, [(tags[index], removed[index]) for index in sorted(removed)]


def _json_entries(obj: dict) -> list[tuple[str, int, str]]:
    """``(key, position, text)`` for the count field and the tag lists of a JSON caption."""
    entries: list[tuple[str, int, str]] = []
    count = obj.get("count")
    if isinstance(count, str):
        entries += [("count", i, t) for i, t in enumerate(CC.split_tags(count))]
    elif isinstance(count, list):
        entries += [("count", i, str(t)) for i, t in enumerate(count)]
    for key in JSON_LIST_KEYS:
        value = obj.get(key)
        if isinstance(value, list):
            entries += [(key, i, str(t)) for i, t in enumerate(value)]
    return entries


def resolve_json(obj: dict) -> tuple[dict, list[tuple[str, str]]]:
    """Same fix for an Anima JSON caption; returns a new dict and what was removed."""
    entries = _json_entries(obj)
    _, removed = resolve_tags([text for _, _, text in entries])
    if not removed:
        return obj, []
    drop = {tag: reason for tag, reason in removed}
    out = dict(obj)
    count = obj.get("count")
    if isinstance(count, (str, list)):
        parts = CC.split_tags(count) if isinstance(count, str) else [str(t) for t in count]
        left = [t for t in parts if t not in drop]
        if left:
            out["count"] = ", ".join(left) if isinstance(count, str) else left
        else:
            out.pop("count")
    for key in JSON_LIST_KEYS:
        if isinstance(obj.get(key), list):
            out[key] = [t for t in obj[key] if str(t) not in drop]
    return out, removed


def caption_tags(caption: Path) -> Optional[list[str]]:
    """The tags a caption file carries (JSON: count field + tag lists); None if unreadable."""
    if caption.suffix.lower() == C.CAPTION_JSON_EXT:
        try:
            obj = json.loads(C.read_text(caption) or "{}")
        except json.JSONDecodeError:
            return None
        return [text for _, _, text in _json_entries(obj)] if isinstance(obj, dict) else None
    return CC.load_caption(caption)["tags"]


def analyze_subjects(root: Path, recursive: bool = True, prefer_json: bool = False) -> list[dict]:
    """At most one ``subject_count_conflict`` issue, as a dict like the other doctor issues."""
    offenders: list[str] = []
    details: list[str] = []
    for image in C.iter_images(root, recursive):
        caption = C.find_caption_path(image, prefer_json)
        tags = caption_tags(caption) if caption else None
        if not tags:
            continue
        _, removed = resolve_tags(tags)
        if removed:
            offenders.append(str(caption))
            details.append(f"{caption.name}: " + "; ".join(reason for _, reason in removed))
    if not offenders:
        return []
    issue = C.Issue(
        C.SEV_MEDIUM,
        "subject_count_conflict",
        f"{len(offenders)} caption(s) contradict themselves about how many people are in the "
        "image (for example 1girl with 2girls, or solo with multiple girls), so the subject "
        "count stops teaching anything.",
        "fix_dataset.py fix-subject-count <dir> (dry-run first): keeps the higher count and drops "
        "solo on images with several subjects; solo focus is left alone",
        offenders[:MAX_EXAMPLES],
    ).to_dict()
    issue["details"] = details[:MAX_EXAMPLES]
    issue["total"] = len(offenders)
    return [issue]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Find captions with conflicting subject counts.")
    parser.add_argument("path")
    parser.add_argument("--prefer-json", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.path)
    if not root.is_dir():
        parser.error(f"path is not a directory: {root}")
    issues = analyze_subjects(root, not args.no_recursive, args.prefer_json)
    if args.json:
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    elif not issues:
        print("no subject-count conflicts")
    else:
        print(f"{issues[0]['code']}: {issues[0]['message']}")
        for line in issues[0]["details"]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
