"""Two more fixer commands, kept out of fix_dataset.py to keep that file small.

    fix-subject-count   subject_count_conflict  keep the higher subject count, drop solo on
                                                images with several subjects
    quarantine-cache    stale_cache             move stale / orphaned training caches to _quarantine/

They follow fix_dataset.py's safety model: dry-run unless ``apply``, caches are moved to
``_quarantine/`` (never deleted), images are never touched. ``fix_dataset.py`` dispatches here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C  # noqa: E402
import check_cache  # noqa: E402
import check_subjects  # noqa: E402
import fix_dataset as F  # noqa: E402

TEXT_CACHE_HINT = (
    "Caption text changed, so any text-encoder cache built before it (…_te.npz / …_te.safetensors) "
    "is out of date: the doctor reports it as stale_cache; fix_dataset.py quarantine-cache moves it aside."
)


def _rewrite_txt(path: Path, apply: bool, actions: list[F.Action]) -> bool:
    tags, suffix = F.split_txt_caption(C.read_text(path))
    kept, removed = check_subjects.resolve_tags(tags)
    if not removed:
        return False
    detail = (
        "remove "
        + ", ".join(tag for tag, _ in removed)
        + " ("
        + "; ".join(r for _, r in removed)
        + ")"
    )
    actions.append(F.Action("rewrite", str(path), detail))
    F._write_txt_tags(path, kept, suffix, apply)
    return True


def _rewrite_json(path: Path, apply: bool, actions: list[F.Action]) -> bool:
    obj = F._load_json_caption(path)
    if obj is None:
        actions.append(
            F.Action("rewrite", str(path), "SKIPPED: invalid JSON, fix by hand or re-tag")
        )
        return False
    fixed, removed = check_subjects.resolve_json(obj)
    if not removed:
        return False
    detail = (
        "remove "
        + ", ".join(tag for tag, _ in removed)
        + " ("
        + "; ".join(r for _, r in removed)
        + ")"
    )
    actions.append(F.Action("rewrite", str(path), detail))
    F._write_json_caption(path, fixed, apply)
    return True


def cmd_fix_subject_count(root: Path, apply: bool) -> dict:
    """Resolve 1girl-vs-2girls style contradictions and solo-with-several-subjects."""
    actions: list[F.Action] = []
    rewritten = 0
    for image in C.iter_images(root):
        caption = C.find_caption_path(image, prefer_json=True)
        if caption is None:
            continue
        if caption.suffix.lower() == C.CAPTION_JSON_EXT:
            changed = _rewrite_json(caption, apply, actions)
        else:
            changed = _rewrite_txt(caption, apply, actions)
        rewritten += int(changed)
    extra: dict = {"rewritten_captions": rewritten}
    if rewritten:
        extra["after_apply_hint"] = TEXT_CACHE_HINT
    return F._result("fix-subject-count", root, apply, actions, extra)


def cmd_quarantine_cache(root: Path, apply: bool) -> dict:
    """Move stale and orphaned training caches into ``_quarantine/``; never images or captions."""
    actions: list[F.Action] = []
    stale = check_cache.stale_items(root)
    for item in stale:
        reason = f"{item.status}: {item.detail}" if item.detail else item.status
        F.move_file(item.path, F.quarantine_dir(root), reason, apply, actions)
    return F._result("quarantine-cache", root, apply, actions, {"moved_caches": len(stale)})
