"""Apply the safe, mechanical fixes that dataset-doctor recommends.

One command per doctor issue code, so a confirmed fix is a single line instead
of hand-rolled shell. Safety model:

  * **Dry-run by default** — every command only *prints* what it would do.
    Nothing changes until you re-run with ``--apply``.
  * **Quarantine, never delete** — files are moved to ``_quarantine/`` inside
    the dataset (with their caption sidecars). The doctor and the trainer both
    ignore that folder, and you can restore anything by moving it back.
  * Caption rewrites keep a one-line plan per file; originals of converted
    images are backed up to quarantine first.

Commands (issue code they fix):

    organize            no_concept_folders   move loose images into <repeats>_<concept>/
    quarantine-corrupt  corrupt_images       set aside undecodable images
    dedupe              exact_duplicates     keep the best copy per duplicate group
                        near_duplicates      (with --near)
    to-rgb              non_rgb_mode         convert RGBA/P/CMYK images to RGB
    add-trigger         trigger_inconsistent ensure the trigger exists in every caption
    strip-tags          artifact_tags        remove source-noise + unwanted tags
                        duplicate_tags       (also de-duplicates tags per caption)
    fix-subject-count   subject_count_conflict  keep the higher count (1girl vs 2girls), drop solo
                                             on images with several subjects
    quarantine-cache    stale_cache          set aside stale / orphaned training caches

Usage:
    python fix_dataset.py <command> <path> [options] [--apply] [--json]

If the trainer is running and reports the folder as used by a queued or running training,
``--apply`` stops with exit 3 before changing anything; pass ``--allow-in-use`` once the user
has agreed. A trainer that is down or too old only prints a note. Dry-runs never ask.

Exit code: 0 on success (dry-run or applied), 1 on bad arguments, 3 when the dataset is in use.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The trainer ships an embedded Python whose ._pth file suppresses the usual
# "script dir on sys.path" behaviour, so make sibling imports explicit.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C  # noqa: E402
import check_captions as CC  # noqa: E402
import scan_dataset  # noqa: E402

MAX_PRINT = 20  # cap per-action lines shown on the console
ANIMA_PREFIX_TAGS: frozenset[str] = frozenset(
    {
        "masterpiece",
        "best quality",
        "good quality",
        "normal quality",
        "low quality",
        "worst quality",
        "newest",
        "recent",
        "mid",
        "early",
        "old",
        "highres",
        "absurdres",
        "safe",
        "sensitive",
        "nsfw",
        "explicit",
    }
)
ANIMA_COUNT_PATTERN = re.compile(r"^\d+(?:girl|boy|other)s?$")


# --- Action plan model ---------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One planned (or executed) change. ``kind`` is a stable machine key."""

    kind: str  # move | rewrite | convert | mkdir
    path: str
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path, "detail": self.detail}


def _result(
    command: str, root: Path, apply: bool, actions: list[Action], extra: Optional[dict] = None
) -> dict:
    out = {
        "tool": "fix_dataset",
        "command": command,
        "root": str(root),
        "applied": apply,
        "action_count": len(actions),
        "actions": [a.to_dict() for a in actions],
    }
    if extra:
        out.update(extra)
    return out


# --- Filesystem helpers --------------------------------------------------------


def quarantine_dir(root: Path) -> Path:
    return root / C.QUARANTINE_DIR_NAME


def unique_dest(folder: Path, name: str) -> Path:
    """A non-colliding destination path inside ``folder``."""
    dest = folder / name
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while dest.exists():
        dest = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    return dest


def sidecar_paths(image: Path) -> list[Path]:
    """Existing caption sidecars (.txt and/or .json) next to ``image``."""
    return [
        p
        for p in (image.with_suffix(C.CAPTION_TXT_EXT), image.with_suffix(C.CAPTION_JSON_EXT))
        if p.is_file()
    ]


def move_file(src: Path, dest_dir: Path, reason: str, apply: bool, actions: list[Action]) -> None:
    """Plan (and with ``apply`` perform) a move of ``src`` into ``dest_dir``."""
    dest = unique_dest(dest_dir, src.name)
    actions.append(Action("move", str(src), f"-> {dest} ({reason})"))
    if apply:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


def move_with_sidecars(
    image: Path,
    dest_dir: Path,
    reason: str,
    apply: bool,
    actions: list[Action],
    reserved: set[Path],
) -> None:
    sources = [image, *sidecar_paths(image)]
    counter = 0
    while True:
        suffix = "" if counter == 0 else f"_{counter}"
        stem = image.stem + suffix
        destinations = [dest_dir / f"{stem}{source.suffix}" for source in sources]
        if all(
            not destination.exists() and destination not in reserved for destination in destinations
        ):
            break
        counter += 1

    reserved.update(destinations)
    for source, destination in zip(sources, destinations, strict=True):
        item_reason = reason if source == image else "caption of " + image.name
        actions.append(Action("move", str(source), f"-> {destination} ({item_reason})"))
        if apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))


# --- organize ------------------------------------------------------------------


def cmd_organize(root: Path, repeats: int, concept: str, apply: bool) -> dict:
    """Move images sitting directly under ``root`` into ``<repeats>_<concept>/``."""
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not concept.strip():
        raise ValueError("--concept must not be empty")
    dest_dir = root / f"{repeats}_{concept.strip()}"

    loose = C.iter_images(root, recursive=False)
    actions: list[Action] = []
    reserved: set[Path] = set()
    if loose:
        actions.append(Action("mkdir", str(dest_dir), "concept folder"))
        if apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
        for image in loose:
            move_with_sidecars(image, dest_dir, "organize", apply, actions, reserved)
    return _result(
        "organize", root, apply, actions, {"moved_images": len(loose), "concept_dir": str(dest_dir)}
    )


# --- quarantine-corrupt ----------------------------------------------------------


def cmd_quarantine_corrupt(root: Path, apply: bool) -> dict:
    """Move images Pillow cannot decode into quarantine (training would crash)."""
    actions: list[Action] = []
    reserved: set[Path] = set()
    corrupt = 0
    for image in C.iter_images(root):
        record = scan_dataset.probe_image(image)
        if "error" in record:
            corrupt += 1
            move_with_sidecars(
                image, quarantine_dir(root), record["error"], apply, actions, reserved
            )
    return _result("quarantine-corrupt", root, apply, actions, {"corrupt_images": corrupt})


# --- dedupe ----------------------------------------------------------------------


def _keep_best(group_records: list[dict]) -> tuple[dict, list[dict]]:
    """Rank a duplicate group: keep highest resolution, then largest file."""
    ranked = sorted(
        group_records,
        key=lambda r: (r.get("megapixels", 0), r.get("filesize", 0), r["path"]),
        reverse=True,
    )
    return ranked[0], ranked[1:]


def merge_duplicate_groups(groups: list[list[str]]) -> list[list[str]]:
    """Merge exact and perceptual groups into disjoint path clusters."""
    parent: dict[str, str] = {}

    def find(path: str) -> str:
        parent.setdefault(path, path)
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for group in groups:
        if not group:
            continue
        for path in group[1:]:
            union(group[0], path)

    clusters: dict[str, list[str]] = {}
    for path in parent:
        clusters.setdefault(find(path), []).append(path)
    return sorted(sorted(cluster) for cluster in clusters.values() if len(cluster) > 1)


def cmd_dedupe(root: Path, near: bool, apply: bool) -> dict:
    """Quarantine redundant copies in exact (and, with --near, near) dup groups."""
    paths = C.iter_images(root)
    records = [scan_dataset.probe_image(p) for p in paths]
    by_path = {r["path"]: r for r in records if "error" not in r}

    groups = scan_dataset.group_exact_duplicates(records, paths)
    if near:
        good = [r for r in records if "error" not in r]
        if len(good) > scan_dataset.NEAR_DUP_MAX_IMAGES:
            raise ValueError(
                f"--near needs <= {scan_dataset.NEAR_DUP_MAX_IMAGES} images "
                f"(found {len(good)}); dedupe exact duplicates first"
            )
        groups += scan_dataset.group_near_duplicates(good)
    groups = merge_duplicate_groups(groups)

    actions: list[Action] = []
    reserved: set[Path] = set()
    removed = 0
    for group in groups:
        group_records = [by_path[p] for p in group if p in by_path]
        if len(group_records) < 2:
            continue
        best, rest = _keep_best(group_records)
        for record in rest:
            removed += 1
            move_with_sidecars(
                Path(record["path"]),
                quarantine_dir(root),
                f"duplicate of {Path(best['path']).name}",
                apply,
                actions,
                reserved,
            )
    return _result(
        "dedupe",
        root,
        apply,
        actions,
        {"duplicate_groups": len(groups), "quarantined_images": removed},
    )


# --- to-rgb ----------------------------------------------------------------------


def _convert_to_rgb(image_path: Path) -> None:
    """Convert in place; alpha is composited over white to avoid black halos."""
    from PIL import Image

    with Image.open(image_path) as img:
        fmt = img.format
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            converted = background
        else:
            converted = img.convert("RGB")
        save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
        converted.save(image_path, format=fmt, **save_kwargs)


def cmd_to_rgb(root: Path, apply: bool) -> dict:
    """Convert non-RGB images to RGB, backing up the original to quarantine."""
    actions: list[Action] = []
    converted = 0
    for image in C.iter_images(root):
        record = scan_dataset.probe_image(image)
        if "error" in record or record["mode"] == "RGB":
            continue
        converted += 1
        backup = unique_dest(quarantine_dir(root), image.name)
        actions.append(
            Action(
                "convert", str(image), f"{record['mode']} -> RGB (original backed up to {backup})"
            )
        )
        if apply:
            quarantine_dir(root).mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(image), str(backup))
            _convert_to_rgb(image)
    return _result("to-rgb", root, apply, actions, {"converted_images": converted})


# --- caption rewriting helpers ----------------------------------------------------


def split_txt_caption(text: str) -> tuple[list[str], str]:
    tag_block, separator, prose = text.strip().partition(". ")
    suffix = f". {prose}" if separator else ""
    return CC.split_tags(tag_block), suffix


def _write_txt_tags(path: Path, tags: list[str], suffix: str, apply: bool) -> None:
    if apply:
        path.write_text(", ".join(tags) + suffix, encoding="utf-8")


def trigger_insert_index(tags: list[str]) -> int:
    index = 0
    for tag in tags:
        if (
            tag in ANIMA_PREFIX_TAGS
            or tag.startswith("score_")
            or tag.startswith("year ")
            or ANIMA_COUNT_PATTERN.fullmatch(tag)
        ):
            index += 1
            continue
        break
    return index


def _ordered_anima_json(obj: dict) -> dict:
    """Rebuild a caption dict in the recommended Anima key order."""
    ordered = {key: obj[key] for key in CC.ANIMA_JSON_ORDER if key in obj}
    ordered.update({key: obj[key] for key in obj if key not in ordered})
    return ordered


def _write_json_caption(path: Path, obj: dict, apply: bool) -> None:
    if apply:
        text = json.dumps(_ordered_anima_json(obj), ensure_ascii=False, indent=2)
        path.write_text(text, encoding="utf-8")


def _load_json_caption(path: Path) -> Optional[dict]:
    try:
        obj = json.loads(C.read_text(path))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# --- add-trigger -------------------------------------------------------------------


def _add_trigger_txt(path: Path, trigger: str, apply: bool, actions: list[Action]) -> bool:
    tags, suffix = split_txt_caption(C.read_text(path))
    key = trigger.strip().lower()
    if key in tags:
        return False
    index = trigger_insert_index(tags)
    new_tags = tags[:index] + [key] + tags[index:]
    actions.append(Action("rewrite", str(path), f"insert trigger '{trigger}' at tag {index + 1}"))
    _write_txt_tags(path, new_tags, suffix, apply)
    return True


def _add_trigger_json(path: Path, trigger: str, apply: bool, actions: list[Action]) -> bool:
    obj = _load_json_caption(path)
    if obj is None:
        actions.append(Action("rewrite", str(path), "SKIPPED: invalid JSON, fix by hand or re-tag"))
        return False
    current = str(obj.get("character", "")).strip()
    if current.lower() == trigger.strip().lower():
        return False
    detail = f"character: '{current}' -> '{trigger}'" if current else f"set character '{trigger}'"
    obj["character"] = trigger.strip()
    actions.append(Action("rewrite", str(path), detail))
    _write_json_caption(path, obj, apply)
    return True


def cmd_add_trigger(root: Path, trigger: str, apply: bool) -> dict:
    """Ensure ``trigger`` exists in every .txt / JSON caption."""
    if not trigger.strip():
        raise ValueError("--trigger must not be empty")
    actions: list[Action] = []
    rewritten = 0
    missing = 0
    for image in C.iter_images(root):
        caption = C.find_caption_path(image, prefer_json=True)
        if caption is None:
            missing += 1
            continue
        if caption.suffix.lower() == C.CAPTION_JSON_EXT:
            changed = _add_trigger_json(caption, trigger, apply, actions)
        else:
            changed = _add_trigger_txt(caption, trigger, apply, actions)
        rewritten += int(changed)
    return _result(
        "add-trigger",
        root,
        apply,
        actions,
        {
            "rewritten_captions": rewritten,
            "missing_captions": missing,
            "hint_for_missing": "run tag_dataset.py for Anima or the trainer WD14 tagger"
            if missing
            else "",
        },
    )


# --- strip-tags --------------------------------------------------------------------


def _strip_txt(path: Path, drop: frozenset[str], apply: bool, actions: list[Action]) -> bool:
    tags, suffix = split_txt_caption(C.read_text(path))
    kept: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in drop or tag in seen:
            continue
        seen.add(tag)
        kept.append(tag)
    if kept == tags:
        return False
    removed = len(tags) - len(kept)
    actions.append(Action("rewrite", str(path), f"removed {removed} tag(s)"))
    _write_txt_tags(path, kept, suffix, apply)
    return True


def _strip_json(path: Path, drop: frozenset[str], apply: bool, actions: list[Action]) -> bool:
    obj = _load_json_caption(path)
    if obj is None:
        return False
    removed = 0
    for key in CC.ANIMA_JSON_LIST_KEYS:
        value = obj.get(key)
        if not isinstance(value, list):
            continue
        kept: list = []
        seen: set[str] = set()
        for item in value:
            norm = str(item).strip().lower()
            if norm in drop or norm in seen:
                removed += 1
                continue
            seen.add(norm)
            kept.append(item)
        obj[key] = kept
    if removed == 0:
        return False
    actions.append(Action("rewrite", str(path), f"removed {removed} tag(s)"))
    _write_json_caption(path, obj, apply)
    return True


def cmd_strip_tags(root: Path, extra_tags: str, apply: bool) -> dict:
    """Remove artefact tags (and ``--tags``) plus in-caption duplicates."""
    drop = frozenset(CC.ARTIFACT_TAGS | set(CC.split_tags(extra_tags or "")))
    actions: list[Action] = []
    rewritten = 0
    for image in C.iter_images(root):
        caption = C.find_caption_path(image, prefer_json=True)
        if caption is None:
            continue
        if caption.suffix.lower() == C.CAPTION_JSON_EXT:
            changed = _strip_json(caption, drop, apply, actions)
        else:
            changed = _strip_txt(caption, drop, apply, actions)
        rewritten += int(changed)
    return _result(
        "strip-tags",
        root,
        apply,
        actions,
        {"rewritten_captions": rewritten, "dropped_tag_set": sorted(drop)},
    )


# --- CLI -----------------------------------------------------------------------------


def _print_human(result: dict) -> None:
    mode = "APPLIED" if result["applied"] else "DRY-RUN (nothing changed)"
    print(f"fix_dataset {result['command']} - {mode}")
    print(f"root: {result['root']}")
    for action in result["actions"][:MAX_PRINT]:
        print(f"  [{action['kind']}] {action['path']}  {action['detail']}")
    hidden = result["action_count"] - min(result["action_count"], MAX_PRINT)
    if hidden > 0:
        print(f"  ... and {hidden} more action(s)")
    if result["action_count"] == 0:
        print("  nothing to do - dataset already clean for this fix")
    elif not result["applied"]:
        print(
            "re-run with --apply to execute (moved files go to _quarantine/, never deleted; "
            "caption rewrites are made in place)"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply dataset-doctor fixes (dry-run by default; --apply to execute)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("path", help="train_data_dir or a <repeats>_<concept> folder")
        p.add_argument("--apply", action="store_true", help="actually make the changes")
        p.add_argument("--json", action="store_true", help="print the JSON plan/result")
        p.add_argument(
            "--trainer-url",
            default=os.environ.get("SD_TRAINER_URL", "http://127.0.0.1:28000"),
            help="SD-Trainer base URL, asked before --apply whether the dataset is in use",
        )
        p.add_argument(
            "--allow-in-use",
            action="store_true",
            help="apply even if the trainer says a queued or running task uses this folder",
        )
        return p

    p_org = add("organize", "move loose images into a <repeats>_<concept> folder")
    p_org.add_argument("--repeats", type=int, required=True)
    p_org.add_argument("--concept", required=True)

    add("quarantine-corrupt", "set aside images that cannot be decoded")

    p_dedupe = add("dedupe", "quarantine redundant duplicate images")
    p_dedupe.add_argument(
        "--near", action="store_true", help="also collapse visually near-identical clusters"
    )

    add("to-rgb", "convert non-RGB images to RGB (originals backed up)")

    p_trig = add("add-trigger", "ensure the trigger word exists in every caption")
    p_trig.add_argument("--trigger", required=True)

    p_strip = add("strip-tags", "remove artefact tags and per-caption duplicate tags")
    p_strip.add_argument("--tags", default="", help="extra comma-separated tags to remove")

    add(
        "fix-subject-count",
        "resolve 1girl-vs-2girls style conflicts and solo with several subjects",
    )
    add(
        "quarantine-cache", "set aside stale or orphaned training caches (images are never touched)"
    )

    args = parser.parse_args(argv)
    root = Path(args.path)
    if not root.is_dir():
        parser.error(f"path is not a directory: {root}")

    if args.apply:
        import fix_extras  # imported here because it imports this module

        blocked = fix_extras.guard_apply(root, args.trainer_url, args.allow_in_use)
        if blocked:
            return blocked

    try:
        if args.command == "organize":
            result = cmd_organize(root, args.repeats, args.concept, args.apply)
        elif args.command == "quarantine-corrupt":
            result = cmd_quarantine_corrupt(root, args.apply)
        elif args.command == "dedupe":
            result = cmd_dedupe(root, args.near, args.apply)
        elif args.command == "to-rgb":
            result = cmd_to_rgb(root, args.apply)
        elif args.command == "add-trigger":
            result = cmd_add_trigger(root, args.trigger, args.apply)
        elif args.command == "strip-tags":
            result = cmd_strip_tags(root, args.tags, args.apply)
        else:
            import fix_extras  # imported here because it imports this module

            run = {
                "fix-subject-count": fix_extras.cmd_fix_subject_count,
                "quarantine-cache": fix_extras.cmd_quarantine_cache,
            }[args.command]
            result = run(root, args.apply)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(C.dump_json(result))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
