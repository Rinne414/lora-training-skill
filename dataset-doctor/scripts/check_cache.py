"""Find training cache files that no longer match the dataset.

Trainers cache latents (and often text-encoder outputs) next to the images so later
runs skip the encoding. If you then edit an image, or a caption, and train again, the
trainer finds the old cache and trains on the OLD latent / OLD caption without a word.
Caches whose image is gone are clutter that hides this.

Recognised caches (files are only matched to an image in the same folder):

  * kohya / sd-scripts   ``<image>.npz``, ``<image>_<W>x<H>_<arch>.npz`` (latents) and
                         ``<image>_<arch>_te.npz`` (text-encoder output)
  * musubi-tuner style   ``<image>_<W>x<H>_<arch>.safetensors`` and ``<image>_<arch>_te.safetensors``;
                         a file only counts when its safetensors HEADER carries an
                         ``architecture`` entry, so a LoRA or model that happens to have such a
                         name is left alone
  * a ``_latent_cache`` or ``*_latent_cache`` folder (entries are named by hash, so the folder is
                         compared with the newest image beside it)

Status of each cache: ``stale_image`` (older than its image), ``stale_caption`` (a text-encoder
cache older than its caption), ``orphaned`` (no image of that name), else ``fresh``. Comparison is
by file modified time with a 2 s tolerance; if the dataset was copied without keeping modified
times, look at the list before acting on it. Nothing here changes a file.

Usage:
    python check_cache.py <path> [--no-recursive] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C  # noqa: E402

MTIME_TOLERANCE_SECONDS = 2.0
MAX_EXAMPLES = 12
MAX_HEADER_BYTES = 64 * 1024 * 1024
CACHE_FOLDER_SUFFIX = "_latent_cache"
# What a trainer appends to the image name: optional bucket size, optional architecture, optional "_te".
TAIL = re.compile(r"^(?:_\d{3,5}x\d{3,5})?(?:_[A-Za-z0-9.]{1,12})?(?:_te)?$")
MUSUBI_NAME = re.compile(r"^.+_(?:\d{3,5}x\d{3,5}_[A-Za-z0-9]+|[A-Za-z0-9]+_te)$", re.IGNORECASE)

STATUS_FRESH = "fresh"
STATUS_STALE_IMAGE = "stale_image"
STATUS_STALE_CAPTION = "stale_caption"
STATUS_ORPHANED = "orphaned"
STALE_STATUSES = (STATUS_STALE_IMAGE, STATUS_STALE_CAPTION, STATUS_ORPHANED)


@dataclass(frozen=True)
class CacheItem:
    path: Path
    kind: str  # npz | safetensors | folder
    role: str  # latents | text | folder
    status: str
    detail: str


def safetensors_metadata(path: Path) -> Optional[dict]:
    """The ``__metadata__`` of a safetensors file, reading only its header; None if unreadable."""
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None
            (length,) = struct.unpack("<Q", raw_length)
            if not 0 < length <= MAX_HEADER_BYTES:
                return None
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError):
        return None
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return meta if isinstance(meta, dict) else {}


def _is_musubi_cache(path: Path) -> bool:
    if not MUSUBI_NAME.match(path.stem):
        return False
    meta = safetensors_metadata(path)
    return bool(meta and str(meta.get("architecture") or "").strip())


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _owner(stem: str, images: dict[str, list[Path]]) -> tuple[Optional[str], str]:
    """The image stem a cache named ``stem`` belongs to (longest match) and the leftover tail."""
    best: Optional[str] = None
    for image_stem in images:
        if stem.startswith(image_stem) and TAIL.match(stem[len(image_stem) :]):
            if best is None or len(image_stem) > len(best):
                best = image_stem
    return best, stem[len(best) :] if best is not None else ""


def _caption_for(image: Path) -> Optional[Path]:
    return C.find_caption_path(image, prefer_json=True)


def _judge_file(path: Path, kind: str, images: dict[str, list[Path]]) -> CacheItem:
    stem = path.name[: -len(path.suffix)]
    owner, tail = _owner(stem, images)
    if owner is None:
        return CacheItem(
            path, kind, "latents", STATUS_ORPHANED, "no image with this name beside it"
        )
    is_text = tail.endswith("_te")
    role = "text" if is_text else "latents"
    cache_time = _mtime(path)
    if is_text:
        for image in images[owner]:
            caption = _caption_for(image)
            if caption and _mtime(caption) > cache_time + MTIME_TOLERANCE_SECONDS:
                return CacheItem(
                    path,
                    kind,
                    role,
                    STATUS_STALE_CAPTION,
                    f"caption {caption.name} was edited later",
                )
        return CacheItem(path, kind, role, STATUS_FRESH, "")
    newest = max(images[owner], key=_mtime)
    if _mtime(newest) > cache_time + MTIME_TOLERANCE_SECONDS:
        return CacheItem(
            path, kind, role, STATUS_STALE_IMAGE, f"image {newest.name} was edited later"
        )
    return CacheItem(path, kind, role, STATUS_FRESH, "")


def _tree_files(folder: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d != C.QUARANTINE_DIR_NAME]
        found.extend(Path(dirpath, name) for name in filenames)
    return found


def _judge_folder(folder: Path, beside: Path) -> CacheItem:
    images = [
        p
        for p in _tree_files(beside)
        if C.is_image(p) and folder not in p.parents and C.QUARANTINE_DIR_NAME not in p.parts
    ]
    if not images:
        return CacheItem(folder, "folder", "folder", STATUS_ORPHANED, "no images beside this cache")
    inside = [_mtime(p) for p in _tree_files(folder)] or [_mtime(folder)]
    newest_image = max(images, key=_mtime)
    if _mtime(newest_image) > max(inside) + MTIME_TOLERANCE_SECONDS:
        return CacheItem(
            folder,
            "folder",
            "folder",
            STATUS_STALE_IMAGE,
            f"image {newest_image.name} is newer than everything in the cache",
        )
    return CacheItem(folder, "folder", "folder", STATUS_FRESH, "")


def find_caches(root: Path, recursive: bool = True) -> list[CacheItem]:
    """Every recognised cache under ``root`` with its status (``_quarantine`` is skipped)."""
    items: list[CacheItem] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != C.QUARANTINE_DIR_NAME)
        here = Path(dirpath)
        for name in list(dirnames):
            if name.endswith(CACHE_FOLDER_SUFFIX):
                items.append(_judge_folder(here / name, here))
                dirnames.remove(name)
        images: dict[str, list[Path]] = {}
        for name in filenames:
            path = here / name
            if C.is_image(path):
                images.setdefault(path.stem, []).append(path)
        for name in sorted(filenames):
            path = here / name
            suffix = path.suffix.lower()
            if suffix == ".npz":
                items.append(_judge_file(path, "npz", images))
            elif suffix == ".safetensors" and _is_musubi_cache(path):
                items.append(_judge_file(path, "safetensors", images))
        if not recursive:
            break
    return items


def stale_items(root: Path, recursive: bool = True) -> list[CacheItem]:
    return [item for item in find_caches(root, recursive) if item.status in STALE_STATUSES]


def _count(items: Iterable[CacheItem], status: str) -> int:
    return sum(1 for item in items if item.status == status)


def analyze_cache(root: Path, recursive: bool = True) -> list[dict]:
    """At most one ``stale_cache`` issue, as a dict like the other doctor issues."""
    stale = stale_items(root, recursive)
    if not stale:
        return []
    by_image = _count(stale, STATUS_STALE_IMAGE)
    by_caption = _count(stale, STATUS_STALE_CAPTION)
    orphaned = _count(stale, STATUS_ORPHANED)
    issue = C.Issue(
        C.SEV_MEDIUM,
        "stale_cache",
        f"{len(stale)} training cache item(s) no longer match the dataset: {by_image} older than "
        f"their image, {by_caption} older than their caption, {orphaned} orphaned (image gone). "
        "Training would reuse the old latent / old caption text instead of your edited files.",
        "fix_dataset.py quarantine-cache <dir> (dry-run first): moves only these cache files to "
        "_quarantine/ so the trainer rebuilds them; images and captions are never touched",
        [str(item.path) for item in stale[:MAX_EXAMPLES]],
    ).to_dict()
    issue["details"] = [
        f"{item.path.name}: {item.status} - {item.detail}" for item in stale[:MAX_EXAMPLES]
    ]
    issue["total"] = len(stale)
    return [issue]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Find stale or orphaned training caches.")
    parser.add_argument("path")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.path)
    if not root.is_dir():
        parser.error(f"path is not a directory: {root}")
    items = find_caches(root, not args.no_recursive)
    if args.json:
        rows = [
            {
                "path": str(i.path),
                "kind": i.kind,
                "role": i.role,
                "status": i.status,
                "detail": i.detail,
            }
            for i in items
        ]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for item in items:
            print(f"{item.status:14} {item.path}  {item.detail}")
        if not items:
            print("no training caches found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
