"""Phase 3b — BUILD DATASET. Copy curated keeps into the kohya
`<repeats>_<concept>` concept folder, normalizing every image to RGB.

RGB images are copied byte-for-byte (no re-encode). RGBA/P/LA images are
flattened onto white and saved as PNG. Originals stay in raw/ (reversible).
Repeats default to the ~1500-step formula clamp(round(150 / keeps), 1, 10).

Usage:
  python build_dataset.py --work "D:/work/mychar" --concept mychar
  python build_dataset.py --work "D:/work/mychar" --concept mychar --repeats 5

If the trainer is running and reports the folder as used by a queued or running
training, this stops (exit 3) before it replaces anything; pass --allow-in-use
once the user has agreed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import in_use_guard  # noqa: E402


def pick_repeats(n_keep: int) -> int:
    if n_keep <= 0:
        return 1
    return max(1, min(10, round(150 / n_keep)))


def validate_concept_name(concept: str) -> str:
    normalized = concept.strip()
    if not normalized:
        raise ValueError("concept must not be empty")
    if normalized in {".", ".."} or any(char in normalized for char in "/\\:*?[]"):
        raise ValueError(
            f"concept must be a single directory name without /, \\, :, *, ?, [, or ]: {concept!r}"
        )
    return normalized


def to_rgb_png(src: Path, dst: Path) -> None:
    with Image.open(src) as source_image:
        if source_image.mode in ("RGBA", "LA", "P"):
            rgba = source_image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            converted = background
        else:
            converted = source_image.convert("RGB")
        converted.save(dst, "PNG")


def load_keep_records(manifest: Path) -> list[dict[str, str]]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    keep = data.get("keep") if isinstance(data, dict) else None
    if not isinstance(keep, list):
        raise ValueError(f"manifest field 'keep' must be a list: {manifest}")

    records: list[dict[str, str]] = []
    for index, record in enumerate(keep):
        if not isinstance(record, dict):
            raise ValueError(f"manifest keep[{index}] must be an object: {manifest}")
        filename = record.get("file")
        mode = record.get("mode")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"manifest keep[{index}].file must be a filename: {filename!r}")
        if not isinstance(mode, str) or not mode:
            raise ValueError(f"manifest keep[{index}].mode must be a non-empty string")
        records.append({"file": filename, "mode": mode})
    return records


def build_dataset(work: Path, concept: str, repeats: int) -> dict[str, int | str]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    normalized_concept = validate_concept_name(concept)

    raw = work / "raw"
    manifest = work / "curation_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"curation manifest not found: {manifest}")
    records = load_keep_records(manifest)
    sources = [raw / record["file"] for record in records]
    missing = [source for source in sources if not source.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"manifest source image(s) not found: {names}")

    dataset_dir = work / "dataset"
    concept_name = f"{repeats}_{normalized_concept}"
    concept_dir = dataset_dir / concept_name
    staging_dir = dataset_dir / f".{concept_name}.tmp"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    copied = 0
    converted = 0
    try:
        for record, source in zip(records, sources, strict=True):
            if record["mode"] == "RGB":
                shutil.copy2(source, staging_dir / record["file"])
                copied += 1
            else:
                destination = staging_dir / f"{Path(record['file']).stem}.png"
                to_rgb_png(source, destination)
                converted += 1

        if concept_dir.exists():
            shutil.rmtree(concept_dir)
        staging_dir.replace(concept_dir)
        for stale_dir in dataset_dir.glob(f"*_{normalized_concept}"):
            if stale_dir != concept_dir and stale_dir.is_dir():
                shutil.rmtree(stale_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "concept_dir": str(concept_dir),
        "repeats": repeats,
        "keeps": len(records),
        "copied": copied,
        "converted": converted,
        "images": copied + converted,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the kohya concept folder from curated keeps.")
    ap.add_argument(
        "--work",
        required=True,
        help="project dir (has raw/ and curation_manifest.json)",
    )
    ap.add_argument("--concept", required=True, help="concept/trigger folder name, e.g. mychar")
    in_use_guard.add_guard_arguments(ap)
    ap.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="override repeats (0 = auto from keep count)",
    )
    a = ap.parse_args()

    work = Path(a.work).resolve()
    manifest = work / "curation_manifest.json"
    if not manifest.exists():
        print(f"ERROR: {manifest} missing — run curate.py first.")
        return 2

    keep_count = len(load_keep_records(manifest))
    repeats = a.repeats or pick_repeats(keep_count)
    # rebuilding replaces this concept folder and any older "<n>_<concept>" sibling
    touched = [work / "dataset" / f"{repeats}_{a.concept.strip()}"]
    touched += sorted(
        str(p) for p in (work / "dataset").glob(f"*_{a.concept.strip()}") if p.is_dir()
    )
    blocked = in_use_guard.guard_paths([str(p) for p in touched], a.trainer_url, a.allow_in_use)
    if blocked:
        return blocked
    result = build_dataset(work, a.concept, repeats)
    print(f"concept folder : {result['concept_dir']}")
    print(f"repeats        : {result['repeats']}  (keeps={result['keeps']})")
    print(
        f"copied RGB     : {result['copied']}   converted->RGB png: "
        f"{result['converted']}   total: {result['images']}"
    )
    print(
        f'\nnext: tag_dataset.py --dataset-dir "{result["concept_dir"]}" '
        "--trigger <TRIGGER> --subject-tag <1girl|1boy|1other>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
