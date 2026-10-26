"""Phase 2 — TAG (Anima-correct). Drive sd-image-sorter to auto-tag a concept folder
and export **Anima-format** captions: model-card section order, @artist prefix, safety
vocab, single line — and, for character LoRAs, the caption-paradox trait prune.

Why this is not a naive WD14 dump (the whole point — see references/collect-and-tag.md
and the public model documentation):
- Anima was pretrained with random tag dropout, so every relevant tag is not required.
  Prefer reviewed, precision-oriented captions because invented tags describe the wrong
  training target; this is an operating rule, not a measured false-positive cost ratio.
- WD14 emits tags in CONFIDENCE order, which interleaves character/artist among generals
  and VIOLATES the Anima model-card order. We export via the `anima` template so tags are
  sectioned: quality → safety → count → trigger → characters → copyright → @artists →
  general → NL, single-lined (kohya reads only line 1).
- The CAPTION PARADOX: for a character LoRA, invariant identity (hair/eye color, body
  traits) should bake into the TRIGGER, not be re-stated in every caption. We fetch trait
  candidates and blacklist the high-frequency ones so the trigger absorbs identity. This
  uses a deterministic frequency heuristic and records every candidate and decision in
  an audit file. The audit makes the decision reviewable; it is not human approval.

Flow (sd-image-sorter, http://127.0.0.1:8487, env SORTER_URL):
  POST /api/scan                    index the folder
  POST /api/images/selection-token  scope a token to the folder; expand ids via /selection-chunk
  POST /api/tag                     WD14 (eva02 default); poll /api/tag/progress
  POST /api/tags/trait-candidates   (character) list identity traits ≥ min_ratio
  POST /api/tags/export-batch       content_mode=template, preset=anima, trigger, prune blacklist

Usage:
  python tag_dataset.py --dataset-dir "D:/work/mychar/dataset/5_mych4r" --trigger mych4r --subject-tag 1girl
  python tag_dataset.py --dataset-dir "D:/work/mychar/dataset/5_mych4r" --trigger mych4r --subject-tag 1girl --model wd-swinv2-tagger-v3
  python tag_dataset.py --dataset-dir "D:/work/style/dataset/5_style" --trigger @mystyle --style
  python tag_dataset.py ... --consensus          # require agreement from 2 of 3 taggers
  python tag_dataset.py ... --no-trait-prune     # keep identity traits in captions
If the trainer is running and reports the folder as used by a queued or running
training, this stops (exit 3) before it writes any caption; pass --allow-in-use once
the user has agreed.
Environment overrides:
  SORTER_URL   base URL of sd-image-sorter (default http://127.0.0.1:8487)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset_profile as profile  # noqa: E402
import in_use_guard  # noqa: E402

SORTER = os.environ.get("SORTER_URL", "http://127.0.0.1:8487").rstrip("/")
REQUEST_ATTEMPTS = 3
TAGGER_GENERAL_THRESHOLDS: dict[str, float] = {
    "wd-eva02-large-tagger-v3": 0.5296,
    "wd-swinv2-tagger-v3": 0.2653,
}
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class TraitCandidate(TypedDict):
    tag: str
    family: str
    ratio: float
    count: int


class SemanticWarning(TypedDict):
    caption: str
    tags: list[str]


# Never store these as tags: negative-prompt / meta noise + WD14 rating words.
# (Rating is placed once in the template's {safety} slot; these must not leak as tags.)
PRE_TAG_BLACKLIST = [
    "general",
    "sensitive",
    "questionable",
    "explicit",
    "watermark",
    "signature",
    "artist_name",
    "web_address",
    "jpeg_artifacts",
    "text",
    "english_text",
    "commentary",
    "commentary_request",
    "logo",
    "username",
    "dated",
    "twitter_username",
    "patreon_username",
    "text_focus",
]

# NOTE on STYLE LoRAs: style/era/medium/artist tags (1990s_(style), retro_artstyle,
# *_(medium), @artist, …) are stripped by the SORTER when we export with
# training_purpose="style" — it drops them by tag CATEGORY (semantic classifier,
# pattern-based), not by any hand-list here. So the trigger absorbs the look and
# there is deliberately no style word-list in this file: a list can't enumerate
# every style tag, but the category rule generalises. (Requires the sorter build
# with the tag_training_filters category-union fix.)


def _req(
    path: str, body: dict | None = None, method: str | None = None, timeout: int = 120
) -> dict:
    url = SORTER + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"} if data else {},
    )
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — local trusted service
                response = json.loads(r.read().decode())
            if not isinstance(response, dict):
                raise TypeError(
                    f"sorter response must be an object: path={path} response={response!r}"
                )
            return response
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            detail = {
                "path": path,
                "status": exc.code,
                "response": response_body,
                "attempt": attempt,
            }
            if attempt == REQUEST_ATTEMPTS:
                raise RuntimeError(f"sorter request failed: {json.dumps(detail)}") from exc
            print(
                "WARNING: sorter request failed",
                json.dumps(detail),
                flush=True,
            )
            time.sleep(attempt)
        except (TimeoutError, urllib.error.URLError) as exc:
            detail = {"path": path, "attempt": attempt, "error": str(exc)}
            if attempt == REQUEST_ATTEMPTS:
                raise RuntimeError(f"sorter request failed: {json.dumps(detail)}") from exc
            print("WARNING: sorter request failed", json.dumps(detail), flush=True)
            time.sleep(attempt)
    raise RuntimeError(f"sorter request exhausted without result: path={path}")


def require_sorter_capabilities(use_smart_tag: bool, use_trait_candidates: bool) -> None:
    schema = _req("/openapi.json", timeout=30)
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("sd-image-sorter OpenAPI response has no paths object")
    required = {
        "/api/scan",
        "/api/scan/progress",
        "/api/images/selection-token",
        "/api/images/selection-chunk",
        "/api/tag",
        "/api/tag/progress",
        "/api/tags/export-batch",
    }
    if use_trait_candidates:
        required.add("/api/tags/trait-candidates")
    if use_smart_tag:
        required.update({"/api/smart-tag/start", "/api/smart-tag/progress"})
    missing = sorted(required - set(paths))
    if missing:
        raise RuntimeError(
            "sd-image-sorter is incompatible; missing API paths: " + ", ".join(missing)
        )


def resolve_general_threshold(model: str, override: float | None) -> float:
    if override is not None:
        return override
    if model not in TAGGER_GENERAL_THRESHOLDS:
        supported = ", ".join(sorted(TAGGER_GENERAL_THRESHOLDS))
        raise ValueError(
            f"no calibrated threshold for model={model!r}; pass --threshold or use {supported}"
        )
    return TAGGER_GENERAL_THRESHOLDS[model]


def is_progress_complete(path: str, progress: dict) -> bool:
    status = str(progress.get("status", "")).lower()
    if status in {"error", "failed", "cancelled"}:
        message = progress.get("message") or json.dumps(progress, ensure_ascii=False)
        raise RuntimeError(f"sorter job failed: path={path} status={status} message={message}")
    if status == "idle":
        raise RuntimeError(f"sorter job returned idle before completion: path={path}")
    return status in {"done", "completed"}


def poll(path: str, tries: int) -> dict:
    last = None
    for _ in range(tries):
        p = _req(path, timeout=30)
        st = p.get("status")
        key = (st, p.get("processed"), p.get("total"))
        if key != last:
            print(
                f"    {st} {p.get('processed', '')}/{p.get('total', '')} {p.get('message', '')}",
                flush=True,
            )
            last = key
        if is_progress_complete(path, p):
            return p
        time.sleep(2)
    raise TimeoutError(f"sorter job timed out: path={path} polls={tries}")


def selection_token(folder: Path) -> str:
    r = _req("/api/images/selection-token", {"folder": str(folder)})
    token = r.get("selection_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"selection-token response has no token: response={r}")
    return token


def expand_ids(token: str) -> list[int]:
    ids: list[int] = []
    offset = 0
    while True:
        r = _req(
            f"/api/images/selection-chunk?selection_token={token}&offset={offset}&limit=2000",
            timeout=60,
        )
        chunk = r.get("image_ids")
        if not isinstance(chunk, list) or not all(
            isinstance(image_id, int) and not isinstance(image_id, bool) for image_id in chunk
        ):
            raise TypeError(f"selection chunk has invalid image_ids: response={r}")
        ids.extend(chunk)
        has_more = r.get("has_more")
        if not isinstance(has_more, bool):
            raise TypeError(f"selection chunk has invalid has_more: response={r}")
        if not has_more:
            break
        next_offset = r.get("next_offset")
        if (
            not isinstance(next_offset, int)
            or isinstance(next_offset, bool)
            or next_offset <= offset
        ):
            raise TypeError(f"selection chunk has invalid next_offset: response={r}")
        offset = next_offset
    return ids


def tag_booru(
    ids: list[int], model: str, thr: float, char_thr: float, gpu: bool, retag: bool
) -> None:
    print(f"  tag: POST /api/tag  model={model} thr={thr} char_thr={char_thr} gpu={gpu}")
    _req(
        "/api/tag",
        {
            "image_ids": ids,
            "model_name": model,
            "threshold": thr,
            "character_threshold": char_thr,
            "use_gpu": gpu,
            "retag_all": retag,
            "pre_tag_blacklist": PRE_TAG_BLACKLIST,
            "max_tags_per_image": 0,
        },
    )
    poll("/api/tag/progress", 1200)


def tag_smart(
    token: str,
    purpose: str,
    trigger: str,
    gpu: bool,
    thr: float | None,
    char_thr: float,
    consensus: bool,
    nl: bool,
    model: str,
) -> None:
    body = {
        "selection_token": token,
        "training_purpose": purpose,
        "trigger_word": trigger,
        "merge_strategy": "replace",
        "use_gpu": gpu,
        "character_threshold": char_thr,
        "enable_wd14": True,
        "enable_vlm": nl,
        "natural_language_mode": "vlm",
    }
    if thr is not None:
        body["general_threshold"] = thr
    if consensus:
        body["taggers"] = [
            {"model": "wd-swinv2-tagger-v3"},
            {"model": "wd-eva02-large-tagger-v3"},
            {"model": "pixai-tagger-v0.9"},
        ]
        body["consensus_min"] = 2
    else:
        body["tagger_model"] = model
    print(f"  smart-tag: consensus={consensus} nl(vlm)={nl}")
    r = _req("/api/smart-tag/start", body)
    job = r.get("job_id") or r.get("id")
    if not job:
        raise RuntimeError(f"smart-tag response has no job id: response={r}")
    poll(f"/api/smart-tag/progress?job_id={job}", 1200)


def parse_trait_candidate(value: object) -> TraitCandidate:
    if not isinstance(value, dict):
        raise TypeError(f"trait candidate must be an object: value={value!r}")
    tag = value.get("tag")
    family = value.get("family")
    ratio = value.get("ratio")
    count = value.get("count")
    if not isinstance(tag, str) or not tag:
        raise TypeError(f"trait candidate tag must be a non-empty string: value={value!r}")
    if not isinstance(family, str) or not family:
        raise TypeError(f"trait candidate family must be a non-empty string: value={value!r}")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise TypeError(f"trait candidate ratio must be numeric: value={value!r}")
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(f"trait candidate count must be an integer: value={value!r}")
    return {"tag": tag, "family": family, "ratio": float(ratio), "count": count}


def trait_prune(
    token: str, min_ratio: float, apply_ratio: float
) -> tuple[list[str], list[TraitCandidate]]:
    """Return frequency-selected trait tags and the complete reviewable candidate list."""
    r = _req(
        "/api/tags/trait-candidates",
        {"selection_token": token, "min_ratio": min_ratio, "limit": 100},
    )
    raw_candidates = r.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise TypeError(f"trait candidates response must contain a list: response={r}")
    cands = [parse_trait_candidate(value) for value in raw_candidates]
    if not cands:
        print("  trait-prune: no identity traits above ratio — nothing to prune.")
        return [], []
    print(
        f"  trait-prune candidates (of {r.get('total_images')} imgs) — auto-prune ratio ≥ {apply_ratio}:"
    )
    prune: list[str] = []
    for c in cands:
        mark = "PRUNE" if c["ratio"] >= apply_ratio else "keep "
        if c["ratio"] >= apply_ratio:
            prune.append(c["tag"])
        print(
            f"    [{mark}] {c['tag']:<28} {c['family']:<6} ratio={c['ratio']:.2f} ({c['count']}x)"
        )
    print(
        f"  → blacklisting {len(prune)} invariant traits (trigger will carry them). "
        f"Re-run with --trait-ratio / --no-trait-prune to adjust."
    )
    return prune, cands


def semantic_warnings(concept_dir: Path, subject_tag: str) -> list[SemanticWarning]:
    warnings: list[SemanticWarning] = []
    for caption in sorted(concept_dir.rglob("*.txt")):
        tags = profile.parse_tags_file(caption)
        mismatches = set(profile.match_keywords(tags, profile.offmodel_keywords(subject_tag)))
        mismatches.update(tags & profile.MULTI_SUBJECT)
        if mismatches:
            warnings.append({"caption": str(caption), "tags": sorted(mismatches)})
    return warnings


def missing_caption_images(concept_dir: Path) -> list[str]:
    return [
        str(image)
        for image in sorted(concept_dir.rglob("*"))
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
        and not image.with_suffix(".txt").is_file()
    ]


def export_anima(
    token: str,
    trigger: str,
    purpose: str,
    preset: str,
    prune: list[str],
    quality: str,
) -> dict:
    body = {
        "selection_token": token,
        "output_mode": "beside_image",
        "content_mode": "template",
        "template_options": {
            "preset_id": preset,
            "trigger": trigger,
            "quality_override": quality,
        },
        "training_purpose": purpose,
        "blacklist": prune,
        "dedupe_implications": True,
        "overwrite_policy": "overwrite",
    }
    print(f"  export: content_mode=template preset={preset} purpose={purpose} prune={len(prune)}")
    return _req("/api/tags/export-batch", body)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Anima-format WD14 tagging (section-ordered, trait-pruned)."
    )
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="one populated <repeats>_<concept> folder",
    )
    ap.add_argument("--trigger", required=True, help="trigger word (absorbs invariant identity)")
    ap.add_argument(
        "--style",
        action="store_true",
        help="style LoRA (training_purpose=style): the sorter strips style/medium/artist "
        "tags by CATEGORY so the trigger absorbs the look; content + character tags kept; "
        "no identity trait prune",
    )
    ap.add_argument(
        "--subject-tag",
        choices=profile.SUBJECT_TAGS,
        help="character subject count tag; required unless --style",
    )
    ap.add_argument(
        "--model",
        default="wd-eva02-large-tagger-v3",
        help="tagger: wd-eva02-large-tagger-v3 (default) | wd-swinv2-tagger-v3 (lighter)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        help="general-tag cutoff; defaults to the selected model's published P=R point",
    )
    ap.add_argument("--character-threshold", type=float, default=0.85)
    ap.add_argument(
        "--nl",
        action="store_true",
        help="add a natural-language sentence (needs a VLM in the sorter)",
    )
    ap.add_argument(
        "--consensus",
        action="store_true",
        help="require 2-of-3 agreement from swinv2, eva02, and pixai",
    )
    ap.add_argument(
        "--trait-ratio",
        type=float,
        default=0.9,
        help="auto-prune traits appearing in ≥ this fraction (default 0.9)",
    )
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=0.6,
        help="trait-candidate reporting floor (default 0.6)",
    )
    ap.add_argument(
        "--no-trait-prune", action="store_true", help="keep identity traits in captions"
    )
    ap.add_argument(
        "--quality",
        default="",
        help="shared training quality tags; empty by default because unscored images must not be relabeled",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="GPU tagging (default CPU — avoids VRAM contention)",
    )
    ap.add_argument(
        "--retag",
        action="store_true",
        help="force re-tag (else only untagged images are tagged)",
    )
    in_use_guard.add_guard_arguments(ap)
    a = ap.parse_args()
    if not a.style and not a.subject_tag:
        ap.error("--subject-tag is required for character tagging")
    if a.style and not a.trigger.startswith("@"):
        ap.error("style triggers must use Anima's @artist namespace")

    purpose = "style" if a.style else "character"
    preset = "anima" if a.nl else "anima_tags_only"  # NL slot only fills when a VLM ran

    concept_dir = Path(a.dataset_dir).resolve()
    if not concept_dir.is_dir():
        print(f"ERROR: dataset directory not found: {concept_dir}")
        return 2

    blocked = in_use_guard.guard_paths([str(concept_dir)], a.trainer_url, a.allow_in_use)
    if blocked:
        return blocked

    use_smart_tag = bool(a.consensus or a.nl)
    use_trait_candidates = purpose == "character" and not a.no_trait_prune
    require_sorter_capabilities(use_smart_tag, use_trait_candidates)
    threshold = (
        None
        if a.consensus and a.threshold is None
        else resolve_general_threshold(a.model, a.threshold)
    )

    print(f"scan   : {concept_dir}")
    # cleanup_missing drops stale DB rows from a previous build (so a rebuilt/curated
    # folder re-indexes to exactly its current files, not a stale superset).
    _req(
        "/api/scan",
        {
            "folder_path": str(concept_dir),
            "recursive": True,
            "cleanup_missing": True,
            "force_reparse": True,
        },
    )
    poll("/api/scan/progress", 1200)
    token = selection_token(concept_dir)
    ids = expand_ids(token)
    print(f"ids    : {len(ids)} images")
    if not ids:
        print("ERROR: sorter indexed 0 images for this folder.")
        return 2

    if use_smart_tag:
        tag_smart(
            token,
            purpose,
            a.trigger,
            a.gpu,
            threshold,
            a.character_threshold,
            a.consensus,
            a.nl,
            a.model,
        )
    else:
        if threshold is None:
            raise RuntimeError("single-model tagging requires a resolved threshold")
        tag_booru(ids, a.model, threshold, a.character_threshold, a.gpu, a.retag)

    # Character: prune invariant identity traits so the trigger absorbs them.
    # Style: nothing to prune here — the sorter strips style/medium/artist tags by
    # CATEGORY at export (training_purpose="style"), so the trigger carries the look.
    prune: list[str] = []
    candidates: list[TraitCandidate] = []
    if purpose == "character" and not a.no_trait_prune:
        prune, candidates = trait_prune(token, a.min_ratio, a.trait_ratio)

    res = export_anima(token, a.trigger, purpose, preset, prune, a.quality)
    if int(res.get("error_count") or 0) > 0:
        raise RuntimeError(f"caption export failed: response={res}")
    print(
        "  ->",
        json.dumps({k: res.get(k) for k in ("exported", "skipped", "error_count", "content_mode")}),
    )
    missing = missing_caption_images(concept_dir)
    if missing:
        raise RuntimeError(
            f"caption export incomplete: folder={concept_dir} missing={missing[:12]} response={res}"
        )
    n_txt = len(list(concept_dir.rglob("*.txt")))
    warnings = semantic_warnings(concept_dir, a.subject_tag) if a.subject_tag else []
    audit_path = concept_dir.parent / f"{concept_dir.name}.tag-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "dataset": str(concept_dir),
                "model": a.model,
                "general_threshold": threshold,
                "character_threshold": a.character_threshold,
                "quality_override": a.quality,
                "trait_candidates": candidates,
                "pruned_traits": prune,
                "semantic_warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"\ncaptions written: {n_txt}  (Anima order: quality, safety, count, {a.trigger}, chars, series, @artists, general)"
    )
    print(f"audit  : {audit_path}  semantic warnings={len(warnings)}")
    if warnings:
        print("WARNING: review semantic warnings before training; they do not block export.")
    print(f'next: doctor.py "{concept_dir.parent}" --trigger {a.trigger} --epochs 10 --report')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
