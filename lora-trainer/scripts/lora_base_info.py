#!/usr/bin/env python3
"""Tell which Anima base a trained LoRA belongs to: 28 blocks (2B / 1.0) or 40 (2.9B).

Anima 2.9B inserts 12 blocks into the 28-block model, so a LoRA only fits the base
it was trained on. A 28-block LoRA on a 2.9B model needs its keys remapped (for
example the ComfyUI-Anima-28to40-Lora-Stack node) or it lands on the wrong layers;
a 40-block LoRA does not load correctly on the 2B model.

The answer comes from the safetensors header alone (8-byte length + JSON, nothing
else is read): the highest ``blocks_<n>`` index in the tensor names. Any block 28..39
means 40 blocks; otherwise 28. A native 40-block LoRA that never touches blocks
28..39 looks exactly like a 28-block one, and the report says so.

The base checkpoint the run was configured with is read the same way (``net.blocks.<n>``)
and compared. Standard library only.

    python lora_base_info.py D:/out/zkz/zkz.safetensors --base-model D:/models/anima-base-v1.0.safetensors
    python lora_base_info.py --output-dir D:/out/zkz --output-name zkz --out ./insights
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Optional

MAX_HEADER_BYTES = 256 * 1024 * 1024  # a sane header is a few MB; refuse a corrupt length
BLOCKS_2B = 28
BLOCKS_2_9B = 40
BLOCK_KEY = re.compile(r"(?:^|[._])blocks[._](\d+)(?=[._])")
# Anima names its attention projections "*_proj"; Wan (also 40 blocks) uses plain ".q" / ".k".
ANIMA_KEY = re.compile(
    r"blocks[._]\d+[._](?:cross|self)_attn[._](?:[qkv]|output)_proj|adaln_modulation[._](?:cross|self)_attn"
)
NAME_2_9B = re.compile(r"2[._]?9\s*b", re.IGNORECASE)

WARN_REMAP = {
    "en": (
        "This LoRA has 28 blocks (Anima 2B / 1.0). On an Anima 2.9B model it needs a key remap, "
        "for example the ComfyUI-Anima-28to40-Lora-Stack node; loaded as is, it applies to the wrong layers."
    ),
    "zh": (
        "这个 LoRA 是 28 块（Anima 2B / 1.0）。放到 Anima 2.9B 上需要先做 key 重映射，"
        "例如 ComfyUI-Anima-28to40-Lora-Stack 节点；直接加载会作用到错误的层。"
    ),
}
WARN_NOT_2B = {
    "en": (
        "This LoRA has 40 blocks (Anima 2.9B). It does not load correctly on the 2B / 1.0 model; "
        "use it with a 2.9B base."
    ),
    "zh": "这个 LoRA 是 40 块（Anima 2.9B）。在 2B / 1.0 底模上无法正确加载，请配 2.9B 底模使用。",
}
NOTE_AMBIGUOUS = {
    "en": (
        "Only blocks 0-27 were found. A native 40-block LoRA that never touches blocks 28-39 "
        "looks the same, so this is a 28-block verdict by key layout."
    ),
    "zh": "只找到 0-27 块。一个从不碰 28-39 块的原生 40 块 LoRA 看起来完全一样，所以这是按 key 布局得出的 28 块结论。",
}


def read_safetensors_header(path: Path) -> tuple[list[str], dict[str, str]]:
    """Return ``(tensor names, __metadata__)``; raises ``ValueError`` on a broken file."""
    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("file is shorter than a safetensors header")
        (length,) = struct.unpack("<Q", raw_length)
        if length == 0 or length > MAX_HEADER_BYTES:
            raise ValueError(f"implausible safetensors header length {length}")
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("file ends inside the safetensors header")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"safetensors header is not JSON ({error})") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    metadata = header.pop("__metadata__", None)
    meta = {str(k): str(v) for k, v in metadata.items()} if isinstance(metadata, dict) else {}
    return list(header), meta


def highest_block(names: list[str]) -> Optional[int]:
    found = [int(m.group(1)) for name in names if (m := BLOCK_KEY.search(name))]
    return max(found) if found else None


def blocks_to_label(highest: Optional[int]) -> tuple[Optional[int], Optional[str]]:
    """``(block_count, model)``; model is None when the index is outside any known Anima."""
    if highest is None:
        return None, None
    if highest >= BLOCKS_2_9B:
        return None, None
    if highest >= BLOCKS_2B:
        return BLOCKS_2_9B, "anima-2.9b"
    return BLOCKS_2B, "anima-2b"


def inspect_lora(path: Path) -> dict[str, Any]:
    """Block count and base family of one LoRA file."""
    names, meta = read_safetensors_header(path)
    is_anima = any(ANIMA_KEY.search(n) for n in names)
    top = highest_block(names)
    count, model = blocks_to_label(top) if is_anima else (None, None)
    result: dict[str, Any] = {
        "file": str(path),
        "tensors": len(names),
        "is_anima": is_anima,
        "highest_block": top,
        "block_count": count,
        "base": model,
        "metadata_base_name": meta.get("ss_sd_model_name") or None,
        "metadata_architecture": meta.get("modelspec.architecture") or None,
    }
    if is_anima and top is not None and top >= BLOCKS_2_9B:
        result["note"] = {"en": f"Block index {top} is beyond 40 blocks; not a known Anima layout."}
    return result


def inspect_base_model(path: Path) -> dict[str, Any]:
    """Block count of a base checkpoint, from its header (never loads weights)."""
    names, _ = read_safetensors_header(path)
    top = highest_block(names)
    count, model = blocks_to_label(top)
    return {"file": str(path), "highest_block": top, "block_count": count, "base": model}


def _verdict_messages(lora: dict, base: Optional[dict]) -> tuple[list[dict], list[dict]]:
    """``(warnings, notes)``, each a ``{"en", "zh"}`` pair."""
    warnings: list[dict] = []
    notes: list[dict] = []
    count = lora["block_count"]
    if count == BLOCKS_2B:
        notes.append(NOTE_AMBIGUOUS)
    if base is not None and base.get("block_count") and count:
        if base["block_count"] == BLOCKS_2_9B and count == BLOCKS_2B:
            warnings.append(WARN_REMAP)
        elif base["block_count"] == BLOCKS_2B and count == BLOCKS_2_9B:
            warnings.append(WARN_NOT_2B)
    elif count == BLOCKS_2B:
        notes.append(WARN_REMAP)  # no base given: the remap fact still matters for later use
    elif count == BLOCKS_2_9B:
        notes.append(WARN_NOT_2B)
    return warnings, notes


def build_report(lora_path: Path, base_model_path: Optional[Path] = None) -> dict[str, Any]:
    """Everything the final report needs about the LoRA's base. Never raises on bad input."""
    report: dict[str, Any] = {"lora": None, "configured_base_model": None, "compatible": None}
    try:
        report["lora"] = inspect_lora(lora_path)
    except (OSError, ValueError) as error:
        report["error"] = f"could not read the LoRA header: {error}"
        return report
    base: Optional[dict] = None
    if base_model_path is not None:
        try:
            base = inspect_base_model(base_model_path)
        except (OSError, ValueError) as error:
            base = {"file": str(base_model_path), "error": str(error)}
        report["configured_base_model"] = base
        if "error" not in base and base["block_count"] is None:
            base["note"] = "no recognisable Anima blocks in this checkpoint"
        if base.get("error") is None and base.get("block_count") and report["lora"]["block_count"]:
            report["compatible"] = base["block_count"] == report["lora"]["block_count"]
    lora = report["lora"]
    if not lora["is_anima"]:
        report["notes"] = [{"en": "Not an Anima LoRA (no Anima attention keys); no block verdict."}]
        return report
    report["warnings"], report["notes"] = _verdict_messages(lora, base)
    return report


def newest_lora(output_dir: Path, output_name: str) -> Optional[Path]:
    """The final ``<name>.safetensors`` if present, else the newest epoch snapshot."""
    final = output_dir / f"{output_name}.safetensors"
    if final.is_file():
        return final
    snaps = [p for p in output_dir.glob(f"{output_name}-*.safetensors") if p.is_file()]
    return max(snaps, key=lambda p: p.stat().st_mtime) if snaps else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("lora", nargs="?", help="a trained LoRA .safetensors")
    parser.add_argument("--output-dir", help="instead of a file: the run's output_dir")
    parser.add_argument(
        "--output-name", default="", help="with --output-dir: the run's output_name"
    )
    parser.add_argument("--base-model", help="pretrained_model_name_or_path of the run")
    parser.add_argument("--out", help="also write <lora name>.base.json into this folder")
    args = parser.parse_args(argv)

    if args.lora:
        lora_path: Optional[Path] = Path(args.lora)
    elif args.output_dir and args.output_name:
        lora_path = newest_lora(Path(args.output_dir), args.output_name)
    else:
        parser.error("give a LoRA file, or --output-dir together with --output-name")
    if lora_path is None or not lora_path.is_file():
        print(f"no LoRA file found: {lora_path or args.output_dir}", file=sys.stderr)
        return 2
    report = build_report(lora_path, Path(args.base_model) if args.base_model else None)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{lora_path.stem}.base.json"
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["written"] = str(target)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
