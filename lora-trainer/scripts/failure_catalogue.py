#!/usr/bin/env python3
"""Known training failures: what a log line means and what to do about it.

A catalogue of rules (category, regex, severity, one-line message in English and
Chinese, concrete fix) plus the two ways the skill uses it:

* live: poll the trainer's ``GET /api/train/log/tail/{task_id}`` while a run is going.
  Only lines that are new since the last poll are read, and the same category is
  reported at most once per 60 seconds (``LiveMonitor``).
* after exit: scan the log tail once (``scan``), one finding per category.

The regexes match what the trainer *says* (tracebacks, warnings, the progress bar's
``avr_loss=nan``), never CLI flag names or config echoes such as ``--nan_guard``,
``xformers = true`` or ``num_repeats: 1``. Lines are split on carriage returns and
joined across a six-line window with trailing ``file.py:123`` columns removed, so a
message the console wrapped over two lines still matches.

Standard library only.

    python failure_catalogue.py --task-id <uuid> --state ./insights/log_watch.json   # live poll
    python failure_catalogue.py --file D:/trainer/logs/train.log --tail 400          # after exit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:28000"
REPEAT_SUPPRESS_SECONDS = 60.0
TAIL_LIMIT = 2000  # the trainer caps /api/train/log/tail at 2000 lines
REQUEST_TIMEOUT_SECONDS = 15
WRAP_WINDOW = 6  # lines joined to catch a message the console wrapped (it can wrap a lot)
SEVERITY_ORDER = {"error": 0, "warn": 1}
# Windows words error 1455 as "Error loading x.dll", so the DLL rule also fires on that same
# line; the more specific category wins when both matched the same line.
SUPERSEDES = {"paging_file_small": "dll_load_failed"}

_FLAGS = re.IGNORECASE


class Rule(NamedTuple):
    category: str
    pattern: re.Pattern[str]
    severity: str  # "error" explains a failed run, "warn" is worth knowing but not fatal
    message_en: str
    message_zh: str
    fix_en: str
    fix_zh: str


def _rule(category: str, regex: str, severity: str, en: str, zh: str, fix_en: str, fix_zh: str):
    return Rule(category, re.compile(regex, _FLAGS), severity, en, zh, fix_en, fix_zh)


MODEL_EXT = r"\.(?:safetensors|ckpt|pt|pth|bin|gguf)\b"

RULES: tuple[Rule, ...] = (
    _rule(
        "cuda_oom",
        r"CUDA out of memory|\bOutOfMemoryError\b|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED",
        "error",
        "The GPU ran out of memory.",
        "显卡显存不足（CUDA OOM）。",
        "Lower train_batch_size, turn on gradient_checkpointing, raise blocks_to_swap (Anima) "
        "or lower resolution; close other programs that use the GPU.",
        "调低 train_batch_size，开启 gradient_checkpointing，提高 blocks_to_swap（Anima），"
        "或降低 resolution；关闭其它占用显卡的程序。",
    ),
    _rule(
        "nan_loss",
        r"\bavr_loss=(?:nan|-?inf)\b|\bnon-finite\s+loss\b|(?<![\w'\"-])loss\s+(?:is|became|was)\s+(?:nan|inf)\b"
        r"|(?<![\w'\"-])nan\s+loss\b(?!_)",
        "error",
        "The loss became NaN or infinite; training is not learning.",
        "loss 变成了 NaN 或无穷大，训练已失效。",
        "Halve the learning rate (unet_lr); with fp16 switch mixed_precision to bf16; "
        "check the dataset for broken or fully black/white images.",
        "把学习率（unet_lr）减半；用 fp16 的话把 mixed_precision 改成 bf16；"
        "检查数据集里有没有损坏或全黑/全白的图。",
    ),
    _rule(
        "model_file_missing",
        rf"(?:FileNotFoundError|no such file or directory).{{0,200}}{MODEL_EXT}"
        r"|model is not found as a file|(?<![\w-])model not found(?![\w-])",
        "error",
        "A model file was not found.",
        "找不到模型文件。",
        "Open the config and check pretrained_model_name_or_path, vae and qwen3: each must be an "
        "existing file path (no moved or renamed files, no trailing spaces).",
        "检查配置里的 pretrained_model_name_or_path、vae、qwen3：每一项都必须是真实存在的文件路径"
        "（文件有没有被移动或改名、路径末尾有没有空格）。",
    ),
    _rule(
        "model_file_corrupt",
        r"SafetensorError.{0,80}(?:deserializing header|HeaderTooLarge|MetadataIncompleteBuffer|InvalidHeader)"
        r"|Error while deserializing header",
        "error",
        "A model file is damaged or was not downloaded completely.",
        "有模型文件损坏，或没有下载完整。",
        "Download that model file again and compare its size with the source; do not reuse a "
        "file whose download was interrupted.",
        "重新下载该模型文件并和来源对比大小；不要继续使用下载中断的文件。",
    ),
    _rule(
        "captions_missing",
        r"no caption file(?:s)? found|caption file is empty|missing caption file",
        "warn",
        "Some images have no caption file and will train without captions.",
        "有图片没有标注文件，会在没有标注的情况下训练。",
        "Run the tagger on the dataset (or lora-pipeline/scripts/tag_dataset.py) so every image has "
        "a .txt next to it, then check with dataset-doctor.",
        "先给数据集打标（或运行 lora-pipeline/scripts/tag_dataset.py），保证每张图旁边有 .txt，再用 dataset-doctor 复查。",
    ),
    _rule(
        "dataset_empty",
        r"No data found\. Please verify|num_samples=0\b|ignore subset with image_dir=.{0,200}no images found"
        r"|\bfound 0 (?:train(?:ing)? )?images?\b|训练数据集路径不存在或没有图片|Data dir .{1,300} (?:not exists|is empty)",
        "error",
        "The trainer found 0 images in the training folder.",
        "训练器在训练目录里找到 0 张图片。",
        "train_data_dir must be the PARENT of folders named <repeats>_<concept> (for example "
        "5_mychar), and those folders must hold the images. Run dataset-doctor to see what is wrong.",
        "train_data_dir 必须是形如 <重复次数>_<概念名>（如 5_mychar）这些文件夹的上一级，图片要放在这些文件夹里。"
        "运行 dataset-doctor 查看问题。",
    ),
    _rule(
        "folder_without_repeats",
        r"ignore directory without repeats",
        "error",
        "A dataset folder is not named <repeats>_<name>, so it is skipped.",
        "有数据集文件夹没有按 <重复次数>_<名字> 命名，被跳过了。",
        "Rename the folder to start with the repeat count, for example 5_mychar "
        "(fix_dataset.py organize does it).",
        "把文件夹改名成以重复次数开头，例如 5_mychar（fix_dataset.py organize 可以代劳）。",
    ),
    _rule(
        "num_repeats_below_1",
        r"num_repeats is less than 1",
        "error",
        "A dataset folder has a repeat count below 1, so it is skipped.",
        "有数据集文件夹的重复次数小于 1，被跳过了。",
        "Rename the folder with a repeat count of 1 or more (for example 0_cat -> 5_cat).",
        "把文件夹改名，重复次数至少为 1（例如 0_cat 改成 5_cat）。",
    ),
    _rule(
        "unsupported_train_type",
        r"不支持的训练类型|unsupported\s+(?:model_)?train(?:ing)?[_ ]type|unknown\s+(?:model_)?train(?:ing)?[_ ]type",
        "error",
        "This trainer does not support the chosen training type.",
        "这个训练器不支持所选的训练类型。",
        "Use one of the model_train_type values in references/trainer-api.md "
        "(anima-lora, sdxl-lora, sd-lora, flux-lora, ...); GET /api/engines shows what this "
        "trainer can run.",
        "改用 references/trainer-api.md 里列出的 model_train_type（anima-lora、sdxl-lora、sd-lora、flux-lora 等）；"
        "GET /api/engines 可查看这个训练器能跑什么。",
    ),
    _rule(
        "paging_file_small",
        r"os error 1455|WinError 1455|paging file is too small|页面文件太小|页面文件.{0,10}不足",
        "error",
        "Windows ran out of virtual memory (paging file too small).",
        "Windows 虚拟内存（页面文件）不足。",
        "Make the Windows paging file larger (System Properties > Advanced > Performance > Virtual "
        "memory, or system managed on a drive with 40+ GB free), set max_data_loader_n_workers=0, "
        "close other programs, then restart the training.",
        "加大 Windows 页面文件（系统属性 > 高级 > 性能 > 虚拟内存，或在剩余 40GB 以上的盘设为系统管理），"
        "把 max_data_loader_n_workers 设为 0，关闭其它程序后重新开训。",
    ),
    _rule(
        "dll_load_failed",
        r"DLL load failed|WinError 12[67]\b|Error loading \"[^\"]+\.dll\" or one of its dependencies",
        "error",
        "A program library (DLL) failed to load.",
        "有程序库（DLL）加载失败。",
        "Install the latest Microsoft Visual C++ Redistributable (x64), then reinstall torch for "
        "your CUDA version inside the trainer's own Python; do not mix it with another Python.",
        "安装最新的 Microsoft Visual C++ 运行库（x64），再在训练器自带的 Python 里按你的 CUDA 版本重装 torch；"
        "不要和别的 Python 混用。",
    ),
    _rule(
        "disk_full",
        r"No space left on device|Errno 28\b|WinError 112\b|not enough space on the disk|磁盘空间不足",
        "error",
        "The disk is full.",
        "磁盘空间已满。",
        "Free space on the drive that holds output_dir and the cache (a LoRA snapshot is 100 MB-1 GB "
        "each), or point output_dir at another drive, then resume.",
        "清理 output_dir 和缓存所在的盘（每个 LoRA 快照 100MB-1GB），或把 output_dir 改到别的盘，再重新开训。",
    ),
    _rule(
        "port_in_use",
        r"address already in use|Errno 98\b|WinError 10048\b|Only one usage of each socket address",
        "error",
        "A network port is already taken by another program.",
        "网络端口已被别的程序占用。",
        "Close the other copy of the trainer (or TensorBoard) that holds the port, or start with a "
        "different port.",
        "关闭占着这个端口的另一个训练器（或 TensorBoard），或换一个端口启动。",
    ),
    _rule(
        "attention_backend_missing",
        r"No module named ['\"](?:xformers|flash_attn)['\"]|xformers can't load C\+\+/CUDA extensions"
        r"|\bflash[-_ ]?attn(?:ention)?\b.{0,40}(?:is |are )?not (?:installed|available|supported)"
        r"|xformers.{0,40}(?:is |are )?not (?:installed|available)",
        "warn",
        "xformers or flash-attn is not available; the trainer needs another attention backend.",
        "xformers 或 flash-attn 不可用，需要换一种注意力后端。",
        "Leave attn_mode empty (auto) or set it to sdpa; install a matching xformers wheel only if "
        "you really need it.",
        "把 attn_mode 留空（自动）或设为 sdpa；确实需要时再安装版本匹配的 xformers。",
    ),
    _rule(
        "cuda_unavailable",
        r"Torch not compiled with CUDA enabled|No CUDA GPUs are available|CUDA is not available"
        r"|no CUDA-capable device",
        "error",
        "PyTorch cannot see a CUDA GPU.",
        "PyTorch 找不到可用的 CUDA 显卡。",
        "Check the NVIDIA driver (nvidia-smi), and that the trainer's torch is a CUDA build; "
        "reinstall torch with the CUDA wheel if it says CPU.",
        "检查 NVIDIA 驱动（nvidia-smi），并确认训练器里的 torch 是 CUDA 版；如果是 CPU 版，重装 CUDA 版 torch。",
    ),
    _rule(
        "host_ram_oom",
        r"DefaultCPUAllocator.{0,60}not enough memory|can't allocate memory|\bMemoryError\b",
        "error",
        "The computer ran out of system memory (RAM).",
        "电脑系统内存（RAM）不足。",
        "Close other programs, set max_data_loader_n_workers=0, train fewer images per run, "
        "or add RAM / enlarge the Windows paging file.",
        "关闭其它程序，把 max_data_loader_n_workers 设为 0，减少单次训练的图片数，或增加内存 / 页面文件。",
    ),
    _rule(
        "bad_image",
        r"UnidentifiedImageError|cannot identify image file|image file is truncated|broken data stream",
        "error",
        "An image file is damaged and cannot be read.",
        "有图片文件损坏，无法读取。",
        "Run dataset-doctor to list unreadable images and move them out of the dataset "
        "(fix_dataset.py quarantines them, nothing is deleted).",
        "运行 dataset-doctor 找出读不了的图，并把它们移出数据集（fix_dataset.py 会隔离，不会删除）。",
    ),
)


class Finding:
    """A rule that matched, with the line that showed it."""

    def __init__(self, rule: Rule, line: str) -> None:
        self.category = rule.category
        self.severity = rule.severity
        self.message_en = rule.message_en
        self.message_zh = rule.message_zh
        self.fix_en = rule.fix_en
        self.fix_zh = rule.fix_zh
        self.line = line
        self.count = 1
        self.suppressed = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "message_en": self.message_en,
            "message_zh": self.message_zh,
            "fix_en": self.fix_en,
            "fix_zh": self.fix_zh,
            "line": self.line,
            "count": self.count,
            "suppressed": self.suppressed,
        }


_FILE_COLUMN = re.compile(r"\s+\S+\.py:\d+\s*$")


def split_lines(lines: Iterable[str]) -> list[str]:
    """Break trainer chunks into single lines (tqdm uses carriage returns)."""
    out: list[str] = []
    for chunk in lines:
        out.extend(part for part in re.split(r"[\r\n]+", str(chunk)) if part.strip())
    return out


def _clean(line: str) -> str:
    return _FILE_COLUMN.sub("", " ".join(line.split()))


def _windows(lines: list[str]) -> Iterable[tuple[int, list[str]]]:
    """``(first line index, cleaned lines)`` for each single line and each wrap window."""
    cleaned = [_clean(line) for line in lines]
    for i in range(len(lines)):
        yield i, cleaned[i : i + 1]
        if i + 1 < len(lines):
            yield i, cleaned[i : i + WRAP_WINDOW]


def _line_of_match(start: int, parts: list[str]) -> int:
    """Index (within ``parts``) of the line a match starting at ``start`` begins on."""
    offset = 0
    for index, part in enumerate(parts):
        offset += len(part) + 1  # the joining space
        if start < offset:
            return index
    return len(parts) - 1


def scan(lines: Iterable[str]) -> list[Finding]:
    """One finding per category: the first matching line, plus how many lines matched."""
    flat = split_lines(lines)
    found: dict[str, Finding] = {}
    for first, parts in _windows(flat):
        text = " ".join(parts)
        for rule in RULES:
            match = rule.pattern.search(text) if rule.category not in found else None
            if match:
                shown = flat[first + _line_of_match(match.start(), parts)].strip()
                found[rule.category] = Finding(rule, shown[:300])
    for winner, loser in SUPERSEDES.items():
        if winner in found and loser in found and found[winner].line == found[loser].line:
            del found[loser]
    for category, finding in found.items():
        rule = next(r for r in RULES if r.category == category)
        single = sum(1 for line in flat if rule.pattern.search(_clean(line)))
        finding.count = max(1, single)  # a message wrapped over lines matches no single line
    return sorted(found.values(), key=lambda f: (SEVERITY_ORDER[f.severity], f.category))


class LiveMonitor:
    """Reports a category at most once per ``REPEAT_SUPPRESS_SECONDS``."""

    def __init__(self, last_seen: Optional[dict[str, float]] = None, held: Optional[dict] = None):
        self._last_emit: dict[str, float] = dict(last_seen or {})
        self._held: dict[str, int] = dict(held or {})

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LiveMonitor":
        return cls(state.get("last_emit"), state.get("held"))

    def state(self) -> dict[str, Any]:
        return {"last_emit": self._last_emit, "held": self._held}

    def feed(self, lines: Iterable[str], now: Optional[float] = None) -> list[Finding]:
        now = time.time() if now is None else now
        out: list[Finding] = []
        for finding in scan(lines):
            last = self._last_emit.get(finding.category)
            if last is not None and now - last < REPEAT_SUPPRESS_SECONDS:
                self._held[finding.category] = self._held.get(finding.category, 0) + 1
                continue
            finding.suppressed = self._held.pop(finding.category, 0)
            self._last_emit[finding.category] = now
            out.append(finding)
        return out


# --------------------------------------------------------------------------
# Trainer API
# --------------------------------------------------------------------------


def _get_json(url: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, None


def _load_state(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}  # a damaged state file only means we re-read the window
    return data if isinstance(data, dict) else {}


def poll_trainer(
    base_url: str,
    task_id: str,
    state_path: Optional[Path] = None,
    now: Optional[float] = None,
    limit: int = TAIL_LIMIT,
) -> dict[str, Any]:
    """Read the new log lines of one task and report the findings not shown in the last 60 s."""
    now = time.time() if now is None else now
    url = f"{base_url.rstrip('/')}/api/train/log/tail/{urllib.parse.quote(task_id, safe='')}?limit={limit}"
    try:
        status, payload = _get_json(url)
    except OSError as error:
        return {"task_id": task_id, "error": f"trainer_unreachable: {error}", "findings": []}
    data = payload.get("data") if isinstance(payload, dict) else None
    if status != 200 or not isinstance(data, dict):
        detail = str(payload.get("detail", "")) if isinstance(payload, dict) else ""
        if status == 404 and detail.startswith("Unknown task_id"):
            reason = "task_unknown_to_trainer"  # trainer restarted, or the task was pruned
        elif status == 404:
            reason = "trainer_has_no_log_tail_api"
        else:
            reason = f"unexpected_response_http_{status}"
        return {"task_id": task_id, "error": reason, "findings": []}

    lines, total, done = (
        data.get("lines") or [],
        int(data.get("total") or 0),
        bool(data.get("done")),
    )
    state = _load_state(state_path)
    previous = int(state.get("total", 0)) if state.get("task_id") == task_id else 0
    new_count = max(0, total - previous)
    fresh = lines[-new_count:] if new_count else []
    missed = max(0, new_count - len(fresh))
    monitor = (
        LiveMonitor.from_state(state.get("monitor", {}))
        if state.get("task_id") == task_id
        else LiveMonitor()
    )
    findings = monitor.feed(fresh, now=now)
    result: dict[str, Any] = {
        "task_id": task_id,
        "total_lines": total,
        "new_lines": len(fresh),
        "done": done,
        "findings": [f.as_dict() for f in findings],
        "suppressed": dict(monitor.state()["held"]),
    }
    if missed:
        result["missed_lines"] = missed
    if done:
        result["final_findings"] = [f.as_dict() for f in scan(lines)]
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"task_id": task_id, "total": total, "monitor": monitor.state()}),
            encoding="utf-8",
        )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--task-id", help="poll the trainer's log tail for this task")
    parser.add_argument("--state", help="remember what was read and reported between polls")
    parser.add_argument("--file", help="scan the tail of a log file instead (after exit)")
    parser.add_argument("--tail", type=int, default=400, help="with --file: last N lines")
    parser.add_argument("--text", help="scan one message, e.g. the failure text of POST /api/run")
    args = parser.parse_args(argv)

    if args.task_id:
        result = poll_trainer(args.base_url, args.task_id, Path(args.state) if args.state else None)
    elif args.file:
        try:
            raw = Path(args.file).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            print(json.dumps({"error": f"cannot read log: {error}"}))
            return 2
        result = {"file": args.file, "findings": [f.as_dict() for f in scan(raw[-args.tail :])]}
    elif args.text is not None:
        result = {"findings": [f.as_dict() for f in scan([args.text])]}
    else:
        parser.error("give --task-id, --file or --text")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
