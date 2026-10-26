"""failure_catalogue.py: rules, false-positive guards, live suppression, trainer log polling."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "failure_catalogue.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("failure_catalogue", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fc = load_script()

# One real-looking line per category (several taken from trainer logs and sd-scripts source).
POSITIVE = {
    "cuda_oom": [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 24.00 GiB",
        "RuntimeError: CUDA error: out of memory",
        "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    ],
    "nan_loss": [
        "steps:  12%|█▎        | 4/32 [00:05<00:30,  1.17s/it, avr_loss=nan]",
        "steps:  12%|█▎        | 4/32 [00:05<00:30,  1.17s/it, avr_loss=inf]",
        "RuntimeError: non-finite loss at global_step=48",
        "Loss is NaN, skipping update",
        "loss became NaN",
    ],
    "model_file_missing": [
        "FileNotFoundError: [Errno 2] No such file or directory: 'I:/models/anima-base-v1.0.safetensors'",
        "OSError: Can't load the model: no such file or directory ./sd-models/anima/qwen_image_vae.safetensors",
    ],
    "model_file_corrupt": [
        "safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge",
        "safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer",
    ],
    "captions_missing": [
        "WARNING  No caption file found for 12 images. Training will continue without captions for these images.",
    ],
    "dataset_empty": [
        "ValueError: No data found. Please verify arguments (train_data_dir must be the parent of folders with images)",
        "ignore subset with image_dir='D:/data/5_cat': no images found / 画像が見つからないためサブセットを無視します",
        "ValueError: num_samples should be a positive integer value, but got num_samples=0",
    ],
    "folder_without_repeats": [
        "WARNING  ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: cat",
    ],
    "num_repeats_below_1": [
        "ignore subset with image_dir='D:/data/0_cat': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: 0",
    ],
    "unsupported_train_type": [
        "不支持的训练类型: anima-lora-x",
        "ValueError: unsupported model_train_type: foo",
    ],
    "paging_file_small": [
        'OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading "C:\\torch\\lib\\cudnn_ops64_9.dll"',
        "Error: os error 1455",
        "页面文件太小，无法完成操作",
    ],
    "dll_load_failed": [
        "ImportError: DLL load failed while importing _C: The specified module could not be found.",
        'OSError: [WinError 126] The specified module could not be found. Error loading "fbgemm.dll" or one of its dependencies.',
    ],
    "disk_full": [
        "OSError: [Errno 28] No space left on device",
        "OSError: [WinError 112] There is not enough space on the disk",
    ],
    "port_in_use": [
        "OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted",
        "ERROR:    [Errno 10048] error while attempting to bind on address ('127.0.0.1', 28000): address already in use",
        "OSError: [Errno 98] Address already in use",
    ],
    "attention_backend_missing": [
        "ModuleNotFoundError: No module named 'xformers'",
        "WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:",
        "ModuleNotFoundError: No module named 'flash_attn'",
        "ImportError: flash-attn is not installed",
    ],
    "cuda_unavailable": [
        "AssertionError: Torch not compiled with CUDA enabled",
        "RuntimeError: No CUDA GPUs are available",
    ],
    "host_ram_oom": [
        "RuntimeError: [enforce fail at alloc_cpu.cpp:114] data. DefaultCPUAllocator: not enough memory: you tried to allocate 1073741824 bytes.",
        "MemoryError",
    ],
    "bad_image": [
        "PIL.UnidentifiedImageError: cannot identify image file 'D:/data/5_cat/001.png'",
        "OSError: image file is truncated (12 bytes not processed)",
    ],
}

# Lines a healthy run prints: CLI flags, config echoes and chatter that contains the same words.
BENIGN = [
    "22  01:55:51-253282 INFO     Anima attn_mode auto-detected: xformers",
    "                             with attn_mode=xformers,",
    "INFO     Loading Anima DiT model with attn_mode=xformers, split_attn: False...",
    "--xformers --mem_eff_attn --gradient_checkpointing --nan_guard --masked_loss",
    "accelerate launch --num_cpu_threads_per_process=4 train.py --nan_guard --nan_guard_recover",
    "nan_guard = true",
    "nan_guard_recover = true",
    "xformers = true",
    "    num_repeats: 1",
    "  num train images * repeats / 学習画像の数×繰り返し回数: 8",
    "INFO     move vae to cpu to save  anima_train_network.py:314",
    "INFO     text_encoder is not needed    train_network.py:1415 for training. deleting to save memory.",
    "blocks_to_swap = 16",
    "gradient_checkpointing = true",
    "steps: 100%|██████████| 32/32 [00:37<00:00,  1.17s/it, avr_loss=0.0011]",
    "steps:  12%|█▎        | 4/32 [00:05<00:30,  1.17s/it, avr_loss=0.00121]",
    "INFO     DiT config: num_blocks=28 (inferred from D:/models/anima.safetensors)",
    "saving checkpoint: D:/out/zkz/zkz.safetensors",
    "INFO     8 train images with repeats.     train_util.py:2307",
    "INFO     0 reg images with repeats.       train_util.py:2311",
    "WARNING  no regularization images /       train_util.py:2316",
    "To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.",
    "UserWarning: pkg_resources is deprecated as an API.",
    'INFO:     127.0.0.1:50190 - "GET /api/train/log/tail/abc?limit=12 HTTP/1.1" 200 OK',
    "INFO: Uvicorn running on http://127.0.0.1:28000",
    'loss_type = "l2"',
    "unet_lr = 5e-05",
    "network_alpha = 16",
    "inference steps: 8, information about infinity",
    "prompt: a red circle on a plain white background, flat colors",
    "port = 28000",
    "Training finished / 训练完成",
]


def categories(lines: list[str]) -> set[str]:
    return {f.category for f in fc.scan(lines)}


class RuleTests(unittest.TestCase):
    def test_every_rule_has_samples_and_a_complete_message(self) -> None:
        rule_names = {r.category for r in fc.RULES}
        self.assertEqual(rule_names, set(POSITIVE), "RULES and POSITIVE samples differ")
        for rule in fc.RULES:
            with self.subTest(rule=rule.category):
                self.assertIn(rule.severity, ("error", "warn"))
                for text in (rule.message_en, rule.message_zh, rule.fix_en, rule.fix_zh):
                    self.assertTrue(text.strip())
                    self.assertNotIn("\n", text)
                self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in rule.message_zh))
                self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in rule.fix_zh))

    def test_positive_samples_match_their_own_category(self) -> None:
        for category, samples in POSITIVE.items():
            for sample in samples:
                with self.subTest(category=category, sample=sample[:60]):
                    self.assertIn(category, categories([sample]))

    def test_benign_lines_match_nothing(self) -> None:
        for line in BENIGN:
            with self.subTest(line=line[:70]):
                self.assertEqual(categories([line]), set())

    def test_whole_benign_block_matches_nothing(self) -> None:
        self.assertEqual(categories(BENIGN), set())

    def test_real_trainer_logs_have_no_findings(self) -> None:
        # A server.log from a finished tiny training; skipped when not present.
        import os

        path = os.environ.get("FAILURE_CATALOGUE_REAL_LOG")
        if not path or not Path(path).is_file():
            self.skipTest("set FAILURE_CATALOGUE_REAL_LOG to a healthy trainer log")
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        self.assertEqual({f.category for f in fc.scan(text.splitlines())}, set())

    def test_message_wrapped_over_lines_still_matches(self) -> None:
        wrapped = [
            "RuntimeError: CUDA out of",
            "memory. Tried to allocate 20.00 MiB",
        ]
        self.assertIn("cuda_oom", categories(wrapped))
        rich = [
            "INFO     ignore subset with image_dir='D:/d/5_cat': no   train_util.py:2093",
            "         images found / 画像が見つからないため",
        ]
        self.assertIn("dataset_empty", categories(rich))

    def test_evidence_line_is_where_the_match_starts(self) -> None:
        lines = [
            "INFO     mean ar error (without repeats): train_util.py:1144",
            "                             0",
            "ERROR    No data found. Please verify   train_network.py:605",
            "                             arguments (train_data_dir must",
        ]
        finding = fc.scan(lines)[0]
        self.assertEqual(finding.category, "dataset_empty")
        self.assertTrue(finding.line.startswith("ERROR    No data found"))

    def test_message_wrapped_over_many_narrow_lines_still_matches(self) -> None:
        lines = [
            "WARNING  ignore subset with image_dir=",
            "         'D:/very/long/path/to/the/dataset/",
            "         0_cat': no images found / 画像が",
            "         見つからないため",
        ]
        self.assertIn("dataset_empty", categories(lines))

    def test_paging_file_error_is_not_also_called_a_dll_problem(self) -> None:
        line = POSITIVE["paging_file_small"][0]
        self.assertEqual(categories([line]), {"paging_file_small"})
        self.assertEqual(
            categories(["ImportError: DLL load failed while importing _C"]), {"dll_load_failed"}
        )

    def test_progress_bar_carriage_returns_are_separate_lines(self) -> None:
        chunk = "steps:  5%|5| 1/32 [00:01, avr_loss=0.5]\rsteps: 6%|6| 2/32 [00:02, avr_loss=nan]"
        self.assertIn("nan_loss", categories([chunk]))

    def test_scan_returns_one_finding_per_category_with_a_count(self) -> None:
        lines = ["avr_loss=nan"] * 3 + ["noise"] + ["CUDA out of memory"]
        found = {f.category: f for f in fc.scan(lines)}
        self.assertEqual(found["nan_loss"].count, 3)
        self.assertEqual(found["cuda_oom"].count, 1)
        self.assertEqual(list(f.severity for f in fc.scan(lines)), ["error", "error"])

    def test_errors_sort_before_warnings(self) -> None:
        lines = ["ModuleNotFoundError: No module named 'xformers'", "CUDA out of memory"]
        self.assertEqual([f.severity for f in fc.scan(lines)], ["error", "warn"])

    def test_finding_serialises_to_json_with_both_languages(self) -> None:
        item = fc.scan(["CUDA out of memory"])[0].as_dict()
        json.dumps(item, ensure_ascii=False)
        for key in ("category", "severity", "message_en", "message_zh", "fix_en", "fix_zh", "line"):
            self.assertIn(key, item)
        self.assertIn("gradient_checkpointing", item["fix_en"])
        self.assertIn("blocks_to_swap", item["fix_en"])


class LiveMonitorTests(unittest.TestCase):
    def test_same_category_is_suppressed_for_sixty_seconds(self) -> None:
        monitor = fc.LiveMonitor()
        first = monitor.feed(["avr_loss=nan"], now=1000.0)
        again = monitor.feed(["avr_loss=nan"], now=1030.0)
        later = monitor.feed(["avr_loss=nan"], now=1060.5)
        self.assertEqual([f.category for f in first], ["nan_loss"])
        self.assertEqual(again, [])
        self.assertEqual([f.category for f in later], ["nan_loss"])
        self.assertEqual(later[0].suppressed, 1)  # one earlier repeat was held back

    def test_other_categories_are_not_held_back(self) -> None:
        monitor = fc.LiveMonitor()
        monitor.feed(["avr_loss=nan"], now=0.0)
        out = monitor.feed(["CUDA out of memory", "avr_loss=nan"], now=5.0)
        self.assertEqual([f.category for f in out], ["cuda_oom"])

    def test_state_survives_between_polls(self) -> None:
        monitor = fc.LiveMonitor()
        monitor.feed(["avr_loss=nan"], now=100.0)
        restored = fc.LiveMonitor.from_state(json.loads(json.dumps(monitor.state())))
        self.assertEqual(restored.feed(["avr_loss=nan"], now=110.0), [])
        self.assertEqual(len(restored.feed(["avr_loss=nan"], now=161.0)), 1)

    def test_empty_and_noise_batches_do_nothing(self) -> None:
        monitor = fc.LiveMonitor()
        self.assertEqual(monitor.feed([], now=0.0), [])
        self.assertEqual(monitor.feed(BENIGN, now=1.0), [])


class FakeTrainer(BaseHTTPRequestHandler):
    lines: list[str] = []
    done = False
    mode = "ok"

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.mode == "old" or not self.path.startswith("/api/train/log/tail/"):
            status, body = 404, {"detail": "Not Found"}
        elif self.path.split("?")[0].endswith("/gone"):
            status, body = 404, {"detail": "Unknown task_id"}
        else:
            limit = int(self.path.split("limit=")[1]) if "limit=" in self.path else 240
            status, body = (
                200,
                {
                    "status": "success",
                    "data": {
                        "lines": self.lines[-limit:],
                        "total": len(self.lines),
                        "done": self.done,
                    },
                },
            )
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class TrainerPollTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeTrainer.lines, FakeTrainer.done, FakeTrainer.mode = [], False, "ok"
        self.server = HTTPServer(("127.0.0.1", 0), FakeTrainer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "watch.json"

    def test_poll_reads_only_new_lines_and_suppresses_repeats(self) -> None:
        FakeTrainer.lines = ["boot", "avr_loss=0.1"]
        first = fc.poll_trainer(self.url, "t1", self.state, now=1000.0)
        self.assertEqual((first["new_lines"], first["findings"]), (2, []))
        FakeTrainer.lines += ["avr_loss=nan", "avr_loss=nan"]
        second = fc.poll_trainer(self.url, "t1", self.state, now=1010.0)
        self.assertEqual(second["new_lines"], 2)
        self.assertEqual([f["category"] for f in second["findings"]], ["nan_loss"])
        FakeTrainer.lines += ["avr_loss=nan"]
        third = fc.poll_trainer(self.url, "t1", self.state, now=1020.0)
        self.assertEqual((third["new_lines"], third["findings"]), (1, []))
        self.assertEqual(third["suppressed"], {"nan_loss": 1})

    def test_done_run_gets_a_final_scan_without_suppression(self) -> None:
        FakeTrainer.lines = ["avr_loss=nan"] * 2 + ["OSError: [Errno 28] No space left on device"]
        FakeTrainer.done = True
        result = fc.poll_trainer(self.url, "t1", self.state, now=0.0)
        self.assertTrue(result["done"])
        self.assertEqual(
            {f["category"] for f in result["final_findings"]}, {"nan_loss", "disk_full"}
        )

    def test_unknown_task_and_old_trainer_say_so_instead_of_failing(self) -> None:
        gone = fc.poll_trainer(self.url, "gone", self.state, now=0.0)
        self.assertEqual(gone["error"], "task_unknown_to_trainer")
        FakeTrainer.mode = "old"
        old = fc.poll_trainer(self.url, "t1", self.state, now=0.0)
        self.assertEqual(old["error"], "trainer_has_no_log_tail_api")

    def test_unreachable_trainer_is_reported(self) -> None:
        result = fc.poll_trainer("http://127.0.0.1:1", "t1", self.state, now=0.0)
        self.assertTrue(result["error"].startswith("trainer_unreachable"))

    def test_lines_lost_between_polls_are_called_out(self) -> None:
        FakeTrainer.lines = ["a"]
        fc.poll_trainer(self.url, "t1", self.state, now=0.0, limit=5)
        FakeTrainer.lines = ["a"] + [f"line {i}" for i in range(20)]
        result = fc.poll_trainer(self.url, "t1", self.state, now=1.0, limit=5)
        self.assertEqual(result["new_lines"], 5)
        self.assertEqual(result["missed_lines"], 15)

    def test_cli_scans_a_log_file(self) -> None:
        log = Path(self._tmp.name) / "train.log"
        log.write_text("fine\nOSError: [WinError 1455] paging file too small\n", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fc.main(["--file", str(log), "--tail", "50"])
        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual([f["category"] for f in payload["findings"]], ["paging_file_small"])


if __name__ == "__main__":
    unittest.main()
