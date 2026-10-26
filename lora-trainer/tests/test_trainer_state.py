"""trainer_state.py: engine report, queue state and a queue-aware submit, with old-trainer fallbacks."""

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

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "trainer_state.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trainer_state", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = load_script()

SYSMEM_ISSUE = {
    "code": "windows_sysmem_fallback",
    "severity": "warning",
    "reason_zh": "显存满了以后，Windows 可能溢出到共享内存",
    "reason_en": "Windows can spill CUDA memory into shared system memory",
    "hint_zh": "NVIDIA 控制面板 > CUDA - 系统内存回退策略",
    "hint_en": "NVIDIA Control Panel > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback",
}
ERROR_ISSUE = {
    "code": "cuda_unavailable",
    "severity": "error",
    "reason_zh": "PyTorch 看不到显卡",
    "reason_en": "PyTorch cannot see a GPU",
    "hint_zh": "检查驱动",
    "hint_en": "Check the driver",
}


class Fake(BaseHTTPRequestHandler):
    """Class-level knobs; ``log`` records "METHOD path" for every request."""

    mode = "new"
    log: list[str] = []
    engines_state = "ready"
    diagnose_issues: list[dict] = []
    config_issues: list[dict] = []
    diagnose_ok = True
    queue: dict = {}
    run_response: dict = {}

    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _route(self, method: str) -> None:
        path = self.path.split("?")[0]
        Fake.log.append(f"{method} {path}")
        body = self._body() if method == "POST" else {}
        if path == "/api/run":
            self._send(200, Fake.run_response)
            return
        if Fake.mode == "old" or path not in (
            "/api/engines",
            "/api/engines/diagnose",
            "/api/queue",
            "/api/queue/start",
        ):
            self._send(404, {"detail": "Not Found"})
            return
        if path == "/api/engines":
            row = {
                "engine": "anima",
                "state": Fake.engines_state,
                "issues": [],
                "reason_zh": "" if Fake.engines_state == "ready" else "缺少依赖",
                "reason_en": "" if Fake.engines_state == "ready" else "A package is missing",
                "hint_zh": "" if Fake.engines_state == "ready" else "pip install x",
                "hint_en": "" if Fake.engines_state == "ready" else "pip install x",
            }
            self._send(
                200,
                {
                    "status": "success",
                    "data": {"engines": [], "by_train_type": {"anima-lora": row}},
                },
            )
        elif path == "/api/engines/diagnose":
            engine = {
                "id": "anima",
                "state": Fake.engines_state,
                "issues": Fake.diagnose_issues,
                "reason_zh": "",
                "reason_en": "",
                "hint_zh": "",
                "hint_en": "",
                "actions": [{"id": "reinstall", "label_en": "Reinstall"}],
            }
            data = {
                "ok": Fake.diagnose_ok,
                "train_type": body.get("train_type"),
                "engines": [engine],
                "config_issues": Fake.config_issues,
            }
            self._send(200, {"status": "success", "data": data})
        elif path == "/api/queue":
            self._send(200, {"status": "success", "data": Fake.queue})
        else:
            self._send(200, {"status": "success", "message": "queue started", "data": None})

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")


def queue_state(entries=None, restored=None, running=None, active=True) -> dict:
    return {
        "active": active,
        "user_paused": not active,
        "busy": bool(running),
        "halt_reason": "",
        "restored": restored,
        "running_entry_id": running["id"] if running else None,
        "running_task_id": running.get("task_id") if running else None,
        "entries": entries or [],
    }


class Case(unittest.TestCase):
    def setUp(self) -> None:
        Fake.mode, Fake.log = "new", []
        Fake.engines_state, Fake.diagnose_ok = "ready", True
        Fake.diagnose_issues, Fake.config_issues = [], []
        Fake.queue, Fake.run_response = queue_state(), {}
        self.server = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class PreflightTests(Case):
    def test_ready_engine_with_a_standing_warning(self) -> None:
        Fake.diagnose_issues = [SYSMEM_ISSUE]
        result = ts.preflight(self.url, "anima-lora", {"output_name": "x"})
        self.assertEqual(result["verdict"], "warnings")
        self.assertEqual(result["engine"]["state"], "ready")
        item = result["issues"][0]
        self.assertEqual(item["code"], "windows_sysmem_fallback")
        self.assertIn("Prefer No Sysmem Fallback", item["hint_en"])
        self.assertIn("共享内存", item["reason_zh"])

    def test_clean_report_is_ok(self) -> None:
        self.assertEqual(ts.preflight(self.url, "anima-lora", {})["verdict"], "ok")

    def test_errors_and_config_problems_are_problems_with_both_languages(self) -> None:
        Fake.diagnose_ok = False
        Fake.diagnose_issues = [ERROR_ISSUE]
        Fake.config_issues = [
            {
                "code": "model_file_missing",
                "severity": "error",
                "reason_zh": "找不到 vae",
                "reason_en": "vae not found",
                "hint_zh": "检查路径",
                "hint_en": "Check the path",
                "field": "vae",
            }
        ]
        result = ts.preflight(self.url, "anima-lora", {"vae": "x"})
        self.assertEqual(result["verdict"], "problems")
        self.assertEqual({i["code"] for i in result["issues"]}, {"cuda_unavailable"})
        self.assertEqual(result["config_issues"][0]["field"], "vae")
        self.assertEqual(result["config_issues"][0]["hint_zh"], "检查路径")

    def test_not_ready_engine_reports_its_reason_and_hint(self) -> None:
        Fake.engines_state = "broken"
        result = ts.preflight(self.url, "anima-lora", {})
        self.assertEqual(result["verdict"], "problems")
        self.assertEqual(result["engine"]["reason_en"], "A package is missing")
        self.assertEqual(result["engine"]["hint_zh"], "pip install x")

    def test_diagnose_gets_the_train_type_and_the_config(self) -> None:
        ts.preflight(self.url, "anima-lora", {"output_name": "x"})
        self.assertIn("POST /api/engines/diagnose", Fake.log)

    def test_repair_is_never_called_and_reinstall_is_only_described(self) -> None:
        Fake.diagnose_issues = [ERROR_ISSUE]
        result = ts.preflight(self.url, "anima-lora", {})
        self.assertFalse([line for line in Fake.log if "repair" in line])
        self.assertIn("reinstall", [a["id"] for a in result["actions_needing_the_user"]])
        self.assertIn("full reinstall", result["actions_note"])

    def test_old_trainer_is_skipped_with_a_note(self) -> None:
        Fake.mode = "old"
        result = ts.preflight(self.url, "anima-lora", {})
        self.assertEqual(result["verdict"], "unknown")
        self.assertTrue(any("older" in n for n in result["notes"]))
        self.assertEqual(result["issues"], [])

    def test_unreachable_trainer_is_a_note_not_a_crash(self) -> None:
        result = ts.preflight("http://127.0.0.1:1", "anima-lora", {})
        self.assertEqual(result["verdict"], "unknown")
        self.assertTrue(any("not reachable" in n for n in result["notes"]))


class QueueTests(Case):
    def test_restored_waiting_queue_needs_the_users_word(self) -> None:
        Fake.queue = queue_state(
            entries=[{"id": "e1", "name": "night", "status": "queued", "error": None}],
            restored={
                "queued": 1,
                "interrupted": 0,
                "waiting": True,
                "message_en": "Queue restored after a restart: 1 waiting.",
                "message_zh": "服务重启后已恢复训练队列：1 个任务在等待。",
            },
            active=False,
        )
        result = ts.queue_summary(self.url)
        self.assertTrue(result["ask_user_before_start"])
        self.assertIn("1 waiting", result["restored"]["message_en"])
        self.assertEqual([line for line in Fake.log if line.startswith("POST")], [])

    def test_interrupted_entries_are_listed_and_never_requeued(self) -> None:
        Fake.queue = queue_state(
            entries=[
                {
                    "id": "e1",
                    "name": "a",
                    "status": "failed",
                    "interrupted": True,
                    "error": "restart",
                },
                {"id": "e2", "name": "b", "status": "failed", "error": "OOM"},
            ],
            restored={"queued": 0, "interrupted": 1, "waiting": False},
        )
        result = ts.queue_summary(self.url)
        self.assertEqual([e["id"] for e in result["interrupted"]], ["e1"])
        self.assertFalse(result["auto_requeue"])
        self.assertFalse(result["ask_user_before_start"])
        self.assertFalse([line for line in Fake.log if "requeue" in line])

    def test_idle_queue_has_nothing_to_ask(self) -> None:
        result = ts.queue_summary(self.url)
        self.assertFalse(result["ask_user_before_start"])
        self.assertEqual(result["interrupted"], [])

    def test_old_trainer_without_a_queue(self) -> None:
        Fake.mode = "old"
        result = ts.queue_summary(self.url)
        self.assertFalse(result["available"])
        self.assertIn("older", result["note"])

    def test_start_needs_confirmation_and_posts_nothing_without_it(self) -> None:
        refused = ts.start_queue(self.url, confirmed=False)
        self.assertEqual(refused["status"], "refused")
        self.assertEqual(Fake.log, [])
        done = ts.start_queue(self.url, confirmed=True)
        self.assertEqual(done["status"], "success")
        self.assertEqual(Fake.log, ["POST /api/queue/start"])

    def test_start_on_an_old_trainer_is_skipped(self) -> None:
        Fake.mode = "old"
        self.assertEqual(ts.start_queue(self.url, confirmed=True)["status"], "unavailable")


class SubmitTests(Case):
    config = {"output_name": "zkz-anima-v1", "output_dir": "./out"}

    def test_refuses_without_confirmation(self) -> None:
        result = ts.submit(self.url, self.config, confirmed=False)
        self.assertEqual(result["status"], "refused")
        self.assertEqual(Fake.log, [])

    def test_queued_submit_follows_the_entry_to_its_task_and_reports_the_rename(self) -> None:
        Fake.run_response = {
            "status": "success",
            "message": "queued",
            "data": {
                "queued": True,
                "entry_id": "e9",
                "output_name_renamed": {"from": "zkz-anima-v1", "to": "zkz-anima-v1_2"},
            },
        }
        Fake.queue = queue_state(
            entries=[{"id": "e9", "name": "zkz-anima-v1_2", "status": "running", "task_id": "t-9"}],
            running={"id": "e9", "task_id": "t-9"},
        )
        result = ts.submit(
            self.url, self.config, confirmed=True, wait_seconds=2, sleep=lambda s: None
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["queued"])
        self.assertEqual(result["task_id"], "t-9")
        self.assertEqual(result["output_name"], "zkz-anima-v1_2")
        self.assertEqual(
            result["output_name_renamed"], {"from": "zkz-anima-v1", "to": "zkz-anima-v1_2"}
        )
        self.assertIn("zkz-anima-v1_2", result["use_this_name_for"])
        self.assertEqual(result["log_tail_url"], f"{self.url}/api/train/log/tail/t-9")

    def test_paused_queue_is_reported_and_not_started_for_the_user(self) -> None:
        Fake.run_response = {"status": "success", "data": {"queued": True, "entry_id": "e1"}}
        Fake.queue = queue_state(
            entries=[{"id": "e1", "name": "zkz-anima-v1", "status": "queued", "task_id": None}],
            active=False,
        )
        result = ts.submit(
            self.url, self.config, confirmed=True, wait_seconds=0, sleep=lambda s: None
        )
        self.assertIsNone(result["task_id"])
        self.assertTrue(result["waiting_for_queue_start"])
        self.assertEqual(result["output_name"], "zkz-anima-v1")
        self.assertFalse([line for line in Fake.log if line == "POST /api/queue/start"])

    def test_trainer_without_a_queue_returns_the_task_id_directly(self) -> None:
        Fake.mode = "old"
        Fake.run_response = {
            "status": "success",
            "data": {"task_id": "t-1", "train_log_stream_url": "http://x/stream"},
        }
        result = ts.submit(self.url, self.config, confirmed=True)
        self.assertEqual(result["task_id"], "t-1")
        self.assertFalse(result["queued"])
        self.assertEqual(result["output_name"], "zkz-anima-v1")
        self.assertIsNone(result["output_name_renamed"])

    def test_failure_message_is_passed_through(self) -> None:
        Fake.run_response = {
            "status": "fail",
            "message": "训练数据集路径不存在或没有图片，请检查目录。",
        }
        result = ts.submit(self.url, self.config, confirmed=True)
        self.assertEqual(result["status"], "fail")
        self.assertIn("没有图片", result["message"])
        self.assertIsNone(result["task_id"])


class CliTests(Case):
    def run_cli(self, *args: str) -> tuple[int, dict]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ts.main(["--base-url", self.url, *args])
        return code, json.loads(out.getvalue())

    def test_preflight_reads_the_config_file(self) -> None:
        cfg = self.tmp / "cfg.json"
        cfg.write_text(
            json.dumps({"model_train_type": "anima-lora", "output_name": "x"}), encoding="utf-8"
        )
        code, data = self.run_cli("preflight", "--config", str(cfg))
        self.assertEqual((code, data["verdict"]), (0, "ok"))

    def test_submit_without_confirmed_exits_2(self) -> None:
        cfg = self.tmp / "cfg.json"
        cfg.write_text(json.dumps({"output_name": "x"}), encoding="utf-8")
        code, data = self.run_cli("submit", "--config", str(cfg))
        self.assertEqual((code, data["status"]), (2, "refused"))

    def test_queue_command(self) -> None:
        code, data = self.run_cli("queue")
        self.assertEqual(code, 0)
        self.assertIn("ask_user_before_start", data)


if __name__ == "__main__":
    unittest.main()
