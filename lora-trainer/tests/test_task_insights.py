"""task_insights.py: trainer API first, folder scan for older trainers."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "task_insights.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task_insights", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


insights = load_script()

TASK_ID = "task-1"
PNG_BYTES = b"\x89PNG-fake"
LOSS = {
    "loss/average": [
        {"step": 1, "value": 0.5},
        {"step": 2, "value": 0.25},
        {"step": 3, "value": 0.3},
    ],
    "loss/epoch_average": [{"step": 1, "value": 0.4}],
}


class FakeTrainer(BaseHTTPRequestHandler):
    """mode: "new" answers the routes, "old" is a trainer without them, "restarted" forgot the id."""

    mode = "new"

    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if self.mode == "old":
            return self._json(404, {"detail": "Not Found"})
        if self.mode == "restarted":
            return self._json(404, {"detail": "Unknown task_id. It is only valid for jobs..."})
        if path == f"/api/tasks/{TASK_ID}/metrics":
            data = {"tags": LOSS, "progress": {"step": 3, "total_steps": 3}}
            return self._json(200, {"status": "success", "data": data})
        if path == f"/api/tasks/{TASK_ID}/previews":
            image = {"name": "run_e000001_00_a.png", "epoch": 1, "step": None, "mtime": 1.0}
            image["url"] = f"/api/tasks/{TASK_ID}/previews/run_e000001_00_a.png"
            return self._json(
                200, {"status": "success", "data": {"images": [image], "preview_enabled": True}}
            )
        if path == f"/api/tasks/{TASK_ID}/previews/run_e000001_00_a.png":
            return self._send(200, PNG_BYTES, "image/png")
        return self._json(404, {"detail": "Not Found"})


class TrainerServer:
    def __init__(self, mode: str) -> None:
        handler = type("Handler", (FakeTrainer,), {"mode": mode})
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "TrainerServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)
    os.utime(path, (mtime, mtime))


class ApiPathTests(unittest.TestCase):
    def test_new_trainer_is_used_and_summarised(self) -> None:
        with TrainerServer("new") as trainer:
            result = insights.collect(trainer.url, TASK_ID)
        self.assertEqual(result["source"], "trainer_api")
        self.assertEqual([i["name"] for i in result["previews"]], ["run_e000001_00_a.png"])
        self.assertTrue(result["previews"][0]["url"].startswith("http://127.0.0.1:"))
        summary = result["loss_summary"]
        self.assertEqual((summary["first"], summary["last"], summary["min"]), (0.5, 0.3, 0.25))
        self.assertEqual(summary["min_step"], 2)
        self.assertFalse(summary["has_nan_or_inf"])
        self.assertEqual(summary["epoch_average"], [0.4])

    def test_previews_download_to_folder(self) -> None:
        with TrainerServer("new") as trainer, tempfile.TemporaryDirectory() as tmp:
            result = insights.collect(trainer.url, TASK_ID)
            saved = insights.download_previews(result["previews"], Path(tmp) / "out")
            self.assertEqual([Path(p).name for p in saved], ["run_e000001_00_a.png"])
            self.assertEqual(Path(saved[0]).read_bytes(), PNG_BYTES)

    def test_old_trainer_falls_back_to_folders(self) -> None:
        with TrainerServer("old") as trainer, tempfile.TemporaryDirectory() as tmp:
            touch(Path(tmp) / "out" / "sample" / "run_e000001_00_a.png", 2000)
            result = insights.collect(
                trainer.url, TASK_ID, str(Path(tmp) / "out"), "run", None, 1500
            )
        self.assertEqual(result["source"], "folder_scan")
        self.assertEqual(result["fallback_because"], "trainer_has_no_task_insights_api")
        self.assertEqual([i["name"] for i in result["previews"]], ["run_e000001_00_a.png"])

    def test_restarted_trainer_falls_back_with_its_own_reason(self) -> None:
        with TrainerServer("restarted") as trainer:
            result = insights.collect(trainer.url, TASK_ID)
        self.assertEqual(result["source"], "folder_scan")
        self.assertEqual(result["fallback_because"], "task_unknown_to_trainer")

    def test_unreachable_trainer_falls_back(self) -> None:
        with TrainerServer("new") as trainer:
            url = trainer.url
        result = insights.collect(url, TASK_ID)  # server already shut down
        self.assertEqual(result["source"], "folder_scan")
        self.assertTrue(result["fallback_because"].startswith("trainer_unreachable"))


class FolderScanTests(unittest.TestCase):
    def test_since_keeps_an_earlier_same_name_run_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample"
            touch(sample / "run_e000001_00_old.png", 1000)
            touch(sample / "run_000010_00_new.png", 5000)
            touch(sample / "other_000010_00_x.png", 5000)
            touch(sample / ".thumbs" / "run_hidden.png", 5000)
            names = [p.name for p in insights.scan_previews(Path(tmp), "run", 4000)]
            self.assertEqual(names, ["run_000010_00_new.png"])
            everything = [p.name for p in insights.scan_previews(Path(tmp), "run", None)]
            self.assertEqual(everything, ["run_e000001_00_old.png", "run_000010_00_new.png"])

    def test_missing_folders_give_reasons_not_errors(self) -> None:
        result = insights.fetch_from_folders("/no/such/dir", "run", "/no/such/logs", 1.0)
        self.assertEqual(result["previews"], [])
        self.assertEqual(result["loss_reason"], "logging_dir_missing")
        self.assertIn("previews_reason", result)

    def test_missing_since_is_called_out(self) -> None:
        result = insights.fetch_from_folders(None, "", None, None)
        self.assertTrue(any("since" in note for note in result["notes"]))


class SummaryTests(unittest.TestCase):
    def test_nan_is_flagged(self) -> None:
        points = [{"step": 1, "value": 0.4}, {"step": 2, "value": float("nan")}]
        summary = insights.summarize_loss({"loss/average": points})
        self.assertTrue(summary["has_nan_or_inf"])
        self.assertEqual(summary["min"], 0.4)

    def test_empty_loss_has_no_summary(self) -> None:
        self.assertEqual(insights.summarize_loss({}), {})

    def test_falls_back_to_current_loss_tag(self) -> None:
        summary = insights.summarize_loss({"loss/current": [{"step": 1, "value": 1.0}]})
        self.assertEqual(summary["tag"], "loss/current")


if __name__ == "__main__":
    unittest.main()
