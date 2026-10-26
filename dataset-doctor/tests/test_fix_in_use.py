"""fix_dataset.py --apply asks the trainer whether the dataset is in use first."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import fix_dataset as F  # noqa: E402
import fix_extras as X  # noqa: E402

TASK = {"task_id": "t1", "name": "night-lora", "status": "running"}


class Handler(BaseHTTPRequestHandler):
    status, body = 200, {}

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        raw = json.dumps(type(self).body).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class Trainer:
    def __init__(self, status: int, body: dict) -> None:
        handler = type("H", (Handler,), {"status": status, "body": body})
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "Trainer":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def in_use(flag: bool) -> dict:
    return {"status": "success", "data": {"in_use": flag, "tasks": [TASK] if flag else []}}


class FixInUseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="dd-inuse-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        concept = self.root / "5_zkz"
        concept.mkdir()
        (concept / "a.png").write_bytes(b"x")
        self.caption = concept / "a.txt"
        self.caption.write_text("1girl, 2girls", encoding="utf-8")

    def run_fix(self, url: str, *extra: str) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = F.main(["fix-subject-count", str(self.root), "--trainer-url", url, *extra])
        return code, err.getvalue()

    def test_apply_on_a_dataset_in_use_stops_and_changes_nothing(self) -> None:
        with Trainer(200, in_use(True)) as trainer:
            code, err = self.run_fix(trainer.url, "--apply")
        self.assertEqual(code, 3)
        self.assertIn("night-lora", err)
        self.assertEqual(self.caption.read_text(encoding="utf-8"), "1girl, 2girls")

    def test_allow_in_use_goes_on(self) -> None:
        with Trainer(200, in_use(True)) as trainer:
            code, err = self.run_fix(trainer.url, "--apply", "--allow-in-use")
        self.assertEqual(code, 0)
        self.assertIn("continuing", err)
        self.assertEqual(self.caption.read_text(encoding="utf-8"), "2girls")

    def test_dry_run_never_asks_the_trainer(self) -> None:
        with mock.patch.object(X, "guard_apply", side_effect=AssertionError("asked")):
            code, _ = self.run_fix("http://127.0.0.1:9")
        self.assertEqual(code, 0)

    def test_older_or_unreachable_trainer_does_not_block(self) -> None:
        with Trainer(404, {"detail": "Not Found"}) as trainer:
            self.assertEqual(self.run_fix(trainer.url, "--apply")[0], 0)
        self.caption.write_text("1girl, 2girls", encoding="utf-8")
        code, err = self.run_fix("http://127.0.0.1:9", "--apply")
        self.assertEqual(code, 0)
        self.assertIn("could not check", err)

    def test_missing_guard_module_is_a_note_not_an_error(self) -> None:
        with mock.patch.object(X, "load_in_use_guard", return_value=None):
            self.assertEqual(X.guard_apply(self.root, "http://127.0.0.1:9", False), 0)


if __name__ == "__main__":
    unittest.main()
