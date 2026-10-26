"""in_use_guard.py and its use in build_dataset.py: stop before rewriting a dataset in use."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType
from unittest import mock

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pipeline script: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_script("in_use_guard")
build_dataset = load_script("build_dataset")
tag_dataset = load_script("tag_dataset")

TASK = {"task_id": "t1", "name": "night-lora", "status": "running", "relation": "inside"}


class _Handler(BaseHTTPRequestHandler):
    # class attributes are set per test: (status, body) per request path prefix
    reply: tuple[int, object] = (200, {})
    seen: list[str] = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append(self.path)
        status, body = type(self).reply
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class TrainerStub:
    def __init__(self, status: int, body: object):
        handler = type("H", (_Handler,), {"reply": (status, body), "seen": []})
        self.handler = handler
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def in_use_body(in_use: bool) -> dict:
    return {"status": "success", "data": {"in_use": in_use, "tasks": [TASK] if in_use else []}}


class CheckInUseTests(unittest.TestCase):
    def test_in_use_and_free(self):
        with TrainerStub(200, in_use_body(True)) as stub:
            result = guard.check_in_use("D:/data/cat", stub.url)
            self.assertEqual(result["state"], "in_use")
            self.assertEqual(result["tasks"][0]["name"], "night-lora")
            self.assertIn("path=D%3A%2Fdata%2Fcat", stub.handler.seen[0])
        with TrainerStub(200, in_use_body(False)) as stub:
            self.assertEqual(guard.check_in_use("D:/data/cat", stub.url)["state"], "free")

    def test_trouble_means_unknown_never_in_use(self):
        with TrainerStub(404, {"detail": "Not Found"}) as stub:
            result = guard.check_in_use("D:/x", stub.url)
        self.assertEqual(result["state"], "unknown")
        self.assertIn("older build", result["why"])
        with TrainerStub(200, {"status": "success", "data": {}}) as stub:
            self.assertEqual(guard.check_in_use("D:/x", stub.url)["state"], "unknown")
        # nothing listens on this port
        self.assertEqual(
            guard.check_in_use("D:/x", "http://127.0.0.1:9", timeout=1)["state"], "unknown"
        )


class GuardPathsTests(unittest.TestCase):
    def test_in_use_stops_with_exit_3_unless_allowed(self):
        with TrainerStub(200, in_use_body(True)) as stub:
            self.assertEqual(guard.guard_paths(["D:/a"], stub.url), guard.EXIT_IN_USE)
            self.assertEqual(guard.guard_paths(["D:/a"], stub.url, allow_in_use=True), 0)

    def test_free_and_unknown_go_on(self):
        with TrainerStub(200, in_use_body(False)) as stub:
            self.assertEqual(guard.guard_paths(["D:/a", "D:/b"], stub.url), 0)
        with TrainerStub(404, {"detail": "Not Found"}) as stub:
            self.assertEqual(guard.guard_paths(["D:/a"], stub.url), 0)


class BuildDatasetGuardTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="in-use-build-"))
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        (self.work / "raw").mkdir()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.work / "raw" / "a.png")
        manifest = {"keep": [{"file": "a.png", "mode": "RGB"}]}
        (self.work / "curation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        old = self.work / "dataset" / "3_mychar"
        old.mkdir(parents=True)
        (old / "old.txt").write_text("keep me", encoding="utf-8")

    def run_main(self, trainer_url: str, *extra: str) -> int:
        argv = [
            "build_dataset.py", "--work", str(self.work), "--concept", "mychar",
            "--repeats", "5", "--trainer-url", trainer_url, *extra,
        ]  # fmt: skip
        with mock.patch.object(sys, "argv", argv):
            return build_dataset.main()

    def test_in_use_dataset_is_left_untouched(self):
        with TrainerStub(200, in_use_body(True)) as stub:
            code = self.run_main(stub.url)
        self.assertEqual(code, 3)
        self.assertTrue((self.work / "dataset" / "3_mychar" / "old.txt").is_file())
        self.assertFalse((self.work / "dataset" / "5_mychar").exists())

    def test_allow_in_use_goes_on(self):
        with TrainerStub(200, in_use_body(True)) as stub:
            code = self.run_main(stub.url, "--allow-in-use")
        self.assertEqual(code, 0)
        self.assertTrue((self.work / "dataset" / "5_mychar" / "a.png").is_file())

    def test_unreachable_trainer_does_not_block_a_build(self):
        self.assertEqual(self.run_main("http://127.0.0.1:9"), 0)
        self.assertTrue((self.work / "dataset" / "5_mychar" / "a.png").is_file())


class TagDatasetGuardTests(unittest.TestCase):
    """tag_dataset.py rewrites captions in the folder, so it asks the trainer first."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="in-use-tag-")) / "5_mychar"
        self.folder.mkdir()
        self.addCleanup(shutil.rmtree, self.folder.parent, ignore_errors=True)

    def run_main(self, trainer_url: str, *extra: str) -> int:
        argv = [
            "tag_dataset.py", "--dataset-dir", str(self.folder), "--trigger", "mychar",
            "--subject-tag", "1girl", "--trainer-url", trainer_url, *extra,
        ]  # fmt: skip
        stop = RuntimeError("reached the sorter")
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(tag_dataset, "require_sorter_capabilities", side_effect=stop),
        ):
            try:
                return tag_dataset.main()
            except RuntimeError as error:
                self.assertIs(error, stop)
                return -1  # got past the guard to the first sorter call

    def test_in_use_dataset_stops_before_the_sorter_is_called(self):
        with TrainerStub(200, in_use_body(True)) as stub:
            self.assertEqual(self.run_main(stub.url), 3)

    def test_allow_in_use_goes_on(self):
        with TrainerStub(200, in_use_body(True)) as stub:
            self.assertEqual(self.run_main(stub.url, "--allow-in-use"), -1)

    def test_free_unknown_and_unreachable_trainers_go_on(self):
        with TrainerStub(200, in_use_body(False)) as stub:
            self.assertEqual(self.run_main(stub.url), -1)
        with TrainerStub(404, {"detail": "Not Found"}) as stub:
            self.assertEqual(self.run_main(stub.url), -1)
        self.assertEqual(self.run_main("http://127.0.0.1:9"), -1)


if __name__ == "__main__":
    unittest.main()
