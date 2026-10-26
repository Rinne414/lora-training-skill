"""wddm_sysmem.py: shared GPU memory of the training process (Windows sysmem fallback)."""

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

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "wddm_sysmem.py"
GIB = 1024**3


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wddm_sysmem", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ws = load_script()


def counter_rows(pid: int, shared: int, dedicated: int, luid: str = "0x0001bba1") -> list[dict]:
    inst = f"pid_{pid}_luid_0x00000000_{luid}_phys_0"
    base = "\\\\PC\\gpu process memory(" + inst + ")\\"
    return [
        {"path": base + "shared usage", "instance": inst, "value": shared},
        {"path": base + "dedicated usage", "instance": inst, "value": dedicated},
    ]


class FakeRunner:
    """Stands in for PowerShell and nvidia-smi: returns canned text per call kind."""

    def __init__(self, rows=None, procs=None, smi="24576, 23000", fail=None) -> None:
        self.rows, self.procs, self.smi, self.fail = rows or [], procs or [], smi, fail

    def counters(self) -> str:
        if self.fail == "counters":
            raise ws.CheckUnavailable("Get-Counter failed: counter set not found")
        return json.dumps(self.rows[0] if len(self.rows) == 1 else self.rows)

    def processes(self) -> str:
        return json.dumps(self.procs)

    def gpu_memory(self) -> str:
        if self.fail == "smi":
            raise ws.CheckUnavailable("nvidia-smi not found")
        return self.smi


class ParseTests(unittest.TestCase):
    def test_rows_are_summed_per_pid_across_adapters(self) -> None:
        rows = (
            counter_rows(10, 2 * GIB, 5 * GIB)
            + counter_rows(10, GIB, 0, luid="0x0002")
            + counter_rows(11, 7, 7)
        )
        usage = ws.usage_by_pid(rows)
        self.assertEqual(usage[10], {"shared": 3 * GIB, "dedicated": 5 * GIB})
        self.assertEqual(usage[11], {"shared": 7, "dedicated": 7})

    def test_find_pids_never_returns_the_calling_script(self) -> None:
        procs = [
            {
                "ProcessId": 5,
                "ParentProcessId": 0,
                "Name": "python.exe",
                "CommandLine": "python wddm_sysmem.py --match train.py",
            },
            {
                "ProcessId": 6,
                "ParentProcessId": 0,
                "Name": "python.exe",
                "CommandLine": "python train.py",
            },
        ]
        self.assertEqual(ws.find_pids(json.dumps(procs), "train.py", exclude={5}), [6])

    def test_single_object_json_is_accepted(self) -> None:
        one = counter_rows(10, 5, 6)[0]
        self.assertEqual(len(ws.parse_rows(json.dumps(one))), 1)
        self.assertEqual(ws.parse_rows(""), [])

    def test_other_counters_and_odd_instances_are_ignored(self) -> None:
        rows = [
            {
                "path": "\\\\PC\\gpu process memory(_total)\\shared usage",
                "instance": "_total",
                "value": 9,
            }
        ]
        rows += [
            {
                "path": "\\\\PC\\gpu process memory(pid_5_x)\\local usage",
                "instance": "pid_5_x",
                "value": 9,
            }
        ]
        self.assertEqual(ws.usage_by_pid(rows), {})

    def test_find_pids_matches_command_line_and_skips_non_python(self) -> None:
        procs = [
            {
                "ProcessId": 1,
                "ParentProcessId": 0,
                "Name": "python.exe",
                "CommandLine": "python train.py --config_file C:/c/20261027-032119.toml",
            },
            {
                "ProcessId": 2,
                "ParentProcessId": 1,
                "Name": "python.exe",
                "CommandLine": "python other.py",
            },
            {"ProcessId": 3, "ParentProcessId": 1, "Name": "python.exe", "CommandLine": None},
        ]
        self.assertEqual(ws.find_pids(json.dumps(procs), "20261027-032119.toml"), [1])
        self.assertEqual(ws.find_pids(json.dumps(procs[0]), "20261027-032119.toml"), [1])
        self.assertEqual(ws.find_pids("", "x"), [])


class EvaluateTests(unittest.TestCase):
    def test_shared_over_threshold_with_full_vram_warns(self) -> None:
        verdict = ws.evaluate(3 * GIB, 23 * GIB, 24 * GIB, 23.5 * GIB)
        self.assertEqual(verdict["status"], "warn")
        self.assertIn("Prefer No Sysmem Fallback", verdict["fix_en"])
        self.assertIn("首选无系统内存回退", verdict["fix_zh"])
        self.assertIn("slower", verdict["message_en"].lower())

    def test_shared_over_threshold_but_vram_not_full_is_not_a_warning(self) -> None:
        verdict = ws.evaluate(3 * GIB, 8 * GIB, 24 * GIB, 9 * GIB)
        self.assertEqual(verdict["status"], "ok")
        self.assertTrue(verdict["note_en"])

    def test_small_shared_usage_is_normal_even_when_vram_is_full(self) -> None:
        self.assertEqual(ws.evaluate(300 * 1024**2, 23 * GIB, 24 * GIB, 23.5 * GIB)["status"], "ok")

    def test_unknown_capacity_downgrades_to_shared_high(self) -> None:
        verdict = ws.evaluate(3 * GIB, 8 * GIB, None, None)
        self.assertEqual(verdict["status"], "shared_high_vram_unknown")

    def test_threshold_is_exclusive_and_configurable(self) -> None:
        self.assertEqual(ws.evaluate(GIB, 0, 24 * GIB, 24 * GIB)["status"], "ok")
        self.assertEqual(ws.evaluate(GIB + 1, 0, 24 * GIB, 24 * GIB)["status"], "warn")
        self.assertEqual(
            ws.evaluate(GIB, 0, 24 * GIB, 24 * GIB, threshold_bytes=GIB // 2)["status"], "warn"
        )


class CheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "wddm.json"

    def test_warns_once_then_stays_quiet(self) -> None:
        runner = FakeRunner(rows=counter_rows(7, 4 * GIB, 23 * GIB), smi="24576, 24000")
        first = ws.check([7], runner=runner, state_path=self.state, platform="win32")
        second = ws.check([7], runner=runner, state_path=self.state, platform="win32")
        self.assertEqual(first["status"], "warn")
        self.assertTrue(first["announce"])
        self.assertEqual(second["status"], "warn")
        self.assertFalse(second["announce"])

    def test_clears_when_the_problem_goes_away_and_can_warn_again(self) -> None:
        bad = FakeRunner(rows=counter_rows(7, 4 * GIB, 23 * GIB), smi="24576, 24000")
        good = FakeRunner(rows=counter_rows(7, 0, 9 * GIB), smi="24576, 9000")
        ws.check([7], runner=bad, state_path=self.state, platform="win32")
        self.assertEqual(
            ws.check([7], runner=good, state_path=self.state, platform="win32")["status"], "ok"
        )
        self.assertTrue(
            ws.check([7], runner=bad, state_path=self.state, platform="win32")["announce"]
        )

    def test_not_windows_is_unavailable_not_an_error(self) -> None:
        result = ws.check([7], runner=FakeRunner(), platform="linux")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("Windows", result["reason"])

    def test_counter_failure_says_could_not_check(self) -> None:
        result = ws.check([7], runner=FakeRunner(fail="counters"), platform="win32")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("could not check", result["message_en"].lower())
        self.assertIn("无法检查", result["message_zh"])
        linux = ws.check([7], runner=FakeRunner(), platform="linux")
        self.assertIn("只有 Windows", linux["message_zh"])

    def test_missing_nvidia_smi_still_reports_shared_usage(self) -> None:
        runner = FakeRunner(rows=counter_rows(7, 4 * GIB, 8 * GIB), fail="smi")
        result = ws.check([7], runner=runner, platform="win32")
        self.assertEqual(result["status"], "shared_high_vram_unknown")

    def test_pid_without_counters_is_reported_as_such(self) -> None:
        runner = FakeRunner(rows=counter_rows(99, 1, 1))
        result = ws.check([7], runner=runner, platform="win32")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no GPU counters", result["reason"])

    def test_no_pids_is_unavailable(self) -> None:
        self.assertEqual(
            ws.check([], runner=FakeRunner(), platform="win32")["status"], "unavailable"
        )

    def test_numbers_are_reported_in_gib(self) -> None:
        runner = FakeRunner(rows=counter_rows(7, 3 * GIB, 20 * GIB), smi="24576, 22000")
        result = ws.check([7], runner=runner, platform="win32")
        self.assertEqual(result["shared_gib"], 3.0)
        self.assertEqual(result["dedicated_gib"], 20.0)
        self.assertEqual(result["vram_total_gib"], 24.0)


class TaskLookup(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        body = {
            "status": "success",
            "data": {
                "tasks": [
                    {
                        "id": "t1",
                        "status": "RUNNING",
                        "metadata": {"config_path": "C:\\cfg\\autosave\\20261027-1.toml"},
                    }
                ]
            },
        }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class CliTests(unittest.TestCase):
    def test_config_match_comes_from_the_trainer_task(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), TaskLookup)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"
        self.assertEqual(ws.task_config_name(url, "t1"), "20261027-1.toml")
        self.assertIsNone(ws.task_config_name(url, "other"))
        self.assertIsNone(ws.task_config_name("http://127.0.0.1:1", "t1"))

    def test_cli_without_a_target_prints_usage_error(self) -> None:
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ws.main([])


if __name__ == "__main__":
    unittest.main()
