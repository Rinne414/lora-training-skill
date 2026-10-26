"""lora_base_info.py: 28 vs 40 blocks from a safetensors header."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lora_base_info.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lora_base_info", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


info = load_script()


def write_safetensors(path: Path, names: list[str], metadata: dict | None = None) -> Path:
    header: dict = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
        for i, name in enumerate(names)
    }
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(names)))
    return path


def lora_keys(blocks: range, prefix: str = "lora_unet_blocks_") -> list[str]:
    return [
        f"{prefix}{n}_{part}.{suffix}"
        for n in blocks
        for part in ("self_attn_q_proj", "cross_attn_k_proj")
        for suffix in ("alpha", "lora_down.weight", "lora_up.weight")
    ]


class LoraBaseInfoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_28_block_lora(self) -> None:
        path = write_safetensors(self.dir / "a.safetensors", lora_keys(range(28)))
        result = info.inspect_lora(path)
        self.assertEqual((result["highest_block"], result["block_count"]), (27, 28))
        self.assertEqual(result["base"], "anima-2b")

    def test_any_block_28_to_39_means_40(self) -> None:
        keys = lora_keys(range(10)) + lora_keys(range(35, 36))
        result = info.inspect_lora(write_safetensors(self.dir / "b.safetensors", keys))
        self.assertEqual((result["block_count"], result["base"]), (40, "anima-2.9b"))

    def test_block_index_beyond_39_is_not_a_known_layout(self) -> None:
        result = info.inspect_lora(
            write_safetensors(self.dir / "c.safetensors", lora_keys(range(41)))
        )
        self.assertIsNone(result["block_count"])
        self.assertIn("note", result)

    def test_other_key_styles(self) -> None:
        for prefix in ("diffusion_model.blocks.", "lycoris_blocks_", "transformer.blocks."):
            sep = "." if prefix.endswith(".") else "_"
            names = [f"{prefix}{n}{sep}self_attn.q_proj.lora_A.weight" for n in (0, 30)]
            names = [n.replace("self_attn.q_proj", "self_attn_q_proj") for n in names]
            result = info.inspect_lora(write_safetensors(self.dir / "d.safetensors", names))
            self.assertEqual(result["block_count"], 40, prefix)

    def test_wan_style_keys_are_not_called_anima(self) -> None:
        names = [f"diffusion_model.blocks.{n}.cross_attn.q.lora_down.weight" for n in range(40)]
        result = info.inspect_lora(write_safetensors(self.dir / "wan.safetensors", names))
        self.assertFalse(result["is_anima"])
        self.assertIsNone(result["block_count"])
        report = info.build_report(self.dir / "wan.safetensors")
        self.assertNotIn("warnings", report)

    def test_28_lora_on_40_base_warns_in_both_languages(self) -> None:
        lora = write_safetensors(self.dir / "l.safetensors", lora_keys(range(28)))
        base = write_safetensors(
            self.dir / "base.safetensors",
            [f"net.blocks.{n}.self_attn.q_proj.weight" for n in range(40)],
        )
        report = info.build_report(lora, base)
        self.assertFalse(report["compatible"])
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("28to40", report["warnings"][0]["en"])
        self.assertIn("重映射", report["warnings"][0]["zh"])
        self.assertEqual(report["configured_base_model"]["block_count"], 40)

    def test_40_lora_on_28_base_warns(self) -> None:
        lora = write_safetensors(self.dir / "l.safetensors", lora_keys(range(40)))
        base = write_safetensors(
            self.dir / "base.safetensors",
            [f"net.blocks.{n}.self_attn.q_proj.weight" for n in range(28)],
        )
        report = info.build_report(lora, base)
        self.assertFalse(report["compatible"])
        self.assertIn("does not load correctly on the 2B", report["warnings"][0]["en"])

    def test_matching_base_has_no_warning(self) -> None:
        lora = write_safetensors(self.dir / "l.safetensors", lora_keys(range(28)))
        base = write_safetensors(
            self.dir / "base.safetensors", [f"net.blocks.{n}.mlp.layer1.weight" for n in range(28)]
        )
        report = info.build_report(lora, base)
        self.assertTrue(report["compatible"])
        self.assertEqual(report["warnings"], [])

    def test_without_a_base_the_fact_is_still_stated_as_a_note(self) -> None:
        report = info.build_report(
            write_safetensors(self.dir / "l.safetensors", lora_keys(range(28)))
        )
        self.assertIsNone(report["compatible"])
        self.assertEqual(report["warnings"], [])
        self.assertTrue(any("remap" in n["en"] for n in report["notes"]))

    def test_unreadable_base_does_not_fail_the_report(self) -> None:
        lora = write_safetensors(self.dir / "l.safetensors", lora_keys(range(28)))
        report = info.build_report(lora, self.dir / "missing.safetensors")
        self.assertIn("error", report["configured_base_model"])
        self.assertIsNone(report["compatible"])

    def test_broken_files(self) -> None:
        for name, content in (("empty", b""), ("short", b"\1\0\0"), ("huge", b"\xff" * 16)):
            path = self.dir / f"{name}.safetensors"
            path.write_bytes(content)
            with self.assertRaises(ValueError, msg=name):
                info.read_safetensors_header(path)
            self.assertIn("error", info.build_report(path))

    def test_newest_lora_prefers_final_then_latest_snapshot(self) -> None:
        self.assertIsNone(info.newest_lora(self.dir, "zkz"))
        first = write_safetensors(self.dir / "zkz-000002.safetensors", lora_keys(range(1)))
        self.assertEqual(info.newest_lora(self.dir, "zkz"), first)
        final = write_safetensors(self.dir / "zkz.safetensors", lora_keys(range(1)))
        self.assertEqual(info.newest_lora(self.dir, "zkz"), final)

    def test_cli_writes_sidecar(self) -> None:
        lora = write_safetensors(self.dir / "zkz.safetensors", lora_keys(range(28)))
        with contextlib.redirect_stdout(io.StringIO()):
            code = info.main([str(lora), "--out", str(self.dir / "ins")])
        self.assertEqual(code, 0)
        data = json.loads((self.dir / "ins" / "zkz.base.json").read_text(encoding="utf-8"))
        self.assertEqual(data["lora"]["block_count"], 28)


if __name__ == "__main__":
    unittest.main()
