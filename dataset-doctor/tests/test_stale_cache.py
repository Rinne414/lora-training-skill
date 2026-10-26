"""Stale training caches: check_cache.py, the doctor warning, and quarantine-cache."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import _common as C  # noqa: E402
import check_cache as K  # noqa: E402
import doctor  # noqa: E402
import fix_dataset as F  # noqa: E402
import fix_extras as X  # noqa: E402
from PIL import Image  # noqa: E402

OLD, NEW = 1_700_000_000, 1_700_100_000  # seconds; a day apart


def set_time(path: Path, when: float) -> Path:
    os.utime(path, (when, when))
    return path


def write_safetensors(
    path: Path, metadata: dict | None, tensors: tuple[str, ...] = ("latents",)
) -> Path:
    header: dict = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
        for i, name in enumerate(tensors)
    }
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4 * len(tensors)))
    return path


class CacheCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dd-cache-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.concept = self.tmp / "5_zkz"
        self.concept.mkdir()

    def image(self, name: str, when: float, caption_when: float | None = None) -> Path:
        path = self.concept / f"{name}.png"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(path)
        set_time(path, when)
        if caption_when is not None:
            cap = self.concept / f"{name}.txt"
            cap.write_text("1girl", encoding="utf-8")
            set_time(cap, caption_when)
        return path

    def cache(self, name: str, when: float) -> Path:
        path = self.concept / name
        path.write_bytes(b"cache")
        return set_time(path, when)

    def statuses(self) -> dict[str, str]:
        return {i.path.name: i.status for i in K.find_caches(self.tmp)}


class LatentNpzTests(CacheCase):
    def test_status_per_file(self) -> None:
        self.image("old_edit", NEW)
        self.cache("old_edit_0512x0512_anima.npz", OLD)  # image edited after caching
        self.image("fine", OLD)
        self.cache("fine_0512x0512_anima.npz", NEW)
        self.cache("gone_0512x0512_anima.npz", NEW)  # image deleted
        self.assertEqual(
            self.statuses(),
            {
                "old_edit_0512x0512_anima.npz": "stale_image",
                "fine_0512x0512_anima.npz": "fresh",
                "gone_0512x0512_anima.npz": "orphaned",
            },
        )

    def test_plain_stem_npz_from_older_trainers(self) -> None:
        self.image("a", NEW)
        self.cache("a.npz", OLD)
        self.assertEqual(self.statuses(), {"a.npz": "stale_image"})

    def test_a_couple_of_seconds_of_clock_noise_is_not_stale(self) -> None:
        self.image("a", OLD + 1.5)
        self.cache("a_0512x0512_anima.npz", OLD)
        self.assertEqual(self.statuses(), {"a_0512x0512_anima.npz": "fresh"})

    def test_longest_matching_image_name_owns_the_cache(self) -> None:
        self.image("a", OLD)
        self.image("a_b", NEW)
        self.cache("a_b_0512x0512_anima.npz", OLD)
        self.assertEqual(self.statuses(), {"a_b_0512x0512_anima.npz": "stale_image"})

    def test_a_different_image_with_a_longer_name_does_not_adopt_the_cache(self) -> None:
        self.image("img", OLD)
        self.cache("img_final_0512x0512_anima.npz", NEW)  # belongs to img_final, which is gone
        self.assertEqual(self.statuses(), {"img_final_0512x0512_anima.npz": "orphaned"})


class TextEncoderTests(CacheCase):
    def test_text_cache_goes_stale_when_the_caption_changes(self) -> None:
        self.image("a", OLD, caption_when=NEW)
        self.cache("a_anima_te.npz", OLD + 100)
        self.cache("a_0512x0512_anima.npz", OLD + 100)
        self.assertEqual(
            self.statuses(),
            {"a_anima_te.npz": "stale_caption", "a_0512x0512_anima.npz": "fresh"},
        )

    def test_text_cache_ignores_a_newer_image(self) -> None:
        self.image("a", NEW, caption_when=OLD)
        self.cache("a_anima_te.npz", OLD + 100)
        self.assertEqual(self.statuses(), {"a_anima_te.npz": "fresh"})

    def test_text_cache_without_caption_is_fresh(self) -> None:
        self.image("a", OLD)
        self.cache("a_anima_te.npz", NEW)
        self.assertEqual(self.statuses(), {"a_anima_te.npz": "fresh"})


class MusubiSafetensorsTests(CacheCase):
    def test_cache_is_recognised_by_its_header_not_its_name(self) -> None:
        self.image("e", NEW)
        write_safetensors(self.concept / "e_0512x0512_zi.safetensors", {"architecture": "z_image"})
        set_time(self.concept / "e_0512x0512_zi.safetensors", OLD)
        # Same name shape but no architecture in the header: not a cache.
        write_safetensors(self.concept / "e_0768x0768_zi.safetensors", {"note": "x"})
        write_safetensors(self.concept / "e_0640x0640_zi.safetensors", None)
        (self.concept / "e_0999x0999_zi.safetensors").write_bytes(b"not safetensors at all")
        self.assertEqual(self.statuses(), {"e_0512x0512_zi.safetensors": "stale_image"})

    def test_text_encoder_file_follows_the_caption(self) -> None:
        self.image("e", OLD, caption_when=NEW)
        te = write_safetensors(self.concept / "e_zi_te.safetensors", {"architecture": "z_image"})
        set_time(te, OLD + 50)
        self.assertEqual(self.statuses(), {"e_zi_te.safetensors": "stale_caption"})

    def test_lora_outputs_in_the_folder_are_not_caches(self) -> None:
        self.image("e", NEW)
        lora = write_safetensors(
            self.concept / "my_lora_0512x0512_zi.safetensors",
            {"ss_network_dim": "16", "modelspec.architecture": "anima/lora"},
        )
        set_time(lora, OLD)
        self.assertEqual(self.statuses(), {})

    def test_orphan(self) -> None:
        write_safetensors(self.concept / "x_0512x0512_zi.safetensors", {"architecture": "z_image"})
        self.assertEqual(self.statuses(), {"x_0512x0512_zi.safetensors": "orphaned"})


class FolderCacheTests(CacheCase):
    def folder(self, name: str, when: float) -> Path:
        folder = self.concept / name
        folder.mkdir()
        inner = folder / "abc123.safetensors"
        inner.write_bytes(b"x")
        set_time(inner, when)
        set_time(folder, when)
        return folder

    def test_folder_older_than_the_newest_image_is_stale(self) -> None:
        self.image("a", NEW)
        self.folder("_latent_cache", OLD)
        self.assertEqual(self.statuses(), {"_latent_cache": "stale_image"})

    def test_prefixed_folder_name_counts_and_fresh_folder_does_not(self) -> None:
        self.image("a", OLD)
        self.folder("zkz_latent_cache", NEW)
        self.assertEqual(self.statuses(), {"zkz_latent_cache": "fresh"})

    def test_folder_without_images_is_orphaned(self) -> None:
        self.folder("_latent_cache", NEW)
        self.assertEqual(self.statuses(), {"_latent_cache": "orphaned"})

    def test_other_folders_are_left_alone(self) -> None:
        self.image("a", OLD)
        (self.concept / "latents").mkdir()
        (self.concept / "latents" / "x.npz").write_bytes(b"x")
        set_time(self.concept / "latents" / "x.npz", NEW)
        self.assertEqual(self.statuses(), {"x.npz": "orphaned"})  # file rule still applies there


class QuarantineIsIgnored(CacheCase):
    def test_quarantined_caches_are_not_reported_again(self) -> None:
        quarantine = self.tmp / C.QUARANTINE_DIR_NAME
        quarantine.mkdir()
        (quarantine / "a_0512x0512_anima.npz").write_bytes(b"x")
        self.assertEqual(K.find_caches(self.tmp), [])


class AnalyzeTests(CacheCase):
    def test_issue_counts_each_kind(self) -> None:
        self.image("a", NEW, caption_when=NEW)
        self.cache("a_0512x0512_anima.npz", OLD)
        self.cache("a_anima_te.npz", OLD)
        self.cache("z_0512x0512_anima.npz", NEW)
        self.image("ok", OLD)
        self.cache("ok_0512x0512_anima.npz", NEW)
        issues = K.analyze_cache(self.tmp, recursive=True)
        self.assertEqual([i["code"] for i in issues], ["stale_cache"])
        issue = issues[0]
        self.assertEqual(issue["severity"], C.SEV_MEDIUM)
        self.assertIn("1 older than their image", issue["message"])
        self.assertIn("1 older than their caption", issue["message"])
        self.assertIn("1 orphaned", issue["message"])
        self.assertEqual(
            sorted(Path(p).name for p in issue["items"]),
            ["a_0512x0512_anima.npz", "a_anima_te.npz", "z_0512x0512_anima.npz"],
        )
        self.assertIn("quarantine-cache", issue["fix"])
        self.assertIn("old latent", issue["message"])

    def test_no_caches_no_issue(self) -> None:
        self.image("a", OLD)
        self.assertEqual(K.analyze_cache(self.tmp, True), [])

    def test_doctor_reports_stale_cache_as_warn(self) -> None:
        self.image("a", NEW)
        self.cache("a_0512x0512_anima.npz", OLD)
        report = doctor.run_doctor(self.tmp)
        self.assertIn("stale_cache", [i["code"] for i in report["scan"]["issues"]])
        self.assertEqual(report["verdict"], doctor.VERDICT_WARN)

    def test_non_recursive_only_looks_at_the_given_folder(self) -> None:
        self.image("a", NEW)
        self.cache("a_0512x0512_anima.npz", OLD)
        self.assertEqual(K.analyze_cache(self.tmp, recursive=False), [])
        self.assertEqual(len(K.analyze_cache(self.concept, recursive=False)), 1)


class QuarantineCacheTests(CacheCase):
    def build(self) -> dict[str, Path]:
        files = {"image": self.image("a", NEW, caption_when=NEW)}
        files["stale"] = self.cache("a_0512x0512_anima.npz", OLD)
        files["stale_te"] = self.cache("a_anima_te.npz", OLD)
        files["orphan"] = self.cache("z_0512x0512_anima.npz", NEW)
        self.image("ok", OLD)
        files["fresh"] = self.cache("ok_0512x0512_anima.npz", NEW)
        files["caption"] = self.concept / "a.txt"
        return files

    def test_dry_run_changes_nothing(self) -> None:
        files = self.build()
        result = X.cmd_quarantine_cache(self.tmp, apply=False)
        self.assertEqual(result["action_count"], 3)
        self.assertFalse(result["applied"])
        for path in files.values():
            self.assertTrue(path.exists())

    def test_apply_moves_only_stale_and_orphaned_caches(self) -> None:
        files = self.build()
        result = X.cmd_quarantine_cache(self.tmp, apply=True)
        self.assertEqual(result["moved_caches"], 3)
        quarantine = self.tmp / C.QUARANTINE_DIR_NAME
        self.assertEqual(
            sorted(p.name for p in quarantine.iterdir()),
            ["a_0512x0512_anima.npz", "a_anima_te.npz", "z_0512x0512_anima.npz"],
        )
        for key in ("image", "caption", "fresh"):
            self.assertTrue(files[key].exists(), key)
        for key in ("stale", "stale_te", "orphan"):
            self.assertFalse(files[key].exists(), key)

    def test_second_run_finds_nothing(self) -> None:
        self.build()
        X.cmd_quarantine_cache(self.tmp, apply=True)
        self.assertEqual(X.cmd_quarantine_cache(self.tmp, apply=False)["action_count"], 0)
        self.assertEqual(K.analyze_cache(self.tmp, True), [])

    def test_stale_cache_folder_is_moved_whole(self) -> None:
        self.image("a", NEW)
        folder = self.concept / "_latent_cache"
        folder.mkdir()
        (folder / "h.safetensors").write_bytes(b"x")
        set_time(folder / "h.safetensors", OLD)
        set_time(folder, OLD)
        X.cmd_quarantine_cache(self.tmp, apply=True)
        self.assertFalse(folder.exists())
        self.assertTrue(
            (self.tmp / C.QUARANTINE_DIR_NAME / "_latent_cache" / "h.safetensors").is_file()
        )
        self.assertTrue((self.concept / "a.png").is_file())

    def test_cli_is_wired_and_dry_run_by_default(self) -> None:
        files = self.build()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(F.main(["quarantine-cache", str(self.tmp)]), 0)
        self.assertTrue(files["stale"].exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(F.main(["quarantine-cache", str(self.tmp), "--apply"]), 0)
        self.assertFalse(files["stale"].exists())


if __name__ == "__main__":
    unittest.main()
