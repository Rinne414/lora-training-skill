"""Subject-count conflicts: check_subjects.py, the doctor warning, and fix-subject-count."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import _common as C  # noqa: E402
import check_subjects as S  # noqa: E402
import doctor  # noqa: E402
import fix_dataset as F  # noqa: E402
import fix_extras as X  # noqa: E402
from PIL import Image  # noqa: E402


NO_TRAINER = ["--trainer-url", "http://127.0.0.1:9"]  # never ask a real trainer from a test


def kept(tags: list[str]) -> list[str]:
    return S.resolve_tags(tags)[0]


class ResolveTests(unittest.TestCase):
    def test_one_girl_with_two_girls_keeps_the_higher_count(self) -> None:
        self.assertEqual(kept(["1girl", "2girls", "red hair"]), ["2girls", "red hair"])

    def test_one_girl_with_multiple_girls(self) -> None:
        self.assertEqual(kept(["1girl", "multiple girls", "smile"]), ["multiple girls", "smile"])
        self.assertEqual(kept(["1girl", "multiple_girls"]), ["multiple_girls"])

    def test_boys_and_others_follow_the_same_rule(self) -> None:
        self.assertEqual(kept(["1boy", "2boys"]), ["2boys"])
        self.assertEqual(kept(["1boy", "multiple boys"]), ["multiple boys"])
        self.assertEqual(kept(["1other", "2others"]), ["2others"])

    def test_two_different_counts_keep_the_highest(self) -> None:
        self.assertEqual(kept(["2girls", "3girls", "1girl"]), ["3girls"])
        self.assertEqual(kept(["6+girls", "5girls"]), ["6+girls"])

    def test_solo_is_dropped_when_there_are_several_subjects(self) -> None:
        self.assertEqual(kept(["solo", "2girls", "smile"]), ["2girls", "smile"])
        self.assertEqual(kept(["solo", "multiple girls"]), ["multiple girls"])
        self.assertEqual(kept(["solo", "1girl", "1boy"]), ["1girl", "1boy"])

    def test_conflict_and_solo_together(self) -> None:
        self.assertEqual(kept(["1girl", "2girls", "solo"]), ["2girls"])

    def test_consistent_captions_are_untouched(self) -> None:
        for tags in (
            ["1girl", "solo", "smile"],
            ["2girls", "multiple girls"],
            ["1girl", "1boy", "hug"],
            ["solo", "solo focus", "1girl"],
            ["solo_focus", "2girls"],
            ["2girls", "solo focus"],
            ["red hair", "solo"],
            [],
        ):
            with self.subTest(tags=tags):
                new_tags, removed = S.resolve_tags(tags)
                self.assertEqual(new_tags, tags)
                self.assertEqual(removed, [])

    def test_solo_focus_is_never_treated_as_solo(self) -> None:
        self.assertEqual(kept(["solo focus", "2girls", "1girl"]), ["solo focus", "2girls"])

    def test_removed_tags_come_with_a_reason(self) -> None:
        _, removed = S.resolve_tags(["1girl", "2girls", "solo"])
        self.assertEqual([tag for tag, _ in removed], ["1girl", "solo"])
        self.assertIn("2girls", removed[0][1])
        self.assertIn("solo", removed[1][1])

    def test_case_and_spacing_are_normalised_for_matching_only(self) -> None:
        self.assertEqual(kept(["1Girl", "2GIRLS"]), ["2GIRLS"])


class JsonTests(unittest.TestCase):
    def test_count_field_and_lists_are_resolved_together(self) -> None:
        obj = {"count": "1girl, solo", "appearance": ["2girls", "red hair"], "tags": ["smile"]}
        new, removed = S.resolve_json(obj)
        self.assertNotIn("count", new)  # nothing left of it, so the key goes
        self.assertEqual(new["appearance"], ["2girls", "red hair"])
        self.assertEqual({tag for tag, _ in removed}, {"1girl", "solo"})


def write_png(path: Path) -> None:
    Image.new("RGB", (64, 64), (10, 20, 30)).save(path)


class DatasetCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dd-subj-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.concept = self.tmp / "5_zkz"
        self.concept.mkdir()

    def add(self, name: str, caption: str) -> Path:
        write_png(self.concept / f"{name}.png")
        path = self.concept / f"{name}.txt"
        path.write_text(caption, encoding="utf-8")
        return path


class AnalyzeTests(DatasetCase):
    def test_conflicting_captions_become_one_medium_issue(self) -> None:
        self.add("a", "1girl, 2girls, red hair")
        self.add("b", "solo, 1girl, smile")
        self.add("c", "solo, multiple girls")
        issues = S.analyze_subjects(self.tmp, recursive=True, prefer_json=False)
        self.assertEqual([i["code"] for i in issues], ["subject_count_conflict"])
        issue = issues[0]
        self.assertEqual(issue["severity"], C.SEV_MEDIUM)
        self.assertIn("2 caption", issue["message"])
        self.assertEqual(sorted(Path(p).name for p in issue["items"]), ["a.txt", "c.txt"])
        self.assertIn("fix-subject-count", issue["fix"])
        self.assertEqual(len(issue["details"]), 2)

    def test_clean_dataset_has_no_issue(self) -> None:
        self.add("a", "1girl, solo, smile")
        self.add("b", "2girls, multiple girls")
        self.assertEqual(S.analyze_subjects(self.tmp, True, False), [])

    def test_doctor_reports_it_as_warn(self) -> None:
        self.add("a", "1girl, 2girls, red hair")
        report = doctor.run_doctor(self.tmp)
        codes = [i["code"] for i in report["captions"]["issues"]]
        self.assertIn("subject_count_conflict", codes)
        self.assertEqual(report["verdict"], doctor.VERDICT_WARN)
        self.assertTrue(any("fix-subject-count" in r["action"] for r in report["recommendations"]))

    def test_quarantined_captions_are_not_looked_at(self) -> None:
        quarantine = self.concept / C.QUARANTINE_DIR_NAME
        quarantine.mkdir()
        write_png(quarantine / "q.png")
        (quarantine / "q.txt").write_text("1girl, 2girls", encoding="utf-8")
        self.assertEqual(S.analyze_subjects(self.tmp, True, False), [])


class FixTests(DatasetCase):
    def test_dry_run_changes_nothing(self) -> None:
        path = self.add("a", "1girl, 2girls, solo, red hair. A sentence about two friends.")
        before = path.read_text(encoding="utf-8")
        result = X.cmd_fix_subject_count(self.tmp, apply=False)
        self.assertFalse(result["applied"])
        self.assertEqual(result["rewritten_captions"], 1)
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertIn("1girl", result["actions"][0]["detail"])
        self.assertIn("solo", result["actions"][0]["detail"])

    def test_apply_keeps_the_higher_count_and_the_prose(self) -> None:
        path = self.add("a", "1girl, 2girls, solo, red hair. A sentence about two friends.")
        other = self.add("b", "1girl, solo, smile")
        untouched = other.read_text(encoding="utf-8")
        result = X.cmd_fix_subject_count(self.tmp, apply=True)
        self.assertEqual(
            path.read_text(encoding="utf-8"), "2girls, red hair. A sentence about two friends."
        )
        self.assertEqual(other.read_text(encoding="utf-8"), untouched)
        self.assertEqual(result["rewritten_captions"], 1)
        self.assertIn("text-encoder", result["after_apply_hint"])

    def test_after_the_fix_the_doctor_is_quiet(self) -> None:
        self.add("a", "1girl, 2girls, solo")
        X.cmd_fix_subject_count(self.tmp, apply=True)
        self.assertEqual(S.analyze_subjects(self.tmp, True, False), [])

    def test_images_are_never_touched(self) -> None:
        self.add("a", "1girl, multiple girls")
        X.cmd_fix_subject_count(self.tmp, apply=True)
        self.assertTrue((self.concept / "a.png").is_file())

    def test_json_caption_is_fixed(self) -> None:
        write_png(self.concept / "j.png")
        obj = {"quality": "masterpiece", "count": "1girl, 2girls", "tags": ["solo", "smile"]}
        path = self.concept / "j.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        X.cmd_fix_subject_count(self.tmp, apply=True)
        fixed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(fixed["count"], "2girls")
        self.assertEqual(fixed["tags"], ["smile"])

    def test_cli_is_wired_and_dry_run_by_default(self) -> None:
        path = self.add("a", "1girl, 2girls")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(F.main(["fix-subject-count", str(self.tmp)]), 0)
        self.assertEqual(path.read_text(encoding="utf-8"), "1girl, 2girls")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                F.main(["fix-subject-count", str(self.tmp), "--apply", *NO_TRAINER]), 0
            )
        self.assertEqual(path.read_text(encoding="utf-8"), "2girls")


if __name__ == "__main__":
    unittest.main()
