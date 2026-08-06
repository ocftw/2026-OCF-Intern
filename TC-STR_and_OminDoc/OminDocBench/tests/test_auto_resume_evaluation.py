from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


TOOL_PATH = pathlib.Path(__file__).parents[1] / "tools" / "auto_resume_evaluation.py"
SPEC = importlib.util.spec_from_file_location("auto_resume_evaluation_tool", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)


class AutoResumeSignatureTests(unittest.TestCase):
    def test_signature_page_is_the_last_unresolved_match_timeout(self):
        log = (
            "[match-timeout] page-a.png: quick_match exceeded 420s, fallback to chunked Hungarian\n"
            "[timeout-fallback] page-a.png: gt=5 pred=10 chunk=3\n"
            "[match-timeout] page-b.png: quick_match exceeded 420s, fallback to chunked Hungarian\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = pathlib.Path(directory) / "docker_stdout.log"
            log_path.write_text(log, encoding="utf-8")
            self.assertEqual(tool.find_stalled_signature_page(log_path), "page-b.png")

    def test_no_signature_when_fallback_resolved_the_last_timeout(self):
        log = (
            "[match-timeout] page-a.png: quick_match exceeded 420s, fallback to chunked Hungarian\n"
            "[timeout-fallback] page-a.png: gt=5 pred=10 chunk=3\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = pathlib.Path(directory) / "docker_stdout.log"
            log_path.write_text(log, encoding="utf-8")
            self.assertIsNone(tool.find_stalled_signature_page(log_path))

    def test_no_signature_without_any_match_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = pathlib.Path(directory) / "docker_stdout.log"
            log_path.write_text("Matching pages: 5/100\n", encoding="utf-8")
            self.assertIsNone(tool.find_stalled_signature_page(log_path))

    def test_missing_log_file_returns_none(self):
        self.assertIsNone(tool.find_stalled_signature_page(pathlib.Path("/nonexistent/log.txt")))


class DegenerateRepetitionTests(unittest.TestCase):
    def test_flags_many_repeated_short_lines(self):
        text = "\n".join(["$\\Delta = ax^2 + bx + c$"] * 200 + ["something else"] * 5)
        verdict = tool.is_degenerate_repetition(text)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["method"], "line_repetition")

    def test_flags_repeated_phrase_crammed_into_few_lines(self):
        # Same shape as the page-59 hang: almost no newlines, one phrase
        # repeated many times inside otherwise-long lines.
        phrase = "Let pi blows up the points L1 L2 L3 L4 and pi_i the strict transforms. "
        text = phrase * 400
        verdict = tool.is_degenerate_repetition(text)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["method"], "ngram_repeat_count")

    def test_normal_prose_is_not_flagged(self):
        text = "\n".join(
            [
                "Section 2.1 introduces the boundary conditions for the diffusion equation.",
                "We first derive the weak formulation before discretizing with finite elements.",
                "The resulting linear system is solved using a conjugate gradient method.",
                "Convergence is verified against a manufactured solution with known error bounds.",
            ]
            * 3
        )
        self.assertIsNone(tool.is_degenerate_repetition(text))

    def test_short_text_is_not_flagged(self):
        self.assertIsNone(tool.is_degenerate_repetition("short answer: 42"))


class FindPredictionFileTests(unittest.TestCase):
    def test_locates_prediction_in_newest_staging_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            model_base = pathlib.Path(directory)
            older = model_base / "official_predictions_full_1651_aaa111111111"
            newer = model_base / "official_predictions_full_1651_bbb222222222"
            older.mkdir()
            newer.mkdir()
            (older / "page-x.md").write_text("old", encoding="utf-8")
            (newer / "page-x.md").write_text("new", encoding="utf-8")
            import os
            import time

            os.utime(older, (time.time() - 100, time.time() - 100))
            found = tool.find_prediction_file(model_base, "page-x.png")
            self.assertEqual(found.read_text(encoding="utf-8"), "new")

    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(tool.find_prediction_file(pathlib.Path(directory), "page-x.png"))


if __name__ == "__main__":
    unittest.main()
