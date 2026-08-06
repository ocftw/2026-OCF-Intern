from __future__ import annotations

import importlib.util
import json
import pathlib
import sqlite3
import tempfile
import unittest


TOOL_PATH = pathlib.Path(__file__).parents[1] / "tools" / "evaluate_model.py"
SPEC = importlib.util.spec_from_file_location("evaluate_model_tool", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)

STATUS_TOOL_PATH = pathlib.Path(__file__).parents[1] / "tools" / "evaluation_status.py"
STATUS_SPEC = importlib.util.spec_from_file_location(
    "evaluation_status_tool", STATUS_TOOL_PATH
)
status_tool = importlib.util.module_from_spec(STATUS_SPEC)
assert STATUS_SPEC.loader is not None
STATUS_SPEC.loader.exec_module(status_tool)


class SingleModelEvaluatorToolTests(unittest.TestCase):
    def test_progress_parser_uses_latest_tqdm_counter(self):
        log = (
            "\rMatching pages: 0%| | 1/1651 [00:02<1:00:00, 2.00s/it]"
            "\rMatching pages: 12%|# | 200/1651 [03:00<20:00, 1.20it/s]"
        )
        progress = status_tool.parse_progress(log)
        self.assertIsNotNone(progress)
        self.assertEqual(progress["stage"], "Matching pages")
        self.assertEqual(progress["current"], 200)
        self.assertEqual(progress["total"], 1651)
        self.assertEqual(progress["percent"], 12)

    def test_failed_page_requires_explicit_empty_policy(self):
        rows = [
            {
                "filename": "page.png",
                "status": "FAILED",
                "attempts": 3,
                "error_history_json": '[{"error":"done=false"}]',
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "--failed-as-empty"):
            tool.prepare_entries(rows, failed_as_empty=False)
        entries, counts = tool.prepare_entries(rows, failed_as_empty=True)
        self.assertEqual(counts, {"SUCCESS": 0, "FAILED_AS_EMPTY": 1})
        self.assertEqual(entries[0]["evaluation_representation"], "EMPTY_PREDICTION")
        self.assertEqual(
            entries[0]["prediction_sha256"],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

    def test_eval_timeout_page_kept_as_success_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "one.md"
            source.write_text("content", encoding="utf-8")
            rows = [
                {
                    "filename": "page.png",
                    "status": "SUCCESS",
                    "prediction_path": str(source),
                    "prediction_sha256": tool.sha256_file(source),
                    "attempts": 1,
                }
            ]
            eval_timeout_pages = {
                "page.png": {"reason": "known hang", "recorded_at": "2026-07-31T00:00:00Z"}
            }
            entries, counts = tool.prepare_entries(
                rows,
                failed_as_empty=False,
                eval_timeout_pages=eval_timeout_pages,
                apply_eval_timeout_as_empty=False,
            )
            self.assertEqual(counts, {"SUCCESS": 1, "FAILED_AS_EMPTY": 0})
            self.assertEqual(entries[0]["checkpoint_status"], "SUCCESS")
            self.assertEqual(entries[0]["source_path"], str(source))

            entries, counts = tool.prepare_entries(
                rows,
                failed_as_empty=False,
                eval_timeout_pages=eval_timeout_pages,
                apply_eval_timeout_as_empty=True,
            )
            self.assertEqual(
                counts, {"SUCCESS": 0, "FAILED_AS_EMPTY": 0, "EVAL_TIMEOUT_AS_EMPTY": 1}
            )
            self.assertEqual(
                entries[0]["evaluation_representation"], "EMPTY_PREDICTION_EVAL_TIMEOUT"
            )
            self.assertNotIn("source_path", entries[0])
            self.assertEqual(
                entries[0]["prediction_sha256"],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            )
            self.assertEqual(entries[0]["original_prediction_sha256"], tool.sha256_file(source))

    def test_split_into_batches_covers_every_page_without_overlap(self):
        pages = [{"id": i} for i in range(7)]
        batches = tool.split_into_batches(pages, 3)
        self.assertEqual([len(b) for b in batches], [3, 3, 1])
        flattened = [item for batch in batches for item in batch]
        self.assertEqual(flattened, pages)

    def test_split_into_batches_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            tool.split_into_batches([{"id": 1}], 0)

    def test_batches_root_dir_changes_with_batch_size(self):
        base = pathlib.Path("/tmp/model_base")
        first = tool.batches_root_dir(base, 50)
        second = tool.batches_root_dir(base, 100)
        self.assertNotEqual(first, second)
        self.assertTrue(str(first).endswith("bs50"))

    def test_batches_root_dir_stable_across_unrelated_content_changes(self):
        # Deliberately NOT keyed by the corpus-wide input hash: registering a
        # newly-discovered eval-timeout page anywhere must not move where an
        # unrelated batch's already-completed work lives on disk.
        base = pathlib.Path("/tmp/model_base")
        self.assertEqual(tool.batches_root_dir(base, 100), tool.batches_root_dir(base, 100))

    def test_batch_content_hash_changes_only_for_affected_pages(self):
        pages = [
            {"page_info": {"image_path": "a.png"}},
            {"page_info": {"image_path": "b.png"}},
        ]
        entries_before = {
            "a.png": {"filename": "a.png", "prediction_filename": "a.md",
                      "checkpoint_status": "SUCCESS", "prediction_sha256": "hash-a"},
            "b.png": {"filename": "b.png", "prediction_filename": "b.md",
                      "checkpoint_status": "SUCCESS", "prediction_sha256": "hash-b"},
        }
        batch_a = [pages[0]]
        batch_b = [pages[1]]
        hash_a_before = tool.batch_content_hash(batch_a, entries_before)
        hash_b_before = tool.batch_content_hash(batch_b, entries_before)

        # Simulate marking page b.png as an eval-timeout page: only its own
        # entry changes.
        entries_after = dict(entries_before)
        entries_after["b.png"] = {
            **entries_before["b.png"],
            "evaluation_representation": "EMPTY_PREDICTION_EVAL_TIMEOUT",
            "prediction_sha256": "empty-hash",
        }
        hash_a_after = tool.batch_content_hash(batch_a, entries_after)
        hash_b_after = tool.batch_content_hash(batch_b, entries_after)

        self.assertEqual(hash_a_before, hash_a_after)
        self.assertNotEqual(hash_b_before, hash_b_after)

    def test_batch_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            batches_root = pathlib.Path(directory)
            state = {"schema_version": 1, "batches": [{"index": 0, "status": "pending"}]}
            self.assertIsNone(tool.load_batch_state(batches_root))
            tool.save_batch_state(batches_root, state)
            self.assertEqual(tool.load_batch_state(batches_root), state)

    def test_find_batch_element_result_locates_unique_file(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = pathlib.Path(directory)
            (result_dir / "predictions_quick_match_text_block_result.json").write_text("[]")
            (result_dir / "predictions_quick_match_table_result.json").write_text("[]")
            found = tool.find_batch_element_result(result_dir, "text_block")
            self.assertEqual(found.name, "predictions_quick_match_text_block_result.json")

    def test_find_batch_element_result_missing_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                tool.find_batch_element_result(pathlib.Path(directory), "table")

    def test_terminal_coverage_must_equal_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            database = root / "results.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE run(signature TEXT);
                INSERT INTO run VALUES ('signature');
                CREATE TABLE page_result(
                  signature TEXT, model_id TEXT, page_id TEXT, filename TEXT,
                  status TEXT, prediction_path TEXT, prediction_sha256 TEXT,
                  attempts INTEGER, error_history_json TEXT
                );
                INSERT INTO page_result VALUES(
                  'signature','model','page-1','one.png','FAILED',NULL,NULL,3,'[]'
                );
                """
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(RuntimeError, "terminal coverage"):
                tool.load_model_rows(root, "model", {"one.png", "two.png"})

    def test_staging_is_pure_markdown_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "canonical.md"
            source.write_text("模型輸出", encoding="utf-8")
            entries = [
                {
                    "filename": "one.png",
                    "prediction_filename": "one.md",
                    "checkpoint_status": "SUCCESS",
                    "source_path": str(source),
                    "prediction_sha256": tool.sha256_file(source),
                    "attempts": 1,
                },
                {
                    "filename": "two.png",
                    "prediction_filename": "two.md",
                    "checkpoint_status": "FAILED",
                    "evaluation_representation": "EMPTY_PREDICTION",
                    "prediction_sha256": tool.hashlib.sha256(b"").hexdigest(),
                    "attempts": 3,
                    "error_history": [],
                },
            ]
            staging, _ = tool.validate_or_create_staging(root, entries, "a" * 64)
            self.assertEqual({path.name for path in staging.iterdir()}, {"one.md", "two.md"})
            self.assertEqual((staging / "one.md").read_text(encoding="utf-8"), "模型輸出")
            self.assertEqual((staging / "two.md").read_bytes(), b"")
            self.assertEqual(source.read_text(encoding="utf-8"), "模型輸出")

    def test_load_eval_timeout_pages_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            model_base = pathlib.Path(directory) / "model_base"
            self.assertEqual(tool.load_eval_timeout_pages(model_base), {})

    def test_load_eval_timeout_pages_indexes_by_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            model_base = pathlib.Path(directory)
            payload = {
                "schema_version": 1,
                "pages": [
                    {"filename": "page-a.png", "reason": "hang", "recorded_at": "t"},
                    {"filename": "page-b.png", "reason": "hang2", "recorded_at": "t2"},
                ],
            }
            tool.eval_timeout_pages_path(model_base).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            loaded = tool.load_eval_timeout_pages(model_base)
            self.assertEqual(set(loaded), {"page-a.png", "page-b.png"})
            self.assertEqual(loaded["page-a.png"]["reason"], "hang")

    def test_unknown_model_is_rejected_before_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            run = root / "run"
            immutable = run / "immutable_config"
            immutable.mkdir(parents=True)
            benchmark = {"benchmark": {}, "paths": {}, "evaluator": {}}
            models = {"models": [{"id": "known"}]}
            (immutable / "benchmark.json").write_text(json.dumps(benchmark))
            (immutable / "models.yaml").write_text(json.dumps(models))
            hashes = {
                "benchmark.json": tool.sha256_file(immutable / "benchmark.json"),
                "models.yaml": tool.sha256_file(immutable / "models.yaml"),
            }
            (immutable / "hashes.json").write_text(json.dumps(hashes))
            args = tool.build_parser().parse_args(
                [
                    "--output-root",
                    str(root),
                    "--run-id",
                    "run",
                    "--model",
                    "unknown",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "unknown model id"):
                tool.run_evaluation(args)


if __name__ == "__main__":
    unittest.main()
