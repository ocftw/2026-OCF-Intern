import tempfile
import unittest
from pathlib import Path

from tc_str_bench.storage import Checkpoint


class StorageTests(unittest.TestCase):
    def test_attempt_history_and_signature_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Checkpoint(Path(directory) / "results.sqlite")
            base = {
                "run_signature": "sig-a", "phase": "smoke", "model_id": "m",
                "sample_index": 7, "attempt_number": 1, "started_at": "a", "ended_at": "b",
                "success": False, "request": {"x": 1}, "response": None, "http_status": None,
                "error_type": "timeout", "error_message": "x", "latency_seconds": 1.0,
                "completion_validation": {"policy_mode": "strict"},
            }
            db.record_attempt(base)
            self.assertFalse(db.successful("sig-a", "smoke", "m", 7))
            self.assertEqual(db.next_attempt("sig-a", "smoke", "m", 7), 2)
            latest = db.latest_attempts("sig-a", "smoke")
            self.assertEqual(len(latest), 1)
            self.assertEqual(latest[0]["request"], {"x": 1})
            self.assertEqual(latest[0]["error_type"], "timeout")
            self.assertEqual(latest[0]["completion_validation"]["policy_mode"], "strict")
            row = {
                "run_signature": "sig-a", "phase": "smoke", "model_id": "m",
                "sample_index": 7, "exact_tag": "m:exact", "model_digest": "digest",
                "image_relative_path": "images/x.png", "image_sha256": "hash", "ground_truth": "甲",
                "prediction_raw": "甲", "prediction_primary": "甲", "prediction_aligned": "甲",
                "aligned_changes": [], "primary_metrics": {"em":1,"cm":1,"anls":1,"f1":1},
                "aligned_metrics": {"em":1,"cm":1,"anls":1,"f1":1},
                "anomaly_flags": {"empty":False}, "ollama_metadata": {},
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1,
                "completion_validation": {
                    "policy_mode": "strict",
                    "truncation_observable": True,
                },
                "record_status": "completed_with_anomaly",
                "http_status": 200,
                "error_type": "incomplete_response",
                "error_message": "metadata missing",
                "latency_seconds": 0.5, "attempt_count": 2,
            }
            db.record_success(row)
            self.assertTrue(db.successful("sig-a", "smoke", "m", 7))
            self.assertFalse(db.successful("sig-b", "smoke", "m", 7))
            db.close()
            reopened = Checkpoint(Path(directory) / "results.sqlite")
            self.assertTrue(reopened.successful("sig-a", "smoke", "m", 7))
            stored = reopened.results("sig-a", "smoke")[0]
            self.assertTrue(stored["completion_validation"]["truncation_observable"])
            self.assertEqual(stored["record_status"], "completed_with_anomaly")
            self.assertEqual(stored["http_status"], 200)
            self.assertEqual(stored["error_type"], "incomplete_response")
            reopened.close()


if __name__ == "__main__":
    unittest.main()
