import pathlib
import tempfile
import unittest

from omnidocbench.core import CheckpointDB, Settings, sha256_file


class CheckpointTests(unittest.TestCase):
    def test_resume_requires_success_files_hash_and_signature(self):
        settings = Settings.load()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            raw = root / "raw.json"
            pred = root / "page.md"
            raw.write_text("{}", encoding="utf-8")
            pred.write_text("prediction", encoding="utf-8")
            db = CheckpointDB(root / "results.sqlite")
            db.ensure_run(settings, "TEST")
            db.save_result(
                settings,
                {
                    "model_id": "m", "page_id": "p", "filename": "p.png", "status": "SUCCESS",
                    "image_sha256": "image", "raw_path": str(raw), "raw_sha256": sha256_file(raw),
                    "prediction_path": str(pred), "prediction_sha256": sha256_file(pred),
                    "attempts": 1, "error_history_json": "[]",
                },
            )
            self.assertTrue(db.reusable(settings, "m", "p", "image"))
            pred.write_text("changed", encoding="utf-8")
            self.assertFalse(db.reusable(settings, "m", "p", "image"))
            self.assertFalse(db.reusable(settings, "m", "p", "different-image"))
            db.close()


if __name__ == "__main__":
    unittest.main()
