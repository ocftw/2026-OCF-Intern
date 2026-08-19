import json
import pathlib
import tempfile
import unittest
from unittest import mock

from omnidocbench.core import Settings, prediction_filename, validate_dataset


class DatasetTests(unittest.TestCase):
    def test_prediction_filename_replaces_only_final_extension(self):
        self.assertEqual(prediction_filename("document.pdf_1.jpg"), "document.pdf_1.md")

    def test_validator_rejects_wrong_page_count_and_missing_image(self):
        settings = Settings.load()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            gt = root / "OmniDocBench.json"
            gt.write_text(json.dumps([{"page_info": {"image_path": "x.png"}}]), encoding="utf-8")
            altered = json.loads(json.dumps(settings.raw))
            altered["paths"]["dataset_root"] = str(root)
            altered["benchmark"]["ground_truth_sha256"] = "not-the-hash"
            fake = mock.Mock(raw=altered)
            fake.gt_path = gt
            fake.image_dir = root / "images"
            report, errors = validate_dataset(fake)
            self.assertEqual(report["page_count"], 1)
            self.assertTrue(any("requires 1651" in error for error in errors))
            self.assertTrue(any("missing official page images" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
