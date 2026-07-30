import json
import tempfile
import unittest
from pathlib import Path

from tc_str_bench.ollama import OllamaClient


class CapturingClient(OllamaClient):
    def _json(self, path, payload=None, timeout=None):
        self.captured_path = path
        self.captured_payload = payload
        return 200, {
            "response": "甲",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }


class OllamaIntegrationTests(unittest.TestCase):
    def test_generate_schema_uses_top_level_think_and_common_options(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"image")
            options = {
                "num_ctx": 65536, "num_predict": 80, "temperature": 0,
                "repeat_penalty": 1.0, "top_k": 64, "top_p": 0.95, "think": False,
            }
            client = CapturingClient("http://127.0.0.1:11434", 5, 1, [])
            result = client.generate("exact:tag", "prompt", image, options)[0]
            self.assertTrue(result.success)
            self.assertEqual(client.captured_path, "/api/generate")
            self.assertFalse(client.captured_payload["think"])
            self.assertNotIn("think", client.captured_payload["options"])
            self.assertEqual(client.captured_payload["options"]["num_ctx"], 65536)
            self.assertEqual(client.captured_payload["model"], "exact:tag")
            self.assertEqual(len(client.captured_payload["images"]), 1)
            self.assertFalse(client.captured_payload["stream"])

    def test_incomplete_non_stream_response_is_retried_and_rejected(self):
        class IncompleteClient(CapturingClient):
            def _json(self, path, payload=None, timeout=None):
                return 200, {"response": "甲", "done": False}

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"image")
            client = IncompleteClient("http://127.0.0.1:11434", 5, 2, [0])
            results = client.generate(
                "exact:tag",
                "prompt",
                image,
                {"num_ctx": 65536, "num_predict": 80, "think": False},
            )
            self.assertEqual(len(results), 2)
            self.assertFalse(results[-1].success)
            self.assertEqual(results[-1].error_type, "incomplete_response")
            self.assertIn("done=False", results[-1].error_message)

    def test_glm_ocr_approved_policy_accepts_nonempty_response_and_records_limitations(self):
        class GlmOcrClient(CapturingClient):
            def _json(self, path, payload=None, timeout=None):
                return 200, {"response": "雜貨舗", "done": False}

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"image")
            client = GlmOcrClient("http://127.0.0.1:11434", 5, 2, [0])
            results = client.generate(
                "glm-ocr:bf16",
                "prompt",
                image,
                {"num_ctx": 65536, "num_predict": 80, "think": False},
                completion_policy={
                    "mode": "allow_nonempty_response_without_terminal_metadata",
                },
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            validation = results[0].completion_validation
            self.assertTrue(validation["accepted_by_exception"])
            self.assertFalse(validation["truncation_observable"])
            self.assertEqual(
                validation["missing_completion_fields"],
                ["done_reason", "prompt_eval_count", "eval_count"],
            )

    def test_glm_ocr_approved_policy_still_rejects_empty_response(self):
        class EmptyClient(CapturingClient):
            def _json(self, path, payload=None, timeout=None):
                return 200, {"response": "  ", "done": False}

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"image")
            client = EmptyClient("http://127.0.0.1:11434", 5, 1, [])
            result = client.generate(
                "glm-ocr:bf16",
                "prompt",
                image,
                {"num_ctx": 65536, "num_predict": 80, "think": False},
                completion_policy={
                    "mode": "allow_nonempty_response_without_terminal_metadata",
                },
            )[0]
            self.assertFalse(result.success)
            self.assertEqual(result.error_type, "incomplete_response")

    def test_strict_policy_retries_completed_but_empty_response(self):
        class EmptyThenValidClient(CapturingClient):
            calls = 0

            def _json(self, path, payload=None, timeout=None):
                self.calls += 1
                return 200, {
                    "response": "" if self.calls == 1 else "甲",
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"image")
            client = EmptyThenValidClient("http://127.0.0.1:11434", 5, 2, [0])
            results = client.generate(
                "exact:tag",
                "prompt",
                image,
                {"num_ctx": 65536, "num_predict": 80, "think": False},
            )
            self.assertEqual(len(results), 2)
            self.assertFalse(results[0].success)
            self.assertEqual(results[0].error_type, "incomplete_response")
            self.assertTrue(results[1].success)


if __name__ == "__main__":
    unittest.main()
