import unittest

from omnidocbench.core import detect_anomalies, postprocess_response


class PostprocessTests(unittest.TestCase):
    def test_preserves_html_json_markdown_and_repetition(self):
        raw = '  <table><tr><td>1</td></tr></table>\n{"x": 1}\naaaa  '
        value, info = postprocess_response({"response": raw})
        self.assertEqual(value, '<table><tr><td>1</td></tr></table>\n{"x": 1}\naaaa')
        self.assertTrue(info["modified"])

    def test_only_exact_empty_thought_wrapper_is_removed(self):
        value, _ = postprocess_response({"response": "<think></think>\n# Title"})
        self.assertEqual(value, "# Title")
        value, _ = postprocess_response({"response": "<think>reasoning</think>\n# Title"})
        self.assertEqual(value, "<think>reasoning</think>\n# Title")

    def test_truncation(self):
        flags = detect_anomalies("x", {"eval_count": 8192}, "prompt", 8192)
        self.assertTrue(flags["truncated"])


if __name__ == "__main__":
    unittest.main()
