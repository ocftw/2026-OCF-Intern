import unittest
import itertools

from omnidocbench.core import (
    aggregate_supplementary,
    levenshtein,
    metrics_from_official_text_pairs,
    normalize_text,
    supplementary_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_normalization_is_nfc_newlines_and_edge_trim_only(self):
        self.assertEqual(normalize_text("  e\u0301\r\n a  \r"), "é\n a")

    def test_levenshtein(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("中文表格", "中文格式"), 2)
        self.assertEqual(levenshtein("", "abc"), 3)

    def test_bit_parallel_levenshtein_matches_reference_exhaustively(self):
        def reference(a, b):
            previous = list(range(len(b) + 1))
            for i, char_a in enumerate(a, 1):
                current = [i]
                for j, char_b in enumerate(b, 1):
                    current.append(
                        min(
                            current[-1] + 1,
                            previous[j] + 1,
                            previous[j - 1] + (char_a != char_b),
                        )
                    )
                previous = current
            return previous[-1]

        values = [""]
        for length in range(1, 5):
            values.extend("".join(chars) for chars in itertools.product("ab中", repeat=length))
        for left in values:
            for right in values:
                self.assertEqual(levenshtein(left, right), reference(left, right))

    def test_exact_and_directional_containment(self):
        exact = supplementary_metrics("abc", "abc")
        self.assertEqual(exact["em"], 1)
        self.assertEqual(exact["cm"], 1)
        contained = supplementary_metrics("abc", "xabcx")
        self.assertEqual(contained["em"], 0)
        self.assertEqual(contained["cm"], 1)
        reverse = supplementary_metrics("xabcx", "abc")
        self.assertEqual(reverse["cm"], 0)

    def test_anls_threshold(self):
        self.assertEqual(supplementary_metrics("abcd", "abxy")["anls"], 0.5)
        self.assertEqual(supplementary_metrics("abcd", "axyy")["anls"], 0.0)

    def test_character_f1_is_multiset(self):
        result = supplementary_metrics("aab", "abb")
        self.assertAlmostEqual(result["character_precision"], 2 / 3)
        self.assertAlmostEqual(result["character_recall"], 2 / 3)
        self.assertAlmostEqual(result["character_f1"], 2 / 3)

    def test_aggregate_has_denominators(self):
        rows = [supplementary_metrics("a", "a"), supplementary_metrics("b", "")]
        aggregate = aggregate_supplementary(rows)
        self.assertEqual(aggregate["denominator"], 2)
        self.assertEqual(aggregate["empty_count"], 1)
        self.assertIn("macro", aggregate)
        self.assertIn("micro", aggregate)

    def test_official_matcher_pair_adapter(self):
        rows, aggregate = metrics_from_official_text_pairs(
            [{"page_id": "p1", "gt": "abc", "pred": "abc"}, {"page_id": "p1", "gt_text": "x"}]
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1]["unmatched"])
        self.assertEqual(aggregate["method"], "official_v1_6_text_matching_pairs")
        self.assertEqual(aggregate["unmatched_count"], 1)

    def test_matcher_adapter_rejects_unknown_schema(self):
        with self.assertRaises(ValueError):
            metrics_from_official_text_pairs([{"reference": "x", "hypothesis": "x"}])


if __name__ == "__main__":
    unittest.main()
