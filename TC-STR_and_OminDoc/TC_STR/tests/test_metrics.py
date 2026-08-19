import unittest

from tc_str_bench.metrics import aligned_supplementary, normalize_primary, score


class MetricsTests(unittest.TestCase):
    def test_nfc_and_newlines_and_outer_whitespace(self):
        self.assertEqual(normalize_primary("  e\u0301\r\n甲\r "), "é\n甲")
        self.assertEqual(normalize_primary("甲 乙"), "甲 乙")

    def test_traditional_english_digits_punctuation(self):
        self.assertEqual(score("繁中ABC123！", "繁中ABC123！")["em"], 1)
        self.assertEqual(score("繁中ABC123", "繁中ABC123！")["em"], 0)

    def test_cm_is_one_way(self):
        self.assertEqual(score("前甲乙後", "甲乙")["cm"], 1)
        self.assertEqual(score("甲", "甲乙")["cm"], 0)

    def test_empty(self):
        self.assertEqual(score("", ""), {"em": 1.0, "cm": 1.0, "anls": 1.0, "f1": 1.0})
        self.assertEqual(score("", "甲")["f1"], 0)

    def test_anls_threshold(self):
        self.assertEqual(score("甲乙", "甲丙")["anls"], 0.5)
        self.assertEqual(score("甲", "乙丙")["anls"], 0)

    def test_multiset_repetition(self):
        self.assertAlmostEqual(score("哈哈", "哈")["f1"], 2 / 3)

    def test_aligned_is_separate_and_records_dangerous_changes(self):
        primary = normalize_primary("哈哈哈哈哈")
        aligned, changes = aligned_supplementary("哈哈哈哈哈")
        self.assertEqual(primary, "哈哈哈哈哈")
        self.assertEqual(aligned, "哈")
        self.assertIn("compress_4plus_repeated_character", changes)
        long_text = "".join(chr(0x4E00 + (index % 200)) for index in range(201))
        aligned, changes = aligned_supplementary(long_text)
        self.assertEqual(len(aligned), 200)
        self.assertIn("truncate_200_characters", changes)

    def test_aligned_json_and_html(self):
        aligned, changes = aligned_supplementary('[{"text":"甲"},{"content":"乙"}]')
        self.assertEqual(aligned, "甲乙")
        self.assertIn("extract_json_array", changes)
        aligned, _ = aligned_supplementary("<p>甲</p>")
        self.assertEqual(aligned, "甲")


if __name__ == "__main__":
    unittest.main()
