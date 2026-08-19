import pytest

from ocf_benchmark.scorers.tc_str import aggregate, normalize_minimal, score


def test_exact_cer_anls_containment_f1():
    metrics = score("甲乙", "甲丙")
    assert metrics["exact"] == 0
    assert metrics["edit_distance"] == 1
    assert metrics["anls"] == 0.5
    assert metrics["containment"] == 0
    assert metrics["char_f1"] == pytest.approx(0.5)


def test_micro_cer_not_mean_of_sample_cer():
    rows = [
        {"metrics": score("甲", "甲")},
        {"metrics": score("", "甲乙乙")},
    ]
    assert aggregate(rows)["cer_micro"] == pytest.approx(3 / 4)


def test_nfc_crlf_and_boundary_whitespace_only():
    assert normalize_minimal("  e\u0301\r\n") == "é"
    assert normalize_minimal("甲 乙") == "甲 乙"


def test_main_scorer_preserves_punctuation_and_repetition():
    assert score("哈哈！", "哈！")["exact"] == 0
    assert score("甲，乙", "甲乙")["exact"] == 0
    assert normalize_minimal("哈哈哈") == "哈哈哈"


def test_raw_exact_separate_from_minimal_exact():
    metrics = score(" 甲\r\n", "甲")
    assert metrics["exact"] == 1
    assert metrics["raw_exact"] == 0


def test_legacy_is_separate(suite_dir):
    from ocf_benchmark.scorers.tc_str import LegacyPostprocessor

    legacy = LegacyPostprocessor(suite_dir.parent / "ablation_experiment/postprocess.py")
    raw = "哈哈哈哈"
    assert normalize_minimal(raw) == raw
    assert legacy.clean(raw) == "哈"
    assert len(legacy.sha256) == 64
