from ocf_benchmark.scorers.vistw_mcq import parse_answer, score


def test_fullwidth_answer():
    assert parse_answer("分析\n答案：Ｃ") == "C"


def test_last_legal_marker_wins():
    assert parse_answer("答案: A\n修正分析\n答案：D") == "D"


def test_last_single_line_fallback():
    assert parse_answer("A 可能，B 也可能\nＢ") == "B"


def test_does_not_guess_from_explanation():
    result = score("我在 A 與 B 間選擇，應該是 C 選項。", "C")
    assert result["parsed_answer"] is None
    assert result["valid"] is False
    assert result["correct"] is False
