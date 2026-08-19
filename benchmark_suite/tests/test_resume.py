import json

from ocf_benchmark.resume import JsonlCheckpoint, ResumeKey, completed_keys, find_resumable_run


def record(config="a", status="completed"):
    return {
        "config_hash": config,
        "model_logical_id": "m",
        "resolved_model_digest": "d",
        "benchmark": "b",
        "benchmark_revision": "r",
        "sample_id": "s",
        "prompt_hash": "p",
        "status": status,
    }


def test_resume_does_not_duplicate(tmp_path):
    path = tmp_path / "pred.jsonl"
    checkpoint = JsonlCheckpoint(path)
    assert checkpoint.append_once(record())
    assert not checkpoint.append_once(record())
    assert len(path.read_text().splitlines()) == 1


def test_partial_last_line_is_ignored(tmp_path):
    path = tmp_path / "pred.jsonl"
    path.write_text(json.dumps(record()) + '\n{"broken"', encoding="utf-8")
    assert len(completed_keys(path)) == 1


def test_config_hash_does_not_resume_old_run(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"config_hash": "old", "status": "running"}), encoding="utf-8"
    )
    assert find_resumable_run(tmp_path, "new") is None
    assert find_resumable_run(tmp_path, "old") == run


def test_digest_and_prompt_are_part_of_key():
    a = ResumeKey("c", "m", "d1", "b", "r", "s", "p")
    b = ResumeKey("c", "m", "d2", "b", "r", "s", "p")
    assert a.text() != b.text()
