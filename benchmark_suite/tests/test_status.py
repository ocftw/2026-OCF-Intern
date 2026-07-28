import json
import os
import subprocess


def test_status_sh_displays_friendly_smoke_progress(suite_dir, tmp_path):
    runs = tmp_path / "runs"
    state = runs / ".state"
    run = runs / "run-1"
    smoke = run / "smoke"
    state.mkdir(parents=True)
    smoke.mkdir(parents=True)
    (state / "current_run").write_text(str(run))
    (state / "requested_run_id").write_text("run-1")
    (state / "active.pid").write_text(str(os.getpid()))
    (state / "active.log").write_text("/tmp/test.log")
    (run / "manifest.json").write_text(json.dumps({"status": "running"}))
    (smoke / "progress.json").write_text(
        json.dumps(
            {
                "current_model": "qwen3_vl_4b",
                "current_benchmark": "tc_str",
                "phase": "inference",
                "completed": 1,
                "total": 3,
                "updated_at_utc": "2026-07-28T00:00:00+00:00",
            }
        )
    )
    (smoke / "combination_status.json").write_text(
        json.dumps(
            [
                {
                    "model": "qwen3_vl_4b",
                    "benchmark": "omnidocbench",
                    "status": "completed",
                }
            ]
        )
    )
    result = subprocess.run(
        [str(suite_dir / "status.sh")],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "BENCHMARK_RUNS_DIR": str(runs)},
    )
    assert "Smoke test" in result.stdout
    assert "Qwen3-VL 4B" in result.stdout
    assert "TC-STR" in result.stdout
    assert "1/3（33.3%）" in result.stdout
    assert "1/15 完成" in result.stdout
    assert "pulling manifest" not in result.stdout
