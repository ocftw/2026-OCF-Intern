import importlib.util
from pathlib import Path

import pytest
import requests


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_429_retries_and_reuses_arguments():
    module = _module()
    calls = []
    sleeps = []

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            response = requests.Response()
            response.status_code = 429
            error = requests.HTTPError("rate limited", response=response)
            raise error
        return "/snapshot"

    result = module.snapshot_download_with_retry(
        snapshot_fn=download,
        sleep_fn=sleeps.append,
        retry_delays=(1, 2),
        repo_id="owner/dataset",
        revision="fixed",
        local_dir="/cache",
    )
    assert result == "/snapshot"
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert all(call["revision"] == "fixed" for call in calls)


def test_hf_404_does_not_retry():
    module = _module()
    calls = 0

    def download(**_kwargs):
        nonlocal calls
        calls += 1
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError("not found", response=response)

    with pytest.raises(requests.HTTPError):
        module.snapshot_download_with_retry(
            snapshot_fn=download,
            sleep_fn=lambda _delay: None,
            retry_delays=(1, 2),
        )
    assert calls == 1
