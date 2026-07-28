import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_models.py"
    spec = importlib.util.spec_from_file_location("prepare_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_quantization_parser(monkeypatch):
    module = _module()

    class Result:
        returncode = 0
        stdout = "  Model\n    architecture qwen3vl\n    quantization        Q4_K_M\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: Result())
    assert module.cli_quantization("model") == "Q4_K_M"
