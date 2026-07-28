from pathlib import Path

import requests

from ocf_benchmark.ollama_client import OllamaClient


class Response:
    def __init__(self, status, data=None, text=""):
        self.status_code = status
        self._data = data or {}
        self.text = text
        self.ok = status < 400

    def json(self):
        return self._data

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(self.text)


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, *_args, **_kwargs):
        return Response(200)


def image(tmp_path: Path) -> Path:
    path = tmp_path / "x.png"
    path.write_bytes(b"image")
    return path


def test_full_chat_request_and_telemetry(tmp_path):
    session = Session(
        [
            Response(
                200,
                {
                    "message": {"content": "答案"},
                    "done_reason": "stop",
                    "eval_count": 10,
                    "eval_duration": 1_000_000_000,
                },
            )
        ]
    )
    client = OllamaClient("http://test", session=session, sleep=lambda _: None)
    result = client.chat(
        model="m",
        prompt="p",
        image_path=image(tmp_path),
        options={"temperature": 0, "seed": 42, "repeat_penalty": 1.0},
        timeout=1,
    )
    assert result.text == "答案"
    payload = session.calls[0][1]["json"]
    assert session.calls[0][0].endswith("/api/chat")
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "user"


def test_transport_retries_at_most_three(tmp_path):
    session = Session([requests.ConnectionError("x")] * 3)
    client = OllamaClient("http://test", session=session, sleep=lambda _: None, max_attempts=3)
    result = client.chat(model="m", prompt="p", image_path=image(tmp_path), options={}, timeout=1)
    assert len(session.calls) == 3
    assert result.status == "terminal_failure"
    assert result.retry_count == 2


def test_content_failure_not_retried(tmp_path):
    session = Session([Response(400, text="bad request")])
    client = OllamaClient("http://test", session=session, sleep=lambda _: None)
    result = client.chat(model="m", prompt="p", image_path=image(tmp_path), options={}, timeout=1)
    assert len(session.calls) == 1
    assert result.error_type == "content"


def test_length_is_completed_and_not_retried(tmp_path):
    session = Session([Response(200, {"message": {"content": "x"}, "done_reason": "length"})])
    client = OllamaClient("http://test", session=session, sleep=lambda _: None)
    result = client.chat(model="m", prompt="p", image_path=image(tmp_path), options={}, timeout=1)
    assert len(session.calls) == 1
    assert result.status == "completed"
    assert result.truncated


def test_model_metadata_excludes_verbose_token_arrays():
    session = Session(
        [
            Response(
                200,
                {
                    "details": {"quantization_level": "Q4_K_M", "family": "qwen"},
                    "template": "template",
                    "model_info": {
                        "general.architecture": "qwen",
                        "tokenizer.ggml.tokens": ["a", "b"],
                        "tokenizer.ggml.merges": ["a b"],
                    },
                    "tensors": [{"name": "large"}],
                },
            )
        ]
    )
    metadata = OllamaClient("http://test", session=session).model_metadata("model")
    assert metadata["model_info"] == {"general.architecture": "qwen"}
    assert "tensors" not in metadata["raw_show"]
    assert len(metadata["full_show_sha256"]) == 64
