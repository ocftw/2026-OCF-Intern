"""Ollama /api/chat client。

請求結構參考既有 ``eval/ollama_client.py``，但改用 chat、保留完整 telemetry，
並只對 transport/timeout/HTTP 5xx 重試。
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests


@dataclass
class OllamaResult:
    text: str = ""
    status: str = "completed"
    error_type: str = ""
    error_message: str = ""
    http_status: int | None = None
    retry_count: int = 0
    wall_clock_latency: float = 0.0
    data: dict[str, Any] | None = None

    @property
    def truncated(self) -> bool:
        return (self.data or {}).get("done_reason") == "length"


class OllamaClient:
    def __init__(
        self,
        host: str,
        retry_delays: list[float] | None = None,
        max_attempts: int = 3,
        circuit_breaker_failures: int = 5,
        circuit_poll_seconds: float = 30,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.host = host.rstrip("/")
        self.retry_delays = retry_delays or [2, 5, 15]
        self.max_attempts = max_attempts
        self.breaker_limit = circuit_breaker_failures
        self.poll_seconds = circuit_poll_seconds
        self.session = session or requests.Session()
        self.sleep = sleep
        self._consecutive_transport = 0

    def health(self, timeout: float = 5) -> bool:
        try:
            return self.session.get(f"{self.host}/api/tags", timeout=timeout).ok
        except requests.RequestException:
            return False

    def _wait_for_health(self) -> None:
        while not self.health():
            self.sleep(self.poll_seconds)
        self._consecutive_transport = 0

    @staticmethod
    def _image_b64(image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("ascii")

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        options: dict[str, Any],
        timeout: float,
    ) -> OllamaResult:
        try:
            image = self._image_b64(image_path)
        except OSError as exc:
            return OllamaResult(
                status="terminal_failure", error_type="content", error_message=str(exc)
            )
        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [image]}],
            "options": options,
        }
        start = time.monotonic()
        last = OllamaResult()
        for attempt in range(self.max_attempts):
            if self._consecutive_transport >= self.breaker_limit:
                self._wait_for_health()
            try:
                response = self.session.post(f"{self.host}/api/chat", json=payload, timeout=timeout)
                if response.status_code >= 500:
                    last = OllamaResult(
                        status="retryable_failure",
                        error_type="http_5xx",
                        error_message=response.text[:1000],
                        http_status=response.status_code,
                    )
                elif response.status_code >= 400:
                    return OllamaResult(
                        status="terminal_failure",
                        error_type="content",
                        error_message=response.text[:1000],
                        http_status=response.status_code,
                        retry_count=attempt,
                        wall_clock_latency=time.monotonic() - start,
                    )
                else:
                    data = response.json()
                    self._consecutive_transport = 0
                    return OllamaResult(
                        text=(data.get("message") or {}).get("content", ""),
                        http_status=response.status_code,
                        retry_count=attempt,
                        wall_clock_latency=time.monotonic() - start,
                        data=data,
                    )
            except requests.Timeout as exc:
                self._consecutive_transport += 1
                last = OllamaResult(
                    status="retryable_failure", error_type="timeout", error_message=str(exc)
                )
            except requests.RequestException as exc:
                self._consecutive_transport += 1
                last = OllamaResult(
                    status="retryable_failure", error_type="transport", error_message=str(exc)
                )
            if attempt + 1 < self.max_attempts:
                self.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
        last.status = "terminal_failure"
        last.retry_count = self.max_attempts - 1
        last.wall_clock_latency = time.monotonic() - start
        return last

    def model_metadata(self, model: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.host}/api/show", json={"model": model, "verbose": True}, timeout=60
        )
        response.raise_for_status()
        data = response.json()
        details = data.get("details") or {}
        return {
            "tag": model,
            "digest": data.get("digest") or "",
            "quantization": details.get("quantization_level") or "",
            "family": details.get("family") or "",
            "parameter_size": details.get("parameter_size") or "",
            "template": data.get("template") or "",
            "modelfile": data.get("modelfile") or "",
            "parameters": data.get("parameters") or "",
            "model_info": data.get("model_info") or {},
            "raw_show": data,
        }

    def model_digest(self, model: str) -> str:
        response = self.session.get(f"{self.host}/api/tags", timeout=30)
        response.raise_for_status()
        for item in response.json().get("models", []):
            if item.get("name") == model or item.get("model") == model:
                return str(item.get("digest") or "")
        raise RuntimeError(f"ollama list 找不到 {model}")

    def unload(self, model: str) -> None:
        payload = {"model": model, "keep_alive": 0}
        response = self.session.post(f"{self.host}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
