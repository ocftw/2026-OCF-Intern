from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import utc_now


@dataclass
class GenerateResult:
    success: bool
    request: dict[str, Any]
    response: dict[str, Any] | None
    http_status: int | None
    error_type: str | None
    error_message: str | None
    started_at: str
    ended_at: str
    latency_seconds: float
    attempt_number: int
    completion_validation: dict[str, Any]


STRICT_COMPLETION_POLICY = {"mode": "strict"}


def transport_accepted(result: GenerateResult) -> bool:
    """Whether Ollama accepted and executed the request, independent of output quality."""
    return (
        result.http_status == 200
        and result.response is not None
        and not result.response.get("error")
    )


def validate_completion(
    response: dict[str, Any],
    completion_policy: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    policy = completion_policy or STRICT_COMPLETION_POLICY
    mode = str(policy.get("mode", "strict"))
    missing_fields = [
        key
        for key in ("done_reason", "prompt_eval_count", "eval_count")
        if response.get(key) is None
    ]
    done = response.get("done")
    nonempty_response = bool(str(response.get("response") or "").strip())
    strict_complete = bool(done) and not missing_fields and nonempty_response
    accepted_by_exception = (
        mode == "allow_nonempty_response_without_terminal_metadata"
        and nonempty_response
    )
    accepted = strict_complete or accepted_by_exception
    validation = {
        "policy_mode": mode,
        "strict_complete": strict_complete,
        "accepted_by_exception": accepted_by_exception and not strict_complete,
        "done": done,
        "missing_completion_fields": missing_fields,
        "nonempty_response": nonempty_response,
        "truncation_observable": response.get("done_reason") is not None or response.get("eval_count") is not None,
    }
    return accepted, validation


class OllamaClient:
    def __init__(
        self,
        host: str,
        timeout_seconds: float,
        max_attempts: int,
        retry_backoff_seconds: list[float],
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff = retry_backoff_seconds

    def _json(self, path: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> tuple[int, dict[str, Any]]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def tags(self) -> list[dict[str, Any]]:
        return self._json("/api/tags", timeout=30)[1].get("models", [])

    def show(self, tag: str) -> dict[str, Any]:
        return self._json("/api/show", {"model": tag, "verbose": False}, timeout=90)[1]

    def unload(self, tag: str) -> None:
        self._json("/api/generate", {"model": tag, "keep_alive": 0, "stream": False}, timeout=90)

    def generate(
        self,
        tag: str,
        prompt: str,
        image_path: Path,
        options: dict[str, Any],
        *,
        initial_attempt: int = 1,
        keep_alive: str | int = "5m",
        completion_policy: dict[str, Any] | None = None,
    ) -> list[GenerateResult]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        top_level_think = bool(options.get("think", False))
        generation_options = {key: value for key, value in options.items() if key != "think"}
        wire_payload = {
            "model": tag,
            "prompt": prompt,
            "images": [encoded],
            "stream": False,
            "think": top_level_think,
            "keep_alive": keep_alive,
            "options": generation_options,
        }
        recorded_request = {
            **{key: value for key, value in wire_payload.items() if key != "images"},
            "images": [
                {
                    "path": str(image_path),
                    "bytes": image_path.stat().st_size,
                    "base64_included_on_wire": True,
                }
            ],
        }
        results: list[GenerateResult] = []
        for offset in range(self.max_attempts):
            attempt = initial_attempt + offset
            started_at = utc_now()
            started = time.monotonic()
            try:
                status, response = self._json("/api/generate", wire_payload, self.timeout)
                api_error = response.get("error")
                complete, completion_validation = validate_completion(response, completion_policy)
                if api_error:
                    error_type = "api_error"
                    error_message = str(api_error)
                elif not complete:
                    error_type = "incomplete_response"
                    error_message = (
                        f"Ollama returned done={response.get('done')!r}; "
                        "missing completion fields: "
                        f"{', '.join(completion_validation['missing_completion_fields']) or 'none'}; "
                        f"policy={completion_validation['policy_mode']}"
                    )
                else:
                    error_type = None
                    error_message = None
                result = GenerateResult(
                    success=complete and not api_error,
                    request=recorded_request,
                    response=response,
                    http_status=status,
                    error_type=error_type,
                    error_message=error_message,
                    started_at=started_at,
                    ended_at=utc_now(),
                    latency_seconds=time.monotonic() - started,
                    attempt_number=attempt,
                    completion_validation=completion_validation,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:4000]
                result = GenerateResult(
                    False, recorded_request, None, exc.code,
                    "http_retryable" if exc.code >= 500 else "http_terminal", body,
                    started_at, utc_now(), time.monotonic() - started, attempt,
                    {"policy_mode": (completion_policy or STRICT_COMPLETION_POLICY).get("mode", "strict")},
                )
            except TimeoutError as exc:
                result = GenerateResult(
                    False, recorded_request, None, None, "timeout", str(exc),
                    started_at, utc_now(), time.monotonic() - started, attempt,
                    {"policy_mode": (completion_policy or STRICT_COMPLETION_POLICY).get("mode", "strict")},
                )
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                result = GenerateResult(
                    False, recorded_request, None, None, "transport", str(exc),
                    started_at, utc_now(), time.monotonic() - started, attempt,
                    {"policy_mode": (completion_policy or STRICT_COMPLETION_POLICY).get("mode", "strict")},
                )
            results.append(result)
            if result.success or result.error_type == "http_terminal":
                break
            if offset + 1 < self.max_attempts:
                time.sleep(self.backoff[min(offset, len(self.backoff) - 1)])
        return results


def ollama_ps() -> dict[str, Any]:
    proc = subprocess.run(["ollama", "ps"], text=True, capture_output=True, timeout=30, check=False)
    result: dict[str, Any] = {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "rows": [],
    }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return result
    headers = ["NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"]
    positions = [lines[0].find(header) for header in headers]
    if any(position < 0 for position in positions[:-1]):
        return result
    for line in lines[1:]:
        row: dict[str, str] = {}
        for i, header in enumerate(headers):
            start = positions[i]
            end = positions[i + 1] if i + 1 < len(positions) else None
            row[header.lower()] = line[start:end].strip()
        result["rows"].append(row)
    return result
