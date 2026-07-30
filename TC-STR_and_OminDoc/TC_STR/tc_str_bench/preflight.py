from __future__ import annotations

import html
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import Settings, run_signature
from .ollama import OllamaClient, ollama_ps, transport_accepted
from .util import atomic_write_json, atomic_write_text, run_command, utc_now


def _command_facts(settings: Settings) -> dict[str, Any]:
    commands = {
        "id": ["id"],
        "ollama_version": ["ollama", "--version"],
        "ollama_list": ["ollama", "list"],
        "nvidia_smi": ["nvidia-smi"],
        "python_version": ["python3", "--version"],
        "git_head": ["git", "-C", str(settings.root.parent.parent), "rev-parse", "HEAD"],
        "git_status": ["git", "-C", str(settings.root.parent.parent), "status", "--short"],
        "disk_space": ["df", "-h", str(settings.raw["paths"]["dataset"]), str(settings.output_root)],
    }
    return {name: run_command(command, timeout=60) for name, command in commands.items()}


def _write_probe(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "writable": False}
    if not path.is_dir():
        result["error"] = "directory_missing"
        return result
    stat = path.stat()
    import grp
    import pwd

    result.update(
        {
            "owner": pwd.getpwuid(stat.st_uid).pw_name,
            "group": grp.getgrgid(stat.st_gid).gr_name,
            "mode": oct(stat.st_mode & 0o7777),
        }
    )
    try:
        fd, name = tempfile.mkstemp(prefix=".tcstr_preflight_", dir=path)
        os.write(fd, b"tc-str preflight\n")
        os.fsync(fd)
        os.close(fd)
        created = Path(name)
        created_stat = created.stat()
        result["created_file_group"] = grp.getgrgid(created_stat.st_gid).gr_name
        created.unlink()
        result["writable"] = True
    except OSError as exc:
        result["error"] = str(exc)
    return result


def _quantization_matches(required: str | None, actual: str) -> bool:
    if not required:
        return True
    required_upper, actual_upper = required.upper(), actual.upper()
    return actual_upper.startswith(required_upper) if required_upper == "Q4" else actual_upper == required_upper


def _context_from_show(show: dict[str, Any]) -> int | None:
    for key, value in (show.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, (int, float)):
            return int(value)
    return None


def _model_record(model: dict[str, Any], tags: dict[str, dict[str, Any]], client: OllamaClient) -> dict[str, Any]:
    record = {
        "id": model["id"],
        "logical_name": model["logical_name"],
        "exact_tag": model.get("exact_tag"),
        "required_quantization": model.get("required_quantization"),
        "completion_policy": model.get("completion_policy") or {"mode": "strict"},
        "status": "PENDING",
        "blockers": [],
        "warnings": [],
    }
    tag = model.get("exact_tag")
    if not tag:
        record["blockers"].append(model.get("confirmation_required") or "exact tag 尚未確認")
        record["status"] = "BLOCKED"
        return record
    if tag not in tags:
        record["blockers"].append(f"exact tag 未安裝: {tag}；禁止自動 pull 或替代")
        record["status"] = "BLOCKED"
        return record
    installed = tags[tag]
    record.update(
        {
            "digest": installed.get("digest", ""),
            "size_bytes": installed.get("size"),
            "installed_details": installed.get("details", {}),
        }
    )
    try:
        show = client.show(tag)
    except Exception as exc:
        record["blockers"].append(f"/api/show 失敗: {exc}")
        record["status"] = "BLOCKED"
        return record
    details = show.get("details") or {}
    record.update(
        {
            "architecture": details.get("family"),
            "parameter_count": details.get("parameter_size"),
            "quantization": details.get("quantization_level") or "",
            "context_limit": _context_from_show(show),
            "capabilities": show.get("capabilities") or [],
            "model_info": show.get("model_info") or {},
            "template_sha256": __import__("hashlib").sha256(str(show.get("template", "")).encode()).hexdigest(),
        }
    )
    capabilities = {str(item).lower() for item in record["capabilities"]}
    if "vision" not in capabilities:
        record["blockers"].append(f"model capabilities 不含 vision: {sorted(capabilities)}")
    if not _quantization_matches(model.get("required_quantization"), record["quantization"]):
        record["blockers"].append(
            f"量化不符：要求 {model.get('required_quantization')}，實際 {record['quantization'] or 'unknown'}"
        )
    record["status"] = "BLOCKED" if record["blockers"] else "METADATA_READY"
    return record


def _report_html(report: dict[str, Any]) -> str:
    model_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(m['logical_name']))}</td>"
        f"<td><code>{html.escape(str(m.get('exact_tag') or 'UNCONFIRMED'))}</code></td>"
        f"<td>{html.escape(str(m.get('digest') or '—'))}</td>"
        f"<td>{html.escape(str(m.get('quantization') or '—'))}</td>"
        f"<td>{html.escape(str(m.get('processor') or '—'))}</td>"
        f"<td class='{m['status'].lower()}'>{html.escape(m['status'])}</td>"
        f"<td>{html.escape('; '.join(m.get('blockers', [])))}</td></tr>"
        for m in report["models"]
    )
    blockers = "".join(f"<li>{html.escape(item)}</li>" for item in report["blockers"])
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in report.get("warnings", []))
    raw = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    return f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>TC-STR Preflight</title><style>
body{{font:15px/1.55 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.45rem;vertical-align:top}}
.blocked{{color:#a00;font-weight:700}}.ready{{color:#076b3b;font-weight:700}}code,pre{{font-family:ui-monospace,monospace}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem}}details{{margin-top:2rem}}
</style><h1>TC-STR Preflight：{html.escape(report['status'])}</h1>
<p>Run <code>{html.escape(report['run_id'])}</code> · {html.escape(report['started_at'])} → {html.escape(report['ended_at'])}</p>
<h2>Blockers</h2><ul>{blockers or '<li>無</li>'}</ul>
<h2>Warnings / protocol exceptions</h2><ul>{warnings or '<li>無</li>'}</ul>
<h2>Models</h2><table><thead><tr><th>Logical model</th><th>Exact tag</th><th>Digest</th>
<th>Quantization</th><th>Processor</th><th>Status</th><th>Notes</th></tr></thead><tbody>{model_rows}</tbody></table>
<details><summary>完整 machine-readable preflight</summary><pre>{raw}</pre></details></html>"""


def execute_preflight(
    settings: Settings,
    run_id: str,
    run_dir: Path,
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    started_at = utc_now()
    run_dir.mkdir(parents=True, exist_ok=True)
    facts = _command_facts(settings)
    probes = [
        _write_probe(Path("/mnt/nvme/scratch")),
        _write_probe(Path("/opt/ocf-ai/outputs")),
        _write_probe(Path(settings.raw["paths"]["scratch"])),
        _write_probe(settings.output_root),
    ]
    blockers: list[str] = []
    for probe in probes:
        if not probe["writable"]:
            blockers.append(f"不可寫入 {probe['path']}: {probe.get('error', '')}")
        if probe.get("group") != "ocfai" or probe.get("created_file_group") != "ocfai":
            blockers.append(
                f"{probe['path']} 群組必須為 ocfai；directory={probe.get('group')} created={probe.get('created_file_group')}"
            )
    client = OllamaClient(
        settings.raw["ollama"]["host"],
        settings.raw["ollama"]["timeout_seconds"],
        settings.raw["ollama"]["max_attempts"],
        settings.raw["ollama"]["retry_backoff_seconds"],
    )
    try:
        tag_items = client.tags()
    except Exception as exc:
        tag_items = []
        blockers.append(f"Ollama /api/tags 無法存取: {exc}")
    tags = {(item.get("name") or item.get("model")): item for item in tag_items}
    models = [_model_record(model, tags, client) for model in settings.models]
    warmup_sample = dataset_manifest["samples"][dataset_manifest["smoke_selection"]["sample_indices"][0]]
    image = settings.dataset_dir / warmup_sample["image_relative_path"]
    for model in models:
        if model["status"] != "METADATA_READY":
            continue
        tag = str(model["exact_tag"])
        try:
            attempts = client.generate(
                tag,
                settings.prompt,
                image,
                settings.options,
                keep_alive="5m",
                completion_policy=model.get("completion_policy"),
            )
            model["warmup"] = [
                {
                    "success": result.success,
                    "http_status": result.http_status,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                    "latency_seconds": result.latency_seconds,
                    "response": {
                        key: value for key, value in (result.response or {}).items()
                        if key != "context"
                    } if result.response is not None else None,
                    "completion_validation": result.completion_validation,
                }
                for result in attempts
            ]
            if not transport_accepted(attempts[-1]):
                last = attempts[-1]
                model["blockers"].append(
                    "正式參數 vision warm-up 未取得可證明模型接受圖片的 HTTP 200 回應"
                    f"：{last.error_type or 'unknown'}"
                    f"；{last.error_message or '無錯誤訊息'}"
                )
            else:
                if not attempts[-1].success:
                    model["warnings"].append(
                        "warm-up 的輸出層 completion/正文不完整；依共同政策視為模型輸出失敗，"
                        "不影響 vision/GPU preflight"
                    )
                if attempts[-1].completion_validation.get("accepted_by_exception"):
                    model["warnings"].append(
                        "使用者核准 completion metadata 例外：有效非空 response 可在 "
                        "done=false / done_reason、prompt_eval_count、eval_count 缺失時接受；"
                        "token 使用與自動截斷診斷標為不可觀測"
                    )
                ps = ollama_ps()
                model["ollama_ps"] = ps
                matching = [row for row in ps["rows"] if row.get("name") == tag]
                if not matching:
                    model["blockers"].append("warm-up 後 ollama ps 找不到 exact tag")
                else:
                    model["processor"] = matching[0].get("processor")
                    model["loaded_size"] = matching[0].get("size")
                    model["loaded_context"] = matching[0].get("context")
                    if model["processor"] != "100% GPU":
                        model["blockers"].append(f"PROCESSOR 必須精確為 100% GPU，實際 {model['processor']}")
        except Exception as exc:
            model["blockers"].append(f"warm-up/ollama ps 失敗: {exc}")
        finally:
            try:
                client.unload(tag)
                model["unloaded"] = True
            except Exception as exc:
                model["unloaded"] = False
                model["blockers"].append(f"模型卸載失敗: {exc}")
        model["status"] = "BLOCKED" if model["blockers"] else "READY"
    for model in models:
        if model["status"] != "READY":
            blockers.extend(f"{model['logical_name']}: {reason}" for reason in model["blockers"])
    warnings = [
        f"{model['logical_name']}: {warning}"
        for model in models
        for warning in model.get("warnings", [])
    ]
    version_line = facts["ollama_version"]["stdout"].strip()
    signature, material = run_signature(
        settings,
        dataset_manifest["dataset_fingerprint"],
        models,
        version_line,
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "READY" if not blockers and len(models) == 8 else "BLOCKED",
        "started_at": started_at,
        "ended_at": utc_now(),
        "run_signature": signature,
        "signature_material": material,
        "environment": {
            "platform": platform.platform(),
            "commands": facts,
            "write_probes": probes,
            "kv_cache_type": "fp16 (主管設定；本工具未修改服務環境變數)",
        },
        "installed_ollama_models": tag_items,
        "models": models,
        "blockers": blockers,
        "warnings": warnings,
    }
    atomic_write_json(run_dir / "preflight_report.json", report)
    atomic_write_text(run_dir / "preflight_report.html", _report_html(report))
    return report
