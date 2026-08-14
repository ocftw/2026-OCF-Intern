from __future__ import annotations

import base64
import contextlib
import csv
import datetime as dt
import difflib
import hashlib
import html
import json
import os
import pathlib
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
TAIPEI = dt.timezone(dt.timedelta(hours=8))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def taipei_time(value: str | None) -> str:
    if not value:
        return ""
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(TAIPEI).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: pathlib.Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    kwargs = {} if isinstance(data, bytes) else {"encoding": "utf-8", "newline": "\n"}
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, mode, **kwargs) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(
    args: list[str], *, timeout: int = 120, check: bool = False, cwd: pathlib.Path | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        result = {
            "command": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {
            "command": args,
            "returncode": 127,
            "stdout": getattr(exc, "stdout", "") or "",
            "stderr": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    if check and result["returncode"] != 0:
        raise RuntimeError(f"command failed: {args!r}: {result['stderr']}")
    return result


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    models: list[dict[str, Any]]
    smoke: dict[str, Any]

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            raw=read_json(CONFIG_DIR / "benchmark.json"),
            models=read_json(CONFIG_DIR / "models.yaml")["models"],
            smoke=read_json(CONFIG_DIR / "smoke_pages.json"),
        )

    @property
    def prompt_hash(self) -> str:
        return sha256_text(self.raw["inference"]["prompt"])

    @property
    def options_hash(self) -> str:
        inf = self.raw["inference"]
        common = {
            "stream": inf["stream"],
            "think": inf["think"],
            "keep_alive": inf["keep_alive"],
            "options": inf["options"],
        }
        return sha256_text(canonical_json(common))

    @property
    def code_hash(self) -> str:
        files = [
            *sorted((ROOT / "omnidocbench").glob("*.py")),
            *sorted((ROOT / "config").glob("*")),
            *sorted(ROOT.glob("*.sh")),
            ROOT / "scripts" / "fetch_dataset.py",
        ]
        digest = hashlib.sha256()
        for path in files:
            if path.is_file():
                digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
        return digest.hexdigest()

    @property
    def run_signature(self) -> str:
        data = {
            "benchmark": self.raw["benchmark"],
            "models": self.models,
            "prompt_hash": self.prompt_hash,
            "options_hash": self.options_hash,
            "code_hash": self.code_hash,
            "evaluator": self.raw["evaluator"],
            "versions": self.raw["versions"],
        }
        return sha256_text(canonical_json(data))

    @property
    def run_id(self) -> str:
        return os.environ.get("OMNIDOCBENCH_RUN_ID", f"v1_6_{self.run_signature[:12]}")

    @property
    def output_dir(self) -> pathlib.Path:
        return pathlib.Path(self.raw["paths"]["output_root"]) / self.run_id

    @property
    def gt_path(self) -> pathlib.Path:
        b = self.raw["benchmark"]
        return pathlib.Path(self.raw["paths"]["dataset_root"]) / b["ground_truth_filename"]

    @property
    def image_dir(self) -> pathlib.Path:
        return pathlib.Path(self.raw["paths"]["dataset_root"]) / "images"


def git_metadata() -> dict[str, Any]:
    sha = run_command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    status = run_command(["git", "-C", str(ROOT), "status", "--porcelain=v1"])
    return {
        "sha": sha["stdout"].strip() if sha["returncode"] == 0 else None,
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else None,
        "status": status["stdout"].splitlines(),
    }


def filename_to_page_id(filename: str) -> str:
    return pathlib.Path(filename).stem


def prediction_filename(image_filename: str) -> str:
    return pathlib.Path(image_filename).with_suffix(".md").name


def image_dimensions(path: pathlib.Path) -> tuple[int | None, int | None]:
    data = path.read_bytes()[:32]
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(b"\xff\xd8"):
        with path.open("rb") as handle:
            handle.read(2)
            while True:
                marker = handle.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    continue
                while marker[1] == 0xFF:
                    marker = bytes([marker[0]]) + handle.read(1)
                if marker[1] in (0xD8, 0xD9):
                    continue
                length_data = handle.read(2)
                if len(length_data) < 2:
                    break
                length = int.from_bytes(length_data, "big")
                if marker[1] in range(0xC0, 0xC4):
                    payload = handle.read(5)
                    return int.from_bytes(payload[3:5], "big"), int.from_bytes(payload[1:3], "big")
                handle.seek(length - 2, 1)
    return None, None


def load_dataset(settings: Settings) -> list[dict[str, Any]]:
    return read_json(settings.gt_path)


def validate_dataset(settings: Settings, *, hash_images: bool = True) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    b = settings.raw["benchmark"]
    report: dict[str, Any] = {
        "path": str(settings.gt_path),
        "expected_revision": b["dataset_revision"],
        "expected_pages": b["expected_pages"],
        "expected_gt_sha256": b["ground_truth_sha256"],
    }
    if not settings.gt_path.is_file():
        errors.append(f"missing official v1.6 GT: {settings.gt_path}")
        report["status"] = "MISSING"
        return report, errors
    revision_manifest_path = settings.gt_path.parent / "huggingface_revision_manifest.json"
    revision_files: dict[str, dict[str, Any]] = {}
    if not revision_manifest_path.is_file():
        errors.append(f"missing pinned Hugging Face revision manifest: {revision_manifest_path}")
        report["revision_manifest_status"] = "MISSING"
    else:
        try:
            revision_manifest = read_json(revision_manifest_path)
            report["revision_manifest"] = {
                "repo": revision_manifest.get("repo"),
                "revision": revision_manifest.get("revision"),
                "file_count": len(revision_manifest.get("files", [])),
            }
            if revision_manifest.get("repo") != b["dataset_repo"]:
                errors.append("dataset repository in revision manifest does not match the pin")
            if revision_manifest.get("revision") != b["dataset_revision"]:
                errors.append("dataset revision manifest is not the pinned v1.6 revision")
            revision_files = {
                pathlib.Path(item["path"]).name: item
                for item in revision_manifest.get("files", [])
                if item.get("path", "").startswith("images/")
            }
            if len(revision_files) != b["expected_pages"]:
                errors.append(
                    f"revision manifest must contain {b['expected_pages']} unique page images, "
                    f"found {len(revision_files)}"
                )
            report["revision_manifest_status"] = "PASS"
        except Exception as exc:
            errors.append(f"cannot parse revision manifest: {exc}")
            report["revision_manifest_status"] = "INVALID"
    actual_hash = sha256_file(settings.gt_path)
    report["actual_gt_sha256"] = actual_hash
    if actual_hash != b["ground_truth_sha256"]:
        errors.append(f"GT SHA-256 mismatch: expected {b['ground_truth_sha256']}, got {actual_hash}")
    try:
        pages = load_dataset(settings)
    except Exception as exc:
        errors.append(f"GT JSON is not parseable: {exc}")
        report["status"] = "INVALID"
        return report, errors
    report["page_count"] = len(pages)
    if len(pages) != b["expected_pages"]:
        errors.append(f"v1.6 full set requires {b['expected_pages']} pages, found {len(pages)}")
    names: list[str] = []
    manifest: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        try:
            info = page["page_info"]
            name = pathlib.Path(info["image_path"]).name
        except Exception as exc:
            errors.append(f"page {index} has invalid schema: {exc}")
            continue
        names.append(name)
        image = settings.image_dir / name
        entry = {
            "index": index,
            "page_id": filename_to_page_id(name),
            "filename": name,
            "image_path": str(image),
            "exists": image.is_file(),
            "attributes": info.get("page_attribute", {}),
            "declared_width": info.get("width"),
            "declared_height": info.get("height"),
        }
        if image.is_file():
            entry["size"] = image.stat().st_size
            if hash_images:
                entry["sha256"] = sha256_file(image)
                expected_image = revision_files.get(name, {}).get("sha256")
                entry["expected_sha256"] = expected_image
                entry["hash_matches_revision"] = expected_image == entry["sha256"]
                if expected_image and expected_image != entry["sha256"]:
                    errors.append(f"image SHA-256 mismatch for {name}")
            width, height = image_dimensions(image)
            entry["actual_width"] = width
            entry["actual_height"] = height
        manifest.append(entry)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    missing = [x["filename"] for x in manifest if not x["exists"]]
    report.update(
        {
            "unique_filenames": len(set(names)),
            "duplicates": duplicates,
            "missing_image_count": len(missing),
            "missing_images_preview": missing[:50],
            "status": "PASS" if not errors and not missing and not duplicates else "FAIL",
            "manifest": manifest,
        }
    )
    if duplicates:
        errors.append(f"duplicate image filenames: {len(duplicates)}")
    if missing:
        errors.append(f"missing official page images: {len(missing)}")
    return report, errors


def validate_smoke_selection(settings: Settings, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {x["filename"]: x for x in manifest}
    selected: list[dict[str, Any]] = []
    for wanted in settings.smoke["pages"]:
        found = by_name.get(wanted["filename"])
        if found is None or found["index"] != wanted["index"]:
            raise ValueError(f"fixed smoke page mismatch: {wanted}")
        selected.append(found)
    if len(selected) != settings.raw["smoke"]["page_count"]:
        raise ValueError("smoke page count mismatch")
    return selected


def make_dataset_manifest(settings: Settings, report: dict[str, Any], output: pathlib.Path) -> None:
    manifest = {
        "benchmark": settings.raw["benchmark"],
        "created_at": utc_now(),
        "page_count": report.get("page_count"),
        "ground_truth_sha256": report.get("actual_gt_sha256"),
        "pages": report.get("manifest", []),
    }
    write_json(output, manifest)


class CheckpointDB:
    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._schema()

    def _schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
              signature TEXT PRIMARY KEY, run_id TEXT NOT NULL, status TEXT NOT NULL,
              config_json TEXT NOT NULL, started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              ended_at TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS page_result (
              signature TEXT NOT NULL, model_id TEXT NOT NULL, page_id TEXT NOT NULL,
              filename TEXT NOT NULL, status TEXT NOT NULL, image_sha256 TEXT NOT NULL,
              raw_path TEXT, raw_sha256 TEXT, prediction_path TEXT, prediction_sha256 TEXT,
              response_json TEXT, attempts INTEGER NOT NULL DEFAULT 0,
              error_history_json TEXT NOT NULL DEFAULT '[]',
              started_at TEXT, ended_at TEXT, elapsed_seconds REAL,
              eval_count INTEGER, prompt_eval_count INTEGER, done_reason TEXT,
              load_duration_ns INTEGER, prompt_eval_duration_ns INTEGER,
              eval_duration_ns INTEGER, total_duration_ns INTEGER,
              PRIMARY KEY(signature, model_id, page_id)
            );
            CREATE TABLE IF NOT EXISTS event (
              id INTEGER PRIMARY KEY AUTOINCREMENT, signature TEXT NOT NULL,
              at TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL,
              detail_json TEXT
            );
            """
        )
        self.db.commit()

    def ensure_run(self, settings: Settings, status: str) -> None:
        now = utc_now()
        payload = canonical_json(settings.raw)
        existing = self.db.execute("SELECT * FROM run").fetchall()
        foreign = [row["signature"] for row in existing if row["signature"] != settings.run_signature]
        if foreign:
            raise RuntimeError(f"checkpoint contains a different signature: {foreign}")
        self.db.execute(
            """INSERT INTO run(signature,run_id,status,config_json,started_at,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(signature) DO UPDATE SET status=excluded.status,updated_at=excluded.updated_at""",
            (settings.run_signature, settings.run_id, status, payload, now, now),
        )
        self.db.commit()

    def event(self, settings: Settings, level: str, message: str, detail: Any = None) -> None:
        self.db.execute(
            "INSERT INTO event(signature,at,level,message,detail_json) VALUES(?,?,?,?,?)",
            (settings.run_signature, utc_now(), level, message, canonical_json(detail) if detail is not None else None),
        )
        self.db.commit()

    def reusable(
        self, settings: Settings, model_id: str, page_id: str, image_hash: str
    ) -> bool:
        row = self.db.execute(
            "SELECT * FROM page_result WHERE signature=? AND model_id=? AND page_id=?",
            (settings.run_signature, model_id, page_id),
        ).fetchone()
        if not row or row["status"] != "SUCCESS" or row["image_sha256"] != image_hash:
            return False
        for key, hash_key in (("raw_path", "raw_sha256"), ("prediction_path", "prediction_sha256")):
            value = row[key]
            if not value:
                return False
            path = pathlib.Path(value)
            if not path.is_file() or sha256_file(path) != row[hash_key]:
                return False
        return True

    def save_result(self, settings: Settings, record: dict[str, Any]) -> None:
        keys = [
            "model_id", "page_id", "filename", "status", "image_sha256", "raw_path",
            "raw_sha256", "prediction_path", "prediction_sha256", "response_json",
            "attempts", "error_history_json", "started_at", "ended_at", "elapsed_seconds",
            "eval_count", "prompt_eval_count", "done_reason", "load_duration_ns",
            "prompt_eval_duration_ns", "eval_duration_ns", "total_duration_ns",
        ]
        values = [record.get(key) for key in keys]
        placeholders = ",".join("?" for _ in range(len(keys) + 1))
        columns = ",".join(["signature"] + keys)
        updates = ",".join(f"{key}=excluded.{key}" for key in keys)
        self.db.execute(
            f"INSERT INTO page_result({columns}) VALUES({placeholders}) "
            f"ON CONFLICT(signature,model_id,page_id) DO UPDATE SET {updates}",
            [settings.run_signature] + values,
        )
        self.db.commit()

    def rows(self, settings: Settings) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM page_result WHERE signature=? ORDER BY model_id,filename",
                (settings.run_signature,),
            )
        ]

    def stored_signatures(self) -> list[str]:
        return [
            str(row["signature"])
            for row in self.db.execute("SELECT signature FROM run ORDER BY started_at")
        ]

    def rows_for_signature(self, signature: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM page_result WHERE signature=? ORDER BY model_id,filename",
                (signature,),
            )
        ]

    def close(self) -> None:
        self.db.close()


def api_json(
    method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: int = 30
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def ollama_models(settings: Settings) -> tuple[list[dict[str, Any]], str | None]:
    try:
        _, response = api_json("GET", settings.raw["inference"]["endpoint"] + "/api/tags")
        return response.get("models", []), None
    except Exception as exc:
        return [], str(exc)


def ollama_show(settings: Settings, tag: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        _, response = api_json(
            "POST",
            settings.raw["inference"]["endpoint"] + "/api/show",
            {"model": tag, "verbose": False},
            timeout=60,
        )
        return response, None
    except Exception as exc:
        return None, str(exc)

def capabilities_from_show(show: dict[str, Any]) -> list[str]:
    return list(show.get("capabilities", []))


def parse_processor(text: str, tag: str) -> str | None:
    for line in text.splitlines():
        if tag in line:
            if re.search(r"\b100%\s+GPU\b", line) and "CPU" not in line:
                return "100% GPU"
            match = re.search(
                r"(\d+%\s+GPU(?:\s*/\s*\d+%\s+CPU)?|\d+%\s+CPU(?:\s*/\s*\d+%\s+GPU)?)",
                line,
            )
            if match:
                return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def unload_model(settings: Settings, tag: str) -> None:
    with contextlib.suppress(Exception):
        api_json(
            "POST",
            settings.raw["inference"]["endpoint"] + settings.raw["inference"]["api"],
            {"model": tag, "keep_alive": 0},
            timeout=60,
        )


def postprocess_response(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = response.get("response")
    if not isinstance(raw, str):
        return "", {"modified": False, "reason": "missing_response_string"}
    stripped = raw.strip()
    wrappers = ("<think></think>", "<thought></thought>")
    modified = stripped != raw
    reasons = ["outer_whitespace"] if modified else []
    for wrapper in wrappers:
        if stripped.startswith(wrapper):
            stripped = stripped[len(wrapper):].lstrip()
            modified = True
            reasons.append(f"exact_empty_wrapper:{wrapper}")
            break
    return stripped, {"modified": modified, "reasons": reasons}


def detect_anomalies(
    prediction: str, response: dict[str, Any], prompt: str, num_predict: int
) -> dict[str, bool]:
    lower = prediction.lower().strip()
    control_only = bool(lower) and bool(
        re.fullmatch(r"(?:</?(?:think|thought|assistant|analysis)>\\s*)+", lower)
    )
    refusal_terms = (
        "i can't", "i cannot", "unable to assist", "抱歉", "無法協助", "不能協助"
    )
    repeated = False
    if len(prediction) > 200:
        chunks = [x for x in re.split(r"[\r\n]+", prediction) if len(x.strip()) > 20]
        repeated = bool(chunks and Counter(x.strip() for x in chunks).most_common(1)[0][1] >= 5)
    return {
        "empty": not bool(prediction.strip()),
        "control_only": control_only,
        "prompt_echo": prompt in prediction,
        "refusal": any(term in lower for term in refusal_terms),
        "fixed_answer": False,
        "formatting": bool(re.search(r"```(?:markdown|md)?\\s*$", lower)),
        "repetition": repeated,
        "unrelated": False,
        "truncated": response.get("done_reason") == "length"
        or (
            isinstance(response.get("eval_count"), int)
            and response["eval_count"] >= num_predict
        ),
        "thinking_pollution": bool(response.get("thinking")) or "<think>" in lower,
    }


def infer_page(
    settings: Settings, model: dict[str, Any], image: pathlib.Path
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    inf = settings.raw["inference"]
    payload = {
        "model": model["ollama_tag"],
        "prompt": inf["prompt"],
        "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        "stream": False,
        "think": False,
        "keep_alive": inf["keep_alive"],
        "options": inf["options"],
    }
    errors: list[dict[str, Any]] = []
    for attempt in range(1, inf["max_attempts"] + 1):
        started = utc_now()
        began = time.monotonic()
        try:
            status, response = api_json(
                "POST",
                inf["endpoint"] + inf["api"],
                payload,
                timeout=inf["timeout_seconds"],
            )
            if response.get("done") is not True:
                raise RuntimeError(
                    "Ollama returned an incomplete non-stream response "
                    f"(done={response.get('done')!r}, keys={sorted(response)})"
                )
            response["_http_status"] = status
            response["_attempt"] = attempt
            response["_client_started_at"] = started
            response["_client_ended_at"] = utc_now()
            response["_client_elapsed_seconds"] = round(time.monotonic() - began, 6)
            return response, errors
        except Exception as exc:
            errors.append({"attempt": attempt, "at": utc_now(), "error": str(exc)})
            if attempt < inf["max_attempts"]:
                time.sleep(inf["backoff_seconds"][attempt - 1])
    return None, errors


def model_output_dirs(settings: Settings, model_id: str) -> dict[str, pathlib.Path]:
    base = settings.output_dir / "models" / model_id
    return {
        "base": base,
        "raw": base / "raw",
        "predictions": base / "predictions",
        "metadata": base / "page_metadata",
        "evaluation": base / "official_evaluation",
    }


def run_model_pages(
    settings: Settings,
    db: CheckpointDB,
    model: dict[str, Any],
    pages: list[dict[str, Any]],
    stop_requested: dict[str, bool],
) -> dict[str, Any]:
    dirs = model_output_dirs(settings, model["id"])
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    stats = {
        "model_id": model["id"],
        "success": 0,
        "failed": 0,
        "model_anomaly_pages": 0,
        "resumed": 0,
        "started_at": utc_now(),
    }
    predictions_seen: dict[str, str] = {}
    for page in pages:
        if stop_requested["value"]:
            break
        image = pathlib.Path(page["image_path"])
        image_hash = page.get("sha256") or sha256_file(image)
        page_id = page["page_id"]
        if db.reusable(settings, model["id"], page_id, image_hash):
            stats["resumed"] += 1
            continue
        started_at = utc_now()
        began = time.monotonic()
        response, errors = infer_page(settings, model, image)
        pred_name = prediction_filename(page["filename"])
        raw_path = dirs["raw"] / f"{page_id}.json"
        pred_path = dirs["predictions"] / pred_name
        metadata_path = dirs["metadata"] / f"{page_id}.json"
        if response is None:
            record = {
                "model_id": model["id"], "page_id": page_id, "filename": page["filename"],
                "status": "FAILED", "image_sha256": image_hash, "attempts": len(errors),
                "error_history_json": canonical_json(errors), "started_at": started_at,
                "ended_at": utc_now(), "elapsed_seconds": time.monotonic() - began,
            }
            db.save_result(settings, record)
            write_json(metadata_path, record)
            stats["failed"] += 1
            continue
        prediction, postprocess = postprocess_response(response)
        flags = detect_anomalies(
            prediction, response, settings.raw["inference"]["prompt"],
            settings.raw["inference"]["options"]["num_predict"],
        )
        # A returned response is a scoreable outcome of the fixed shared
        # configuration. Flags describe model behavior; transport/runtime
        # failures remain the only FAILED records.
        status = "SUCCESS"
        if any(flags.values()):
            stats["model_anomaly_pages"] += 1
        raw_doc = {
            "run_signature": settings.run_signature,
            "model_id": model["id"],
            "page_id": page_id,
            "filename": page["filename"],
            "image_sha256": image_hash,
            "request": {
                "model": model["ollama_tag"],
                "prompt_hash": settings.prompt_hash,
                "options_hash": settings.options_hash,
                "options": settings.raw["inference"]["options"],
                "think": False,
                "stream": False,
            },
            "response": response,
            "error_history": errors,
            "postprocess": postprocess,
            "anomaly_flags": flags,
        }
        write_json(raw_path, raw_doc)
        atomic_write(pred_path, prediction)
        metadata = {
            "page": page,
            "model": model,
            "status": status,
            "anomaly_flags": flags,
            "postprocess": postprocess,
            "raw_path": str(raw_path),
            "prediction_path": str(pred_path),
            "prediction_sha256": sha256_file(pred_path),
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_seconds": response["_client_elapsed_seconds"],
            "attempts": response["_attempt"],
            "error_history": errors,
        }
        write_json(metadata_path, metadata)
        record = {
            "model_id": model["id"], "page_id": page_id, "filename": page["filename"],
            "status": status, "image_sha256": image_hash, "raw_path": str(raw_path),
            "raw_sha256": sha256_file(raw_path), "prediction_path": str(pred_path),
            "prediction_sha256": sha256_file(pred_path), "response_json": canonical_json(response),
            "attempts": response["_attempt"], "error_history_json": canonical_json(errors),
            "started_at": started_at, "ended_at": utc_now(),
            "elapsed_seconds": response["_client_elapsed_seconds"],
            "eval_count": response.get("eval_count"), "prompt_eval_count": response.get("prompt_eval_count"),
            "done_reason": response.get("done_reason"), "load_duration_ns": response.get("load_duration"),
            "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
            "eval_duration_ns": response.get("eval_duration"),
            "total_duration_ns": response.get("total_duration"),
        }
        db.save_result(settings, record)
        stats["success"] += 1
        predictions_seen[page_id] = prediction
    stats["ended_at"] = utc_now()
    return stats


def warmup_and_gpu_check(
    settings: Settings, model: dict[str, Any], page: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    response, retries = infer_page(settings, model, pathlib.Path(page["image_path"]))
    ps = run_command(["ollama", "ps"], timeout=30)
    processor = parse_processor(ps["stdout"], model["ollama_tag"])
    report = {
        "model_id": model["id"], "tag": model["ollama_tag"], "response_received": response is not None,
        "retries": retries, "ollama_ps": ps, "processor": processor, "at": utc_now(),
    }
    if response is None:
        errors.append(f"{model['id']} warm-up failed")
    if processor != "100% GPU":
        errors.append(f"{model['id']} PROCESSOR is not exactly 100% GPU: {processor!r}")
    return report, errors


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def levenshtein(a: str, b: str) -> int:
    """Exact Unicode Levenshtein distance using Myers' bit-vector algorithm.

    Python's arbitrary-width integers make this exact for long document
    strings while avoiding the O(len(a) * len(b)) Python-level inner loop that
    is prohibitively slow on model runaway outputs.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Use the shorter input as the bit-vector pattern.
    if len(a) > len(b):
        a, b = b, a
    width = len(a)
    full_mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    char_masks: dict[str, int] = {}
    for index, char in enumerate(a):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)
    positive = full_mask
    negative = 0
    score = width
    for char in b:
        equal = char_masks.get(char, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(xh | positive)
        mh = positive & xh
        if ph & high_bit:
            score += 1
        elif mh & high_bit:
            score -= 1
        ph = ((ph << 1) | 1) & full_mask
        mh = (mh << 1) & full_mask
        positive = (mh | ~(xv | ph)) & full_mask
        negative = ph & xv
    return score


def supplementary_metrics(gt: str, pred: str) -> dict[str, Any]:
    gt_n, pred_n = normalize_text(gt), normalize_text(pred)
    denominator = max(len(gt_n), len(pred_n))
    distance = levenshtein(gt_n, pred_n)
    similarity = 1.0 if denominator == 0 else 1 - distance / denominator
    anls = similarity if similarity >= 0.5 else 0.0
    gt_count, pred_count = Counter(gt_n), Counter(pred_n)
    overlap = sum((gt_count & pred_count).values())
    precision = overlap / len(pred_n) if pred_n else (1.0 if not gt_n else 0.0)
    recall = overlap / len(gt_n) if gt_n else (1.0 if not pred_n else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "em": float(gt_n == pred_n),
        "cm": float(gt_n in pred_n),
        "anls": anls,
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": f1,
        "gt_characters": len(gt_n),
        "prediction_characters": len(pred_n),
        "edit_distance": distance,
        "empty_gt": not bool(gt_n),
        "empty_prediction": not bool(pred_n),
    }


def canonical_visible_text(page: dict[str, Any]) -> str:
    blocks: list[tuple[float, str]] = []
    for offset, block in enumerate(page.get("layout_dets", [])):
        if block.get("ignore"):
            continue
        category = block.get("category_type")
        if category == "table":
            value = block.get("html") or block.get("latex") or block.get("text") or ""
        elif category in {"equation_isolated", "equation_inline"}:
            value = block.get("latex") or block.get("text") or ""
        else:
            value = block.get("text") or block.get("latex") or ""
        if value:
            order = block.get("order")
            try:
                key = float(order)
            except (TypeError, ValueError):
                key = float(10**9 + offset)
            blocks.append((key, str(value)))
    return "\n\n".join(value for _, value in sorted(blocks, key=lambda x: x[0]))


def aggregate_supplementary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "denominator": 0, "unmatched_count": 0, "empty_count": 0,
            "macro": {}, "micro": {}, "method": "diagnostic_page_flattened_visible_text",
        }
    names = ["em", "cm", "anls", "character_f1"]
    macro = {name: statistics.fmean(row[name] for row in rows) for name in names}
    total_weight = sum(max(row["gt_characters"], 1) for row in rows)
    micro = {
        name: sum(row[name] * max(row["gt_characters"], 1) for row in rows) / total_weight
        for name in names
    }
    return {
        "denominator": len(rows),
        "unmatched_count": sum(bool(row.get("unmatched")) for row in rows),
        "empty_count": sum(bool(row["empty_prediction"]) for row in rows),
        "macro": macro,
        "micro": micro,
        "method": "diagnostic_page_flattened_visible_text",
        "official_match_pairs_available": False,
    }


def metrics_from_official_text_pairs(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt an explicit official text-pair artifact without inventing matching.

    Accepted stable keys are gt/pred or gt_text/pred_text. Missing prediction is
    treated as an unmatched GT. Pair records without a GT field are rejected.
    """
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        if "gt" in pair:
            gt = pair["gt"]
        elif "gt_text" in pair:
            gt = pair["gt_text"]
        else:
            raise ValueError(f"official pair {index} has no recognized GT text field")
        pred_present = "pred" in pair or "pred_text" in pair
        pred = pair.get("pred", pair.get("pred_text", ""))
        if not isinstance(gt, str) or not isinstance(pred, str):
            raise ValueError(f"official pair {index} text fields must be strings")
        row = supplementary_metrics(gt, pred)
        row.update(
            {
                "pair_index": index,
                "page_id": pair.get("page_id"),
                "unmatched": not pred_present or bool(pair.get("unmatched")),
            }
        )
        rows.append(row)
    aggregate = aggregate_supplementary(rows)
    aggregate["method"] = "official_v1_6_text_matching_pairs"
    aggregate["official_match_pairs_available"] = True
    return rows, aggregate


def official_config(settings: Settings, gt_container_path: str = "./gt/OmniDocBench.json") -> str:
    ev = settings.raw["evaluator"]
    return f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: {ev['cdm_workers']}
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {ev['teds_workers']}
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {gt_container_path}
    prediction:
      data_path: ./data_md/predictions
    match_method: {ev['match_method']}
    match_workers: {ev['match_workers']}
    quick_match_truncated_timeout_sec: {ev['quick_match_truncated_timeout_sec']}
    match_timeout_sec: {ev['match_timeout_sec']}
    timeout_fallback_max_chunk_span: {ev['timeout_fallback_max_chunk_span']}
    timeout_fallback_order_penalty: {ev['timeout_fallback_order_penalty']}
"""


def run_official_evaluator(
    settings: Settings, model_id: str, gt_path: pathlib.Path
) -> dict[str, Any]:
    dirs = model_output_dirs(settings, model_id)
    result_dir = dirs["evaluation"]
    result_dir.mkdir(parents=True, exist_ok=True)
    config_path = settings.output_dir / "immutable_config" / f"{model_id}_official.yaml"
    atomic_write(config_path, official_config(settings))
    ev = settings.raw["evaluator"]
    image_ref = f"{ev['docker_image']}@{ev['docker_digest']}"
    command = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,src={gt_path},dst=/workspace/gt/OmniDocBench.json,readonly",
        "--mount", f"type=bind,src={dirs['predictions']},dst=/workspace/data_md/predictions,readonly",
        "--mount", f"type=bind,src={result_dir},dst=/workspace/result",
        "--mount", f"type=bind,src={config_path},dst=/workspace/configs/run.yaml,readonly",
        image_ref, "--config", "configs/run.yaml",
    ]
    started_at = utc_now()
    result = run_command(command, timeout=7200)
    result.update({"started_at": started_at, "ended_at": utc_now(), "config_path": str(config_path)})
    write_json(result_dir / "runner_invocation.json", result)
    return result


def parse_official_results(result_dir: pathlib.Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {"files": {}, "missing_expected": []}
    expected_fragments = ["metric_result", "text_block_result", "display_formula_result", "table_result", "reading_order_result"]
    for path in sorted(result_dir.glob("*.json")):
        try:
            parsed["files"][path.name] = read_json(path)
        except Exception as exc:
            parsed["files"][path.name] = {"parse_error": str(exc)}
    names = list(parsed["files"])
    parsed["missing_expected"] = [
        fragment for fragment in expected_fragments if not any(fragment in name for name in names)
    ]
    parsed["status"] = "PASS" if not parsed["missing_expected"] else "FAIL"
    return parsed


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write(path, "")
        return
    fields = sorted({key for row in rows for key in row})
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def safe_preview(markdown: str, limit: int = 1200) -> str:
    return html.escape(markdown[:limit])


def build_report(settings: Settings, *, smoke: bool = False) -> pathlib.Path:
    output = settings.output_dir
    preflight = read_json(output / "preflight_report.json") if (output / "preflight_report.json").exists() else {}
    db_path = output / "results.sqlite"
    rows: list[dict[str, Any]] = []
    inference_signature = settings.run_signature
    if db_path.exists():
        db = CheckpointDB(db_path)
        rows = db.rows(settings)
        if not rows:
            stored = db.stored_signatures()
            if len(stored) != 1:
                db.close()
                raise RuntimeError(
                    "report rebuild requires exactly one immutable inference signature "
                    f"when scorer code changed; found {stored}"
                )
            inference_signature = stored[0]
            rows = db.rows_for_signature(inference_signature)
        db.close()
    page_rows = []
    gt_by_name: dict[str, dict[str, Any]] = {}
    if settings.gt_path.exists():
        gt_by_name = {
            pathlib.Path(page["page_info"]["image_path"]).name: page for page in load_dataset(settings)
        }
    thumbs = output / "thumbnails"
    thumbs.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = row["filename"]
        model_id = row["model_id"]
        pred = pathlib.Path(row["prediction_path"]).read_text(encoding="utf-8") if row["prediction_path"] else ""
        raw = pathlib.Path(row["raw_path"]).read_text(encoding="utf-8") if row["raw_path"] else ""
        if sha256_file(pathlib.Path(row["prediction_path"])) != row["prediction_sha256"]:
            raise RuntimeError(f"prediction hash mismatch while rebuilding report: {row['prediction_path']}")
        if sha256_file(pathlib.Path(row["raw_path"])) != row["raw_sha256"]:
            raise RuntimeError(f"raw response hash mismatch while rebuilding report: {row['raw_path']}")
        gt_page = gt_by_name.get(filename, {})
        gt_text = canonical_visible_text(gt_page)
        supplemental = supplementary_metrics(gt_text, pred)
        source_image = settings.image_dir / filename
        thumb_name = filename
        thumb_path = thumbs / thumb_name
        if source_image.exists() and not thumb_path.exists():
            shutil.copy2(source_image, thumb_path)
        response = json.loads(row["response_json"]) if row["response_json"] else {}
        raw_doc = json.loads(raw) if raw else {}
        flags = raw_doc.get("anomaly_flags", {})
        page_rows.append(
            {
                **row,
                "prediction": pred,
                "raw": raw,
                "gt": gt_text,
                "attributes": gt_page.get("page_info", {}).get("page_attribute", {}),
                "supplementary": supplemental,
                "flags": flags,
                "thumbnail": f"thumbnails/{thumb_name}" if thumb_path.exists() else None,
                "eval_count": response.get("eval_count", row["eval_count"]),
            }
        )
    duplicate_predictions: dict[str, set[str]] = {}
    for model_id in {row["model_id"] for row in page_rows}:
        counts = Counter(
            row["prediction"]
            for row in page_rows
            if row["model_id"] == model_id and row["prediction"].strip()
        )
        duplicate_predictions[model_id] = {
            value for value, count in counts.items() if count >= 3
        }
    for row in page_rows:
        if row["prediction"] in duplicate_predictions.get(row["model_id"], set()):
            row["flags"]["fixed_answer"] = True
    long_rows = []
    for row in page_rows:
        long_rows.append(
            {
                "model_id": row["model_id"], "page_id": row["page_id"], "filename": row["filename"],
                "status": row["status"], "prediction": row["prediction"],
                "raw_path": row["raw_path"], "prediction_path": row["prediction_path"],
                **{f"flag_{k}": v for k, v in row["flags"].items()},
                **{f"supplementary_{k}": v for k, v in row["supplementary"].items() if not isinstance(v, (dict, list))},
            }
        )
    write_csv(output / "predictions_long.csv", long_rows)
    by_page: dict[str, dict[str, Any]] = {}
    for row in page_rows:
        wide = by_page.setdefault(row["page_id"], {"page_id": row["page_id"], "filename": row["filename"]})
        wide[f"{row['model_id']}_status"] = row["status"]
        wide[f"{row['model_id']}_prediction"] = row["prediction"]
    write_csv(output / "pages_wide.csv", list(by_page.values()))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in page_rows:
        groups.setdefault(row["model_id"], []).append(row["supplementary"])
    summary_supp = {model: aggregate_supplementary(values) for model, values in groups.items()}
    write_json(output / "summary_supplementary.json", summary_supp)
    summary_csv = []
    for model, summary in summary_supp.items():
        summary_csv.append(
            {"model_id": model, "denominator": summary["denominator"], **summary["macro"], **{f"micro_{k}": v for k, v in summary["micro"].items()}}
        )
    write_csv(output / "summary_supplementary.csv", summary_csv)
    official = {}
    for model in [x["id"] for x in settings.models]:
        result_dir = model_output_dirs(settings, model)["evaluation"]
        official[model] = parse_official_results(result_dir) if result_dir.exists() else {"status": "NOT_RUN"}
    runtime_summary: dict[str, Any] = {}
    for model_id in {row["model_id"] for row in page_rows}:
        model_rows = [row for row in page_rows if row["model_id"] == model_id]
        successful = [
            row
            for row in model_rows
            if row["status"] == "SUCCESS" and isinstance(row.get("elapsed_seconds"), (int, float))
        ]
        elapsed = [float(row["elapsed_seconds"]) for row in successful]
        throughputs = [
            float(row["eval_count"]) / (float(row["eval_duration_ns"]) / 1_000_000_000)
            for row in successful
            if isinstance(row.get("eval_count"), int)
            and isinstance(row.get("eval_duration_ns"), int)
            and row["eval_duration_ns"] > 0
        ]
        sorted_elapsed = sorted(elapsed)
        p95_index = max(0, min(len(sorted_elapsed) - 1, int(0.95 * len(sorted_elapsed)) - 1))
        runtime_summary[model_id] = {
            "denominator": len(successful),
            "latency_mean_seconds": statistics.fmean(elapsed) if elapsed else None,
            "latency_median_seconds": statistics.median(elapsed) if elapsed else None,
            "latency_p95_seconds": sorted_elapsed[p95_index] if elapsed else None,
            "tokens_per_second_mean": statistics.fmean(throughputs) if throughputs else None,
            "error_count": sum(row["status"] != "SUCCESS" for row in model_rows),
            "retry_count": sum(max(0, int(row.get("attempts") or 0) - 1) for row in model_rows),
            "processor_requirement": "100% GPU",
        }
    write_json(output / "summary_official.json", official)
    write_csv(output / "summary_official.csv", [{"model_id": k, "status": v.get("status")} for k, v in official.items()])
    title = "OmniDocBench v1.6 Smoke Report" if smoke else "OmniDocBench v1.6 Report"
    state = preflight.get("state", "UNKNOWN")
    cards = []
    for row in page_rows:
        attr = html.escape(canonical_json(row["attributes"]))
        flags = html.escape(canonical_json(row["flags"]))
        image_tag = f'<img loading="lazy" src="{html.escape(row["thumbnail"])}" alt="page image">' if row["thumbnail"] else "<em>thumbnail unavailable</em>"
        cards.append(
            f"""<article class="page" data-model="{html.escape(row['model_id'])}" data-status="{html.escape(row['status'])}">
<h3>{html.escape(row['model_id'])} · {html.escape(row['filename'])}</h3>
<p>Status: <strong>{html.escape(row['status'])}</strong> · latency {row['elapsed_seconds']}s · tokens {row['eval_count']} · attempts {row['attempts']}</p>
<p>Attributes: <code>{attr}</code></p><p>Flags: <code>{flags}</code></p>
{image_tag}
<details><summary>Official GT canonical/annotations</summary><pre>{html.escape(row['gt'])}</pre></details>
<details><summary>Raw response JSON</summary><pre>{html.escape(row['raw'])}</pre></details>
<details open><summary>Markdown prediction (escaped safe preview)</summary><pre>{safe_preview(row['prediction'])}</pre></details>
<details><summary>Complete Markdown prediction</summary><pre>{html.escape(row['prediction'])}</pre></details>
<details><summary>Official evaluator page artifacts</summary><pre>{html.escape(json.dumps(official_page_artifacts(official.get(row['model_id'], {}), row['filename']), ensure_ascii=False, indent=2))}</pre></details>
<p>Diagnostic supplementary metrics: <code>{html.escape(canonical_json(row['supplementary']))}</code></p>
</article>"""
        )
    blockers = "".join(f"<li>{html.escape(str(item))}</li>" for item in preflight.get("blockers", []))
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:2rem;line-height:1.55}}
.state{{padding:1rem;border:2px solid #a33;background:#fee}} table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:.4rem}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1rem;max-height:35rem;overflow:auto}}
img{{max-width:420px;max-height:540px;border:1px solid #aaa}}.page{{border-top:2px solid #555;margin-top:2rem;padding-top:1rem}}
code{{overflow-wrap:anywhere}}details{{margin:.5rem 0}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="state"><strong>{html.escape(state)}</strong><ul>{blockers}</ul></div>
<h2>Version pin</h2><pre>{html.escape(json.dumps(settings.raw['benchmark'] | settings.raw['evaluator'], ensure_ascii=False, indent=2))}</pre>
<h2>Common prompt/options</h2><p>Prompt SHA-256: <code>{settings.prompt_hash}</code>; options SHA-256: <code>{settings.options_hash}</code></p>
<pre>{html.escape(settings.raw['inference']['prompt'])}</pre><pre>{html.escape(json.dumps(settings.raw['inference']['options'], ensure_ascii=False, indent=2))}</pre>
<h2>Preflight</h2><details><summary>Complete preflight JSON</summary><pre>{html.escape(json.dumps(preflight, ensure_ascii=False, indent=2))}</pre></details>
<h2>Official metrics (official leaderboard metrics)</h2><pre>{html.escape(json.dumps(official, ensure_ascii=False, indent=2))}</pre>
<h2>Supplementary OCR diagnostics (not official leaderboard metrics)</h2><pre>{html.escape(json.dumps(summary_supp, ensure_ascii=False, indent=2))}</pre>
<h2>Runtime (comparable only at 100% GPU and identical inputs/options)</h2><pre>{html.escape(json.dumps(runtime_summary, ensure_ascii=False, indent=2))}</pre>
<h2>Pages</h2>
<p><label>Search <input id="q" type="search"></label>
<label>Model <select id="model"><option value="">all</option>{''.join(f'<option>{html.escape(x["id"])}</option>' for x in settings.models)}</select></label>
<label>Status <select id="status"><option value="">all</option><option>SUCCESS</option><option>FAILED</option></select></label></p>
{''.join(cards) if cards else '<p>No inference pages were produced.</p>'}
<script>
function filterPages(){{
 const q=document.getElementById('q').value.toLowerCase(),m=document.getElementById('model').value,s=document.getElementById('status').value;
 document.querySelectorAll('.page').forEach(x=>{{const ok=(!q||x.textContent.toLowerCase().includes(q))&&(!m||x.dataset.model===m)&&(!s||x.dataset.status===s);x.hidden=!ok;}});
}}
document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',filterPages));
</script>
</body></html>"""
    report_path = output / ("smoke_report.html" if smoke else "report.html")
    atomic_write(report_path, document)
    report_json = {
        "state": state, "created_at": utc_now(), "run_id": settings.run_id,
        "run_signature": inference_signature,
        "report_code_hash": settings.code_hash,
        "blockers": preflight.get("blockers", []),
        "models": {model: {"pages": len(values), "supplementary": summary_supp.get(model)} for model, values in groups.items()},
        "official": official,
        "runtime": runtime_summary,
        "ai_page_review": {
            "status": "NOT_PERFORMED" if not page_rows else "REQUIRES_EXPLICIT_REVIEW",
            "reviewed_pages": 0,
            "note": "A human/AI must inspect all 40 raw responses before READY_FOR_FULL_RUN.",
        },
    }
    write_json(output / ("smoke_report.json" if smoke else "report.json"), report_json)
    return report_path


def html_preflight(report: dict[str, Any]) -> str:
    blockers = "".join(f"<li>{html.escape(str(x))}</li>" for x in report.get("blockers", []))
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>OmniDocBench preflight</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:auto;padding:2rem}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem}}.state{{border:2px solid #a33;padding:1rem}}</style>
</head><body><h1>OmniDocBench v1.6 preflight</h1><div class="state"><strong>{html.escape(report['state'])}</strong><ul>{blockers}</ul></div>
<p>UTC: {html.escape(report['ended_at'])}<br>Asia/Taipei: {html.escape(taipei_time(report['ended_at']))}</p>
<pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre></body></html>"""


def docker_preflight(settings: Settings) -> tuple[dict[str, Any], list[str]]:
    ev = settings.raw["evaluator"]
    errors: list[str] = []
    image = ev["docker_image"]
    inspect = run_command(["docker", "image", "inspect", image], timeout=60)
    report: dict[str, Any] = {"inspect": inspect}
    if inspect["returncode"] != 0:
        errors.append(f"required Docker image missing or inaccessible: {image}")
        report["status"] = "FAIL"
        return report, errors
    try:
        info = json.loads(inspect["stdout"])[0]
        repo_digests = info.get("RepoDigests", [])
        image_id = info.get("Id")
    except Exception as exc:
        errors.append(f"cannot parse Docker inspect: {exc}")
        repo_digests, image_id = [], None
    report.update({"image_id": image_id, "repo_digests": repo_digests})
    expected_ref = f"{image}@{ev['docker_digest']}"
    digest_matches = any(ref.endswith(f"@{ev['docker_digest']}") for ref in repo_digests)
    if not digest_matches or image_id != ev["docker_digest"]:
        errors.append(
            f"Docker digest mismatch: expected digest {ev['docker_digest']}, "
            f"got {repo_digests}/{image_id}"
        )
    verify_script = (
        "set -e; python --version; pdflatex --version | head -1; "
        "magick --version | head -1; gs --version; "
        "bash /workspace/script/verify_repro_runtime.sh; "
        "python /workspace/script/cdm_runtime_smoke.py; "
        f"sha256sum /workspace/{ev['supervisor_config_path']}; "
        "sha256sum /workspace/gt/OmniDocBench_v1.6_base.json; "
        "python - <<'PY'\nimport json\np='/workspace/gt/OmniDocBench_v1.6_base.json'\nprint('embedded_gt_pages',len(json.load(open(p))))\nPY"
    )
    runtime = run_command(
        ["docker", "run", "--rm", "--entrypoint", "bash", expected_ref, "-lc", verify_script],
        timeout=180,
    )
    report["runtime_verification"] = runtime
    if runtime["returncode"] != 0:
        errors.append("Docker runtime verification failed")
    if ev["supervisor_config_sha256"] not in runtime["stdout"]:
        errors.append("supervisor v1.6 reproduction config hash mismatch")
    embedded_match = re.search(r"embedded_gt_pages\s+(\d+)", runtime["stdout"])
    embedded_pages = int(embedded_match.group(1)) if embedded_match else None
    report["embedded_gt_pages"] = embedded_pages
    report["embedded_gt_usable_for_full_v1_6"] = embedded_pages == settings.raw["benchmark"]["expected_pages"]
    if embedded_pages != settings.raw["benchmark"]["expected_pages"]:
        report["warning"] = (
            "Image-embedded GT is not the 1,651-page v1.6 full set and must not be used. "
            "The externally pinned official GT is mounted read-only instead."
        )
    supervisor = (CONFIG_DIR / "supervisor_reproduction.yaml").read_text(encoding="utf-8")
    generated = official_config(settings)
    report["supervisor_config_copy_sha256"] = sha256_text(supervisor)
    report["generated_run_config_sha256"] = sha256_text(generated)
    report["config_diff"] = list(
        difflib.unified_diff(
            supervisor.splitlines(),
            generated.splitlines(),
            fromfile="supervisor_reproduction.yaml",
            tofile="generated_run.yaml",
            lineterm="",
        )
    )
    report["status"] = "PASS" if not errors else "FAIL"
    return report, errors


def model_preflight(settings: Settings) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    available, api_error = ollama_models(settings)
    report: dict[str, Any] = {"api_error": api_error, "available_models": available, "models": []}
    if api_error:
        errors.append(f"Ollama API unavailable: {api_error}")
        return report, errors, []
    by_name = {x.get("name") or x.get("model"): x for x in available}
    valid: list[dict[str, Any]] = []
    for spec in settings.models:
        item: dict[str, Any] = {"mapping": spec}
        tag = spec.get("ollama_tag")
        if not spec.get("supervisor_approved") or not tag:
            reason = spec.get("blocker") or f"{spec['logical_name']} is not supervisor-approved"
            errors.append(reason)
            item.update({"status": "BLOCKED", "error": reason})
            report["models"].append(item)
            continue
        listed = by_name.get(tag)
        if not listed:
            errors.append(f"exact approved Ollama tag missing: {tag}")
            item.update({"status": "BLOCKED", "error": "exact tag missing"})
            report["models"].append(item)
            continue
        show, show_error = ollama_show(settings, tag)
        compact_show = None
        if show:
            model_info = show.get("model_info", {})
            compact_show = {
                "details": show.get("details"),
                "capabilities": show.get("capabilities"),
                "parameters": show.get("parameters"),
                "model_info": {
                    key: value
                    for key, value in model_info.items()
                    if any(
                        token in key.lower()
                        for token in (
                            "architecture", "context", "embedding", "parameter",
                            "block_count", "vision", "projector", "quant",
                        )
                    )
                    and isinstance(value, (str, int, float, bool, type(None)))
                },
            }
        item.update({"listed": listed, "show": compact_show, "show_error": show_error})
        if show_error or not show:
            errors.append(f"cannot inspect exact model {tag}: {show_error}")
            item["status"] = "BLOCKED"
            report["models"].append(item)
            continue
        details = show.get("details", {})
        capabilities = capabilities_from_show(show)
        context_value = show.get("model_info", {}).get(f"{spec['expected_architecture']}.context_length")
        # Default floor matches inference.options.num_ctx in benchmark.json. A model
        # whose native context is genuinely below that (e.g. InternVL3.5's 40960) can
        # declare "expected_min_context" in its config_variants entry to acknowledge
        # the shortfall explicitly, rather than silently failing this check forever.
        min_context = spec.get("expected_min_context", 65536)
        checks = {
            "architecture": details.get("family") == spec["expected_architecture"],
            "quantization": details.get("quantization_level") == spec["expected_quantization"],
            "vision": "vision" in capabilities,
            "context_at_least_65536": isinstance(context_value, int) and context_value >= min_context,
        }
        item["context_length"] = context_value
        item["checks"] = checks
        item["digest"] = listed.get("digest")
        item["size"] = listed.get("size")
        item["capabilities"] = capabilities
        item["status"] = "PASS" if all(checks.values()) else "BLOCKED"
        if item["status"] != "PASS":
            errors.append(f"model metadata mismatch for {tag}: {checks}")
        else:
            valid.append(spec)
        report["models"].append(item)
    report["status"] = "PASS" if not errors and len(valid) == len(settings.models) else "FAIL"
    return report, errors, valid


def write_immutable_inputs(settings: Settings) -> None:
    target = settings.output_dir / "immutable_config"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "benchmark.json", "models.yaml", "smoke_pages.json",
        "supervisor_reproduction.yaml", "official_v1_6_end2end.yaml",
    ):
        source = CONFIG_DIR / name
        destination = target / name
        content = source.read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise RuntimeError(f"immutable config changed in existing run: {destination}")
        atomic_write(destination, content)
    atomic_write(target / "official_run.yaml", official_config(settings))
    source_target = target / "runner_source"
    source_files = [
        *sorted((ROOT / "omnidocbench").glob("*.py")),
        ROOT / "scripts" / "fetch_dataset.py",
        *sorted(ROOT.glob("*.sh")),
    ]
    for source in source_files:
        relative = source.relative_to(ROOT)
        atomic_write(source_target / relative, source.read_bytes())
    write_json(
        target / "hashes.json",
        {
            name: sha256_file(target / name)
            for name in (
                "benchmark.json", "models.yaml", "smoke_pages.json",
                "supervisor_reproduction.yaml", "official_v1_6_end2end.yaml",
                "official_run.yaml",
            )
        },
    )


def preflight(settings: Settings, *, hash_images: bool = True) -> dict[str, Any]:
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_immutable_inputs(settings)
    started_at = utc_now()
    blockers: list[str] = []
    identity = run_command(["id"])
    groups = identity["stdout"]
    if "ocfai" not in groups:
        blockers.append("current login session does not include ocfai group")
    if "docker" not in groups:
        blockers.append("current login session does not include docker group")
    path_checks = {}
    for key in ("dataset_root", "scratch_root", "cache_root", "output_root"):
        path = pathlib.Path(settings.raw["paths"][key])
        parent = path if path.exists() else path.parent
        path_checks[key] = {
            "path": str(path), "exists": path.exists(),
            "parent": str(parent), "parent_writable": os.access(parent, os.W_OK),
        }
        if key == "dataset_root" and not path.exists():
            blockers.append(f"required dataset path missing: {path}")
        if key != "dataset_root" and not path_checks[key]["parent_writable"]:
            blockers.append(f"required path is not writable: {path}")
    dataset, dataset_errors = validate_dataset(settings, hash_images=hash_images)
    blockers.extend(dataset_errors)
    if dataset.get("manifest"):
        make_dataset_manifest(settings, dataset, output / "dataset_manifest.json")
    else:
        write_json(output / "dataset_manifest.json", {"benchmark": settings.raw["benchmark"], "pages": [], "status": dataset.get("status")})
    docker, docker_errors = docker_preflight(settings)
    blockers.extend(docker_errors)
    models, model_errors, valid_models = model_preflight(settings)
    blockers.extend(model_errors)
    system = {
        "id": identity,
        "os_release": pathlib.Path("/etc/os-release").read_text(encoding="utf-8"),
        "python": run_command(["python3", "--version"]),
        "ollama_version": run_command(["ollama", "--version"]),
        "ollama_list": run_command(["ollama", "list"]),
        "ollama_service_environment": run_command(
            ["systemctl", "show", "ollama", "--property=Environment", "--no-pager"]
        ),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "disk": run_command(["df", "-h", "/", "/mnt/nvme", "/opt/ocf-ai"]),
        "git": git_metadata(),
    }
    if system["nvidia_smi"]["returncode"] != 0:
        blockers.append("nvidia-smi failed")
    service_env = system["ollama_service_environment"]
    if service_env["returncode"] != 0:
        blockers.append("cannot verify Ollama service environment / fp16 KV cache baseline")
    else:
        match = re.search(r"(?:^|\s)OLLAMA_KV_CACHE_TYPE=([^\s\"]+)", service_env["stdout"])
        actual_kv = match.group(1) if match else "fp16 (unset default)"
        system["ollama_kv_cache_type"] = actual_kv
        if match and match.group(1).lower() not in {"f16", "fp16"}:
            blockers.append(f"Ollama KV cache is not fp16 baseline: {match.group(1)}")
    report = {
        "state": "PASS" if not blockers else "BLOCKED",
        "run_id": settings.run_id,
        "run_signature": settings.run_signature,
        "started_at": started_at,
        "ended_at": utc_now(),
        "prompt_hash": settings.prompt_hash,
        "options_hash": settings.options_hash,
        "code_hash": settings.code_hash,
        "blockers": list(dict.fromkeys(blockers)),
        "paths": path_checks,
        "dataset": dataset,
        "docker": docker,
        "models": models,
        "valid_model_count": len(valid_models),
        "system": system,
    }
    write_json(output / "preflight_report.json", report)
    atomic_write(output / "preflight_report.html", html_preflight(report))
    write_json(
        output / "metadata.json",
        {
            "run_id": settings.run_id, "status": report["state"], "started_at": started_at,
            "ended_at": report["ended_at"], "logical_models": settings.models,
            "prompt": settings.raw["inference"]["prompt"], "prompt_hash": settings.prompt_hash,
            "options": settings.raw["inference"]["options"], "options_hash": settings.options_hash,
            "code_hash": settings.code_hash,
            "kv_cache_type": settings.raw["inference"]["kv_cache_type"],
            "dataset": settings.raw["benchmark"], "evaluator": settings.raw["evaluator"],
            "resolved_models": models,
            "input_source": "official_page_image", "rasterize_dpi": None,
            "note": "Official page images are used without re-rasterization or enhancement.",
            "timing": {"started_at": started_at, "ended_at": report["ended_at"]},
            "git": system["git"],
        },
    )
    write_json(
        output / "status.json",
        {"state": report["state"], "run_id": settings.run_id, "updated_at": utc_now(), "blockers": report["blockers"]},
    )
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        log_dir / "preflight.log",
        "\n".join(
            [
                f"started_at={started_at}",
                f"ended_at={report['ended_at']}",
                f"state={report['state']}",
                f"run_id={settings.run_id}",
                f"run_signature={settings.run_signature}",
                *(f"blocker={item}" for item in report["blockers"]),
            ]
        )
        + "\n",
    )
    atomic_write(
        log_dir / "error.log",
        "".join(f"{report['ended_at']} BLOCKER {item}\n" for item in report["blockers"]),
    )
    return report


def official_page_artifacts(value: Any, filename: str, *, limit: int = 100) -> list[Any]:
    """Return evaluator records explicitly mentioning this filename or stem."""
    stem = pathlib.Path(filename).stem
    matches: list[Any] = []

    def walk(node: Any) -> None:
        if len(matches) >= limit:
            return
        if isinstance(node, dict):
            serialized = canonical_json(node)
            if filename in serialized or stem in serialized:
                # Prefer the smallest matching record and do not recursively
                # duplicate its descendants.
                matches.append(node)
                return
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return matches


def create_smoke_gt(settings: Settings, selected: list[dict[str, Any]]) -> pathlib.Path:
    selected_names = {x["filename"] for x in selected}
    pages = [
        page for page in load_dataset(settings)
        if pathlib.Path(page["page_info"]["image_path"]).name in selected_names
    ]
    if len(pages) != len(selected):
        raise RuntimeError("cannot build exact smoke GT subset")
    path = settings.output_dir / "smoke_gt" / "OmniDocBench_v1.6_smoke20.json"
    write_json(path, pages)
    return path


def install_signal_handlers(stop_requested: dict[str, bool], db: CheckpointDB, settings: Settings) -> None:
    def handler(signum: int, _frame: Any) -> None:
        stop_requested["value"] = True
        db.event(settings, "WARNING", "signal received; stopping after current atomic page", {"signal": signum})
        write_json(
            settings.output_dir / "status.json",
            {"state": "INTERRUPTING", "signal": signum, "updated_at": utc_now()},
        )
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def export_raw_jsonl(settings: Settings, db: CheckpointDB) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in db.rows(settings):
        if row["raw_path"] and pathlib.Path(row["raw_path"]).exists():
            grouped.setdefault(row["model_id"], []).append(read_json(pathlib.Path(row["raw_path"])))
    for model_id, docs in grouped.items():
        atomic_write(
            settings.output_dir / "models" / model_id / "raw_responses.jsonl",
            "".join(canonical_json(doc) + "\n" for doc in docs),
        )
