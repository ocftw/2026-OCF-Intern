from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .util import utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS attempts (
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_signature TEXT NOT NULL,
  phase TEXT NOT NULL,
  model_id TEXT NOT NULL,
  sample_index INTEGER NOT NULL,
  attempt_number INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT NOT NULL,
  success INTEGER NOT NULL,
  request_json TEXT NOT NULL,
  response_json TEXT,
  http_status INTEGER,
  error_type TEXT,
  error_message TEXT,
  completion_validation_json TEXT NOT NULL DEFAULT '{}',
  latency_seconds REAL NOT NULL,
  UNIQUE(run_signature, phase, model_id, sample_index, attempt_number)
);
CREATE TABLE IF NOT EXISTS results (
  run_signature TEXT NOT NULL,
  phase TEXT NOT NULL,
  model_id TEXT NOT NULL,
  sample_index INTEGER NOT NULL,
  exact_tag TEXT NOT NULL,
  model_digest TEXT NOT NULL,
  image_relative_path TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  ground_truth TEXT NOT NULL,
  prediction_raw TEXT NOT NULL,
  prediction_primary TEXT NOT NULL,
  prediction_aligned TEXT NOT NULL,
  aligned_changes_json TEXT NOT NULL,
  primary_metrics_json TEXT NOT NULL,
  aligned_metrics_json TEXT NOT NULL,
  anomaly_flags_json TEXT NOT NULL,
  ollama_metadata_json TEXT NOT NULL,
  done_reason TEXT,
  prompt_eval_count INTEGER,
  eval_count INTEGER,
  completion_validation_json TEXT NOT NULL DEFAULT '{}',
  record_status TEXT NOT NULL DEFAULT 'completed',
  http_status INTEGER,
  error_type TEXT,
  error_message TEXT,
  latency_seconds REAL NOT NULL,
  attempt_count INTEGER NOT NULL,
  completed_at TEXT NOT NULL,
  PRIMARY KEY(run_signature, phase, model_id, sample_index)
);
CREATE INDEX IF NOT EXISTS idx_results_phase_model ON results(run_signature, phase, model_id);
"""


class Checkpoint:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._ensure_column(
            "attempts",
            "completion_validation_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            "results",
            "completion_validation_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column("results", "record_status", "TEXT NOT NULL DEFAULT 'completed'")
        self._ensure_column("results", "http_status", "INTEGER")
        self._ensure_column("results", "error_type", "TEXT")
        self._ensure_column("results", "error_message", "TEXT")
        self.db.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        existing = {
            str(row["name"])
            for row in self.db.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        self.db.close()

    def successful(self, signature: str, phase: str, model_id: str, index: int) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM results WHERE run_signature=? AND phase=? AND model_id=? AND sample_index=?",
            (signature, phase, model_id, index),
        ).fetchone()
        return row is not None

    def next_attempt(self, signature: str, phase: str, model_id: str, index: int) -> int:
        row = self.db.execute(
            "SELECT COALESCE(MAX(attempt_number),0)+1 n FROM attempts WHERE run_signature=? AND phase=? AND model_id=? AND sample_index=?",
            (signature, phase, model_id, index),
        ).fetchone()
        return int(row["n"])

    def record_attempt(self, row: dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                """INSERT INTO attempts
                (run_signature,phase,model_id,sample_index,attempt_number,started_at,ended_at,success,
                 request_json,response_json,http_status,error_type,error_message,completion_validation_json,
                 latency_seconds)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["run_signature"], row["phase"], row["model_id"], row["sample_index"],
                    row["attempt_number"], row["started_at"], row["ended_at"], int(row["success"]),
                    json.dumps(row["request"], ensure_ascii=False, sort_keys=True),
                    json.dumps(row.get("response"), ensure_ascii=False, sort_keys=True) if row.get("response") is not None else None,
                    row.get("http_status"), row.get("error_type"), row.get("error_message"),
                    json.dumps(row.get("completion_validation") or {}, ensure_ascii=False, sort_keys=True),
                    row["latency_seconds"],
                ),
            )

    def record_success(self, row: dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                """INSERT INTO results
                (run_signature,phase,model_id,sample_index,exact_tag,model_digest,image_relative_path,
                 image_sha256,ground_truth,prediction_raw,prediction_primary,prediction_aligned,
                 aligned_changes_json,primary_metrics_json,aligned_metrics_json,anomaly_flags_json,
                 ollama_metadata_json,done_reason,prompt_eval_count,eval_count,latency_seconds,
                 completion_validation_json,record_status,http_status,error_type,error_message,
                 attempt_count,completed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_signature,phase,model_id,sample_index) DO NOTHING""",
                (
                    row["run_signature"], row["phase"], row["model_id"], row["sample_index"],
                    row["exact_tag"], row["model_digest"], row["image_relative_path"], row["image_sha256"],
                    row["ground_truth"], row["prediction_raw"], row["prediction_primary"],
                    row["prediction_aligned"], json.dumps(row["aligned_changes"], ensure_ascii=False),
                    json.dumps(row["primary_metrics"], ensure_ascii=False, sort_keys=True),
                    json.dumps(row["aligned_metrics"], ensure_ascii=False, sort_keys=True),
                    json.dumps(row["anomaly_flags"], ensure_ascii=False, sort_keys=True),
                    json.dumps(row["ollama_metadata"], ensure_ascii=False, sort_keys=True),
                    row.get("done_reason"), row.get("prompt_eval_count"), row.get("eval_count"),
                    row["latency_seconds"],
                    json.dumps(row.get("completion_validation") or {}, ensure_ascii=False, sort_keys=True),
                    row.get("record_status", "completed"), row.get("http_status"),
                    row.get("error_type"), row.get("error_message"),
                    row["attempt_count"], utc_now(),
                ),
            )

    def results(self, signature: str, phase: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM results WHERE run_signature=?"
        args: list[Any] = [signature]
        if phase:
            sql += " AND phase=?"
            args.append(phase)
        sql += " ORDER BY sample_index, model_id"
        rows = []
        for raw in self.db.execute(sql, args):
            row = dict(raw)
            for field in (
                "aligned_changes_json",
                "primary_metrics_json",
                "aligned_metrics_json",
                "anomaly_flags_json",
                "ollama_metadata_json",
                "completion_validation_json",
            ):
                row[field.removesuffix("_json")] = json.loads(row.pop(field))
            rows.append(row)
        return rows

    def latest_attempts(self, signature: str, phase: str) -> list[dict[str, Any]]:
        rows = []
        query = """
        SELECT a.* FROM attempts a
        JOIN (
          SELECT model_id, sample_index, MAX(attempt_number) AS maximum
          FROM attempts WHERE run_signature=? AND phase=?
          GROUP BY model_id, sample_index
        ) latest
        ON a.model_id=latest.model_id AND a.sample_index=latest.sample_index
           AND a.attempt_number=latest.maximum
        WHERE a.run_signature=? AND a.phase=?
        ORDER BY a.sample_index, a.model_id
        """
        for raw in self.db.execute(query, (signature, phase, signature, phase)):
            row = dict(raw)
            row["request"] = json.loads(row.pop("request_json"))
            response_json = row.pop("response_json")
            row["response"] = json.loads(response_json) if response_json else None
            row["completion_validation"] = json.loads(row.pop("completion_validation_json"))
            rows.append(row)
        return rows

    def counts(self, signature: str, phase: str) -> dict[str, int]:
        success = self.db.execute(
            "SELECT COUNT(*) n FROM results WHERE run_signature=? AND phase=?", (signature, phase)
        ).fetchone()["n"]
        failed = self.db.execute(
            "SELECT COUNT(DISTINCT a.model_id || ':' || a.sample_index) n FROM attempts a "
            "WHERE run_signature=? AND phase=? AND NOT EXISTS "
            "(SELECT 1 FROM results r WHERE r.run_signature=a.run_signature AND r.phase=a.phase AND r.model_id=a.model_id AND r.sample_index=a.sample_index)",
            (signature, phase),
        ).fetchone()["n"]
        return {"successful_results": int(success), "unresolved_failed_attempts": int(failed)}
