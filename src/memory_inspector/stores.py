from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any

from memory_inspector.types import RetrievalRecord, ScoredResult


def _record_to_row(record: RetrievalRecord) -> tuple[Any, ...]:
    results_json = json.dumps(
        [
            {
                "content": r.content,
                "score": r.score,
                "rank": r.rank,
                "metadata": r.metadata,
                "document_id": r.document_id,
            }
            for r in record.results
        ]
    )
    return (
        record.id,
        record.timestamp.isoformat(),
        record.query,
        results_json,
        record.top_k,
        record.latency_ms,
        json.dumps(record.metadata),
    )


def _row_to_record(row: tuple[Any, ...]) -> RetrievalRecord:
    from datetime import datetime, timezone

    record_id, timestamp_str, query, results_json, top_k, latency_ms, metadata_json = row
    results = [
        ScoredResult(
            content=r["content"],
            score=r["score"],
            rank=r["rank"],
            metadata=r.get("metadata", {}),
            document_id=r.get("document_id"),
        )
        for r in json.loads(results_json)
    ]
    return RetrievalRecord(
        id=record_id,
        timestamp=datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc),
        query=query,
        results=tuple(results),
        top_k=top_k,
        latency_ms=latency_ms,
        metadata=json.loads(metadata_json),
    )


class InMemoryStore:
    def __init__(self, max_records: int = 1000) -> None:
        self._records: deque[RetrievalRecord] = deque(maxlen=max_records)

    def save(self, record: RetrievalRecord) -> None:
        self._records.append(record)

    def list(self, limit: int = 20) -> list[RetrievalRecord]:
        records = list(self._records)
        return list(reversed(records))[:limit]

    def get(self, record_id: str) -> RetrievalRecord | None:
        for record in self._records:
            if record.id == record_id:
                return record
        return None

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS retrieval_records (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    query       TEXT NOT NULL,
    results     TEXT NOT NULL,
    top_k       INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    metadata    TEXT NOT NULL
)
"""


class SQLiteStore:
    def __init__(self, path: str = ".memory_inspector/traces.db") -> None:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def save(self, record: RetrievalRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO retrieval_records "
            "(id, timestamp, query, results, top_k, latency_ms, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            _record_to_row(record),
        )
        self._conn.commit()

    def list(self, limit: int = 20) -> list[RetrievalRecord]:
        cursor = self._conn.execute(
            "SELECT id, timestamp, query, results, top_k, latency_ms, metadata "
            "FROM retrieval_records ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_record(row) for row in cursor.fetchall()]

    def get(self, record_id: str) -> RetrievalRecord | None:
        cursor = self._conn.execute(
            "SELECT id, timestamp, query, results, top_k, latency_ms, metadata "
            "FROM retrieval_records WHERE id = ?",
            (record_id,),
        )
        row = cursor.fetchone()
        return _row_to_record(row) if row else None

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM retrieval_records")
        result = cursor.fetchone()
        return int(result[0])

    def clear(self) -> None:
        self._conn.execute("DELETE FROM retrieval_records")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
