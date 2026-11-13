"""SQLite-backed mind persistence: full snapshot save / load."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from ..core.builder import MindBuilder
from ..core.mind import Mind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mind_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    data TEXT NOT NULL
);
"""


class SqliteMindStore:
    """Saves complete mind snapshots (facts, memories, goals, emotions,
    personality, neural weights, Q-table) into a SQLite database."""

    def __init__(self, path: str = "cognify_minds.db") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ API
    def save(self, mind: Mind, name: Optional[str] = None) -> int:
        name = name or mind.name
        data = json.dumps(mind.to_dict(), ensure_ascii=False)
        import time
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO mind_snapshots (name, created, updated, data) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET updated=excluded.updated, data=excluded.data",
            (name, now, now, data))
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def load(self, name: str) -> Mind:
        row = self._conn.execute(
            "SELECT data FROM mind_snapshots WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"no snapshot named {name!r}")
        return Mind.from_dict(json.loads(row[0]))

    def list(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, created, updated FROM mind_snapshots ORDER BY updated DESC").fetchall()
        return [{"name": r[0], "created": r[1], "updated": r[2]} for r in rows]

    def delete(self, name: str) -> bool:
        cur = self._conn.execute("DELETE FROM mind_snapshots WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()
