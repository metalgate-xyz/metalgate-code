"""SQLite-backed cache for resolved symbols and file outlines.

Keys are always (file, mtime) so stale entries are never served.
Four tables:
  - outlines     (tree-sitter-extracted symbols; root-independent)
  - definitions  (LSP-resolved goto; scoped by root)
  - callees      (LSP-resolved call targets of a function; scoped by root)
  - symbols      (LSP workspace/symbol results; scoped by root)

``definitions``, ``callees`` and ``symbols`` are keyed by the
language-server root because their results depend on what the LSP
indexes, which in turn depends on the workspace root.  Steering the
root (see ``Tracer.set_root``) therefore produces clean cache misses
instead of serving answers resolved under a previous root — and the
persisted ``root`` column makes stale entries visible when inspecting
the DB.
"""

import json
import os
import sqlite3
import threading
from typing import Any

# Bumped on any incompatible schema change.  The cache DB lives under
# gitignored ``.metalgate/`` and is disposable, so a version mismatch
# drops and recreates the tables rather than migrating rows.
_SCHEMA_VERSION = 3

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS outlines (
    file    TEXT    NOT NULL,
    mtime   REAL    NOT NULL,
    symbols TEXT    NOT NULL,
    PRIMARY KEY (file)
);

CREATE TABLE IF NOT EXISTS definitions (
    root      TEXT    NOT NULL,
    file      TEXT    NOT NULL,
    mtime     REAL    NOT NULL,
    line      INTEGER NOT NULL,
    name      TEXT    NOT NULL,
    result    TEXT,
    PRIMARY KEY (root, file, line, name)
);

CREATE INDEX IF NOT EXISTS idx_def_file ON definitions(root, file, mtime);

CREATE TABLE IF NOT EXISTS callees (
    root      TEXT    NOT NULL,
    file      TEXT    NOT NULL,
    mtime     REAL    NOT NULL,
    line      INTEGER NOT NULL,
    results   TEXT    NOT NULL,
    PRIMARY KEY (root, file, line)
);

CREATE INDEX IF NOT EXISTS idx_callees_file ON callees(root, file, mtime);

CREATE TABLE IF NOT EXISTS symbols (
    root    TEXT    NOT NULL,
    name    TEXT    NOT NULL,
    results TEXT    NOT NULL,
    PRIMARY KEY (root, name)
);
"""


def _mtime(file: str) -> float:
    try:
        return os.path.getmtime(file)
    except OSError:
        return 0.0


class CodeCache:
    """Thread-safe SQLite cache using thread-local connections."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._maybe_migrate()
        self._execute_script(_SCHEMA)

    # connection management

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _execute_script(self, sql: str) -> None:
        conn = self._conn()
        conn.executescript(sql)
        conn.commit()

    def _maybe_migrate(self) -> None:
        """Drop any tables from an incompatible schema version.

        The cache is disposable (lives under gitignored ``.metalgate/``),
        so on a ``user_version`` mismatch we drop the known tables and
        let ``_SCHEMA`` recreate them empty.  ``user_version`` is the
        connection's schema stamp, written after a successful rebuild.
        """
        conn = self._conn()
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current == _SCHEMA_VERSION:
            return
        conn.executescript(
            "DROP TABLE IF EXISTS outlines;"
            "DROP TABLE IF EXISTS definitions;"
            "DROP TABLE IF EXISTS callees;"
            "DROP TABLE IF EXISTS symbols;"
        )
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()

    # outline cache

    def get_outline(self, file: str) -> list[dict] | None:
        current_mtime = _mtime(file)
        row = (
            self._conn()
            .execute("SELECT mtime, symbols FROM outlines WHERE file = ?", (file,))
            .fetchone()
        )
        if row and row["mtime"] == current_mtime:
            return json.loads(row["symbols"])
        return None

    def set_outline(self, file: str, symbols: list[dict]) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO outlines(file, mtime, symbols) VALUES (?, ?, ?)",
            (file, _mtime(file), json.dumps(symbols)),
        )
        self._conn().commit()

    # definition cache

    def get_definition(self, root: str, file: str, line: int, name: str) -> Any | None:
        """Returns cached result (may be None if we cached a miss)."""
        current_mtime = _mtime(file)
        row = (
            self._conn()
            .execute(
                "SELECT mtime, result FROM definitions "
                "WHERE root = ? AND file = ? AND line = ? AND name = ?",
                (root, file, line, name),
            )
            .fetchone()
        )
        if row and row["mtime"] == current_mtime:
            raw = row["result"]
            return json.loads(raw) if raw else None
        return _CACHE_MISS  # sentinel: not in cache at all

    def set_definition(
        self,
        root: str,
        file: str,
        line: int,
        name: str,
        result: dict | None,
    ) -> None:
        self._conn().execute(
            """INSERT OR REPLACE INTO definitions(root, file, mtime, line, name, result)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (root, file, _mtime(file), line, name, json.dumps(result)),
        )
        self._conn().commit()

    # callees cache (get_callees / LSP definition per call site)

    def get_callees(self, root: str, file: str, line: int) -> list[dict] | None:
        """Returns cached ``get_callees`` results, or None on a miss."""
        current_mtime = _mtime(file)
        row = (
            self._conn()
            .execute(
                "SELECT mtime, results FROM callees "
                "WHERE root = ? AND file = ? AND line = ?",
                (root, file, line),
            )
            .fetchone()
        )
        if row and row["mtime"] == current_mtime:
            return json.loads(row["results"])
        return None

    def set_callees(
        self,
        root: str,
        file: str,
        line: int,
        results: list[dict],
    ) -> None:
        self._conn().execute(
            """INSERT OR REPLACE INTO callees(root, file, mtime, line, results)
               VALUES (?, ?, ?, ?, ?)""",
            (root, file, _mtime(file), line, json.dumps(results)),
        )
        self._conn().commit()

    # symbol cache (find_symbol / LSP workspace/symbol)

    def get_symbol(self, root: str, name: str) -> list[dict] | None:
        """Returns cached ``find_symbol`` results, or None on a miss."""
        row = (
            self._conn()
            .execute(
                "SELECT results FROM symbols WHERE root = ? AND name = ?",
                (root, name),
            )
            .fetchone()
        )
        if row is None:
            return None
        return json.loads(row["results"])

    def set_symbol(self, root: str, name: str, results: list[dict]) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO symbols(root, name, results) VALUES (?, ?, ?)",
            (root, name, json.dumps(results)),
        )
        self._conn().commit()

    # cache clearing

    def clear_outlines(self) -> None:
        """Remove every cached tree-sitter outline."""
        self._conn().execute("DELETE FROM outlines")
        self._conn().commit()

    def clear_definitions(self) -> None:
        """Remove every cached LSP definition resolution."""
        self._conn().execute("DELETE FROM definitions")
        self._conn().commit()

    def clear_callees(self) -> None:
        """Remove every cached get_callees result."""
        self._conn().execute("DELETE FROM callees")
        self._conn().commit()

    def clear_symbols(self) -> None:
        """Remove every cached find_symbol result."""
        self._conn().execute("DELETE FROM symbols")
        self._conn().commit()

    def clear_cache(self) -> None:
        """Clear all caches: outlines, definitions, callees, and symbols."""
        self.clear_outlines()
        self.clear_definitions()
        self.clear_callees()
        self.clear_symbols()


# Sentinel to distinguish "cached as None (miss)" from "not in cache"
class _CacheMiss:
    pass


_CACHE_MISS = _CacheMiss()
