"""SQLite schema and queries.

The schema is versioned with PRAGMA user_version; each entry in MIGRATIONS
upgrades the database by one version. Tables for later milestones are added as
new migrations when those milestones need them.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = "mukhtabar.db"

MIGRATIONS = [
    """
    CREATE TABLE study_sets (
        id         INTEGER PRIMARY KEY,
        name       TEXT NOT NULL UNIQUE,
        language   TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE documents (
        id           INTEGER PRIMARY KEY,
        study_set_id INTEGER NOT NULL REFERENCES study_sets(id) ON DELETE CASCADE,
        filename     TEXT NOT NULL,
        sha256       TEXT NOT NULL,
        page_count   INTEGER NOT NULL,
        added_at     TEXT NOT NULL,
        UNIQUE (study_set_id, sha256)
    );
    CREATE TABLE pages (
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page        INTEGER NOT NULL,
        text        TEXT NOT NULL,
        flags       TEXT NOT NULL DEFAULT '[]',  -- JSON list, e.g. possible-columns-not-split
        removed     TEXT NOT NULL DEFAULT '[]',  -- JSON list of header/footer lines removed
        PRIMARY KEY (document_id, page)
    );
    CREATE TABLE chunks (
        id           INTEGER PRIMARY KEY,
        study_set_id INTEGER NOT NULL REFERENCES study_sets(id) ON DELETE CASCADE,
        document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page         INTEGER NOT NULL,
        page_end     INTEGER NOT NULL,
        text         TEXT NOT NULL,
        position     INTEGER NOT NULL,
        UNIQUE (document_id, position)
    );
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path = DEFAULT_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
        with conn:
            conn.executescript(script)
            conn.execute(f"PRAGMA user_version = {i}")


def get_study_set(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM study_sets WHERE name = ?", (name,)).fetchone()


def create_study_set(conn: sqlite3.Connection, name: str, language: str) -> int:
    cur = conn.execute(
        "INSERT INTO study_sets (name, language, created_at) VALUES (?, ?, ?)", (name, language, _now())
    )
    return cur.lastrowid


def set_language(conn: sqlite3.Connection, study_set_id: int, language: str) -> None:
    conn.execute("UPDATE study_sets SET language = ? WHERE id = ?", (language, study_set_id))


def find_document(conn: sqlite3.Connection, study_set_id: int, sha256: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE study_set_id = ? AND sha256 = ?", (study_set_id, sha256)
    ).fetchone()


def add_document(conn: sqlite3.Connection, study_set_id: int, filename: str, sha256: str, page_count: int) -> int:
    cur = conn.execute(
        "INSERT INTO documents (study_set_id, filename, sha256, page_count, added_at) VALUES (?, ?, ?, ?, ?)",
        (study_set_id, filename, sha256, page_count, _now()),
    )
    return cur.lastrowid


def documents(conn: sqlite3.Connection, study_set_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE study_set_id = ? ORDER BY id", (study_set_id,)
    ).fetchall()


def add_pages(conn: sqlite3.Connection, document_id: int, pages: list[tuple[int, str, list, list]]) -> None:
    """pages: (page number, text, flags, removed lines)."""
    conn.executemany(
        "INSERT INTO pages (document_id, page, text, flags, removed) VALUES (?, ?, ?, ?, ?)",
        [(document_id, n, text, json.dumps(flags, ensure_ascii=False), json.dumps(removed, ensure_ascii=False))
         for n, text, flags, removed in pages],
    )


def pages(conn: sqlite3.Connection, document_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM pages WHERE document_id = ? ORDER BY page", (document_id,)
    ).fetchall()


def replace_chunks(conn: sqlite3.Connection, study_set_id: int, document_id: int, chunks) -> None:
    """Replace a document's chunks. chunks: objects with text, page, page_end, position."""
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.executemany(
        "INSERT INTO chunks (study_set_id, document_id, page, page_end, text, position) VALUES (?, ?, ?, ?, ?, ?)",
        [(study_set_id, document_id, c.page, c.page_end, c.text, c.position) for c in chunks],
    )


def chunks(conn: sqlite3.Connection, document_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY position", (document_id,)
    ).fetchall()
