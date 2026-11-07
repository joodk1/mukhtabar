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
    # 2: topic map (milestone 2)
    """
    CREATE TABLE topics (
        id           INTEGER PRIMARY KEY,
        study_set_id INTEGER NOT NULL REFERENCES study_sets(id) ON DELETE CASCADE,
        title        TEXT NOT NULL,
        summary      TEXT NOT NULL DEFAULT '',
        chapter      TEXT,
        position     INTEGER NOT NULL,
        edited       INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );
    CREATE TABLE topic_chunks (
        topic_id   INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
        chunk_id   INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        is_primary INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (topic_id, chunk_id)
    );
    -- At most one primary topic per chunk; secondary links can be added later.
    CREATE UNIQUE INDEX one_primary_topic_per_chunk ON topic_chunks (chunk_id) WHERE is_primary = 1;
    ALTER TABLE chunks ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE chunks ADD COLUMN excluded_reason TEXT;
    CREATE TABLE llm_calls (
        id            INTEGER PRIMARY KEY,
        study_set_id  INTEGER REFERENCES study_sets(id) ON DELETE SET NULL,
        purpose       TEXT NOT NULL,
        model         TEXT NOT NULL,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cost_usd      REAL,
        attempts      INTEGER NOT NULL,
        ok            INTEGER NOT NULL,
        error         TEXT,
        seconds       REAL,
        created_at    TEXT NOT NULL
    );
    """,
    # 3: finished map batches, so an interrupted build resumes without repeating paid calls
    """
    CREATE TABLE llm_cache (
        key          TEXT PRIMARY KEY,           -- sha256 of model + exact messages
        study_set_id INTEGER NOT NULL REFERENCES study_sets(id) ON DELETE CASCADE,
        purpose      TEXT NOT NULL,
        reply        TEXT NOT NULL,              -- the validated JSON reply
        created_at   TEXT NOT NULL
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


# --- Topics (milestone 2) ---

def study_chunks(conn: sqlite3.Connection, study_set_id: int) -> list[sqlite3.Row]:
    """All chunks of a set with their document, in document and position order."""
    return conn.execute(
        """SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.study_set_id = ? ORDER BY d.id, c.position""", (study_set_id,)
    ).fetchall()


def topics(conn: sqlite3.Connection, study_set_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT t.*, COUNT(tc.chunk_id) AS n_chunks FROM topics t
           LEFT JOIN topic_chunks tc ON tc.topic_id = t.id
           WHERE t.study_set_id = ? GROUP BY t.id ORDER BY t.position, t.id""", (study_set_id,)
    ).fetchall()


def get_topic(conn: sqlite3.Connection, topic_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()


def add_topic(conn: sqlite3.Connection, study_set_id: int, title: str, summary: str,
              chapter: str | None, position: int) -> int:
    now = _now()
    cur = conn.execute(
        """INSERT INTO topics (study_set_id, title, summary, chapter, position, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""", (study_set_id, title, summary, chapter, position, now, now)
    )
    return cur.lastrowid


def update_topic(conn: sqlite3.Connection, topic_id: int, **fields) -> None:
    fields = {**fields, "edited": 1, "updated_at": _now()}
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE topics SET {sets} WHERE id = ?", (*fields.values(), topic_id))


def delete_topic(conn: sqlite3.Connection, topic_id: int) -> None:
    conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))


def link_chunks(conn: sqlite3.Connection, topic_id: int, chunk_ids: list[int]) -> None:
    conn.executemany(
        "INSERT INTO topic_chunks (topic_id, chunk_id, is_primary) VALUES (?, ?, 1)",
        [(topic_id, c) for c in chunk_ids],
    )


def move_links(conn: sqlite3.Connection, from_topic: int, to_topic: int) -> None:
    conn.execute("UPDATE topic_chunks SET topic_id = ? WHERE topic_id = ?", (to_topic, from_topic))


def topic_chunk_rows(conn: sqlite3.Connection, topic_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT c.*, d.filename FROM topic_chunks tc JOIN chunks c ON c.id = tc.chunk_id
           JOIN documents d ON d.id = c.document_id WHERE tc.topic_id = ? ORDER BY d.id, c.position""",
        (topic_id,),
    ).fetchall()


def unassigned_chunks(conn: sqlite3.Connection, study_set_id: int) -> list[sqlite3.Row]:
    """Study chunks (not excluded) with no primary topic."""
    return conn.execute(
        """SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.study_set_id = ? AND c.excluded = 0
             AND NOT EXISTS (SELECT 1 FROM topic_chunks tc WHERE tc.chunk_id = c.id AND tc.is_primary = 1)
           ORDER BY d.id, c.position""", (study_set_id,)
    ).fetchall()


def set_excluded(conn: sqlite3.Connection, chunk_id: int, reason: str | None) -> None:
    conn.execute("UPDATE chunks SET excluded = ?, excluded_reason = ? WHERE id = ?",
                 (1 if reason else 0, reason, chunk_id))


def clear_topic_map(conn: sqlite3.Connection, study_set_id: int) -> None:
    """Delete a set's topics (links go with them) and the model's exclusions."""
    conn.execute("DELETE FROM topics WHERE study_set_id = ?", (study_set_id,))
    conn.execute("UPDATE chunks SET excluded = 0, excluded_reason = NULL WHERE study_set_id = ?", (study_set_id,))


def linked_chunk_count(conn: sqlite3.Connection, document_id: int) -> tuple[int, int]:
    """(chunks of this document linked to a topic, distinct topics they're linked to)."""
    row = conn.execute(
        """SELECT COUNT(DISTINCT tc.chunk_id), COUNT(DISTINCT tc.topic_id) FROM topic_chunks tc
           JOIN chunks c ON c.id = tc.chunk_id WHERE c.document_id = ?""", (document_id,)
    ).fetchone()
    return row[0], row[1]


def empty_topics(conn: sqlite3.Connection, study_set_id: int) -> list[sqlite3.Row]:
    return [t for t in topics(conn, study_set_id) if t["n_chunks"] == 0]


def log_llm_call(conn: sqlite3.Connection, study_set_id: int | None, record: dict) -> None:
    conn.execute(
        """INSERT INTO llm_calls (study_set_id, purpose, model, input_tokens, output_tokens, cost_usd,
                                  attempts, ok, error, seconds, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (study_set_id, record["purpose"], record["model"], record.get("input_tokens"),
         record.get("output_tokens"), record.get("cost_usd"), record["attempts"], int(record["ok"]),
         record.get("error"), record.get("seconds"), _now()),
    )
    conn.commit()


def rename_study_set(conn: sqlite3.Connection, old: str, new: str) -> None:
    cur = conn.execute("UPDATE study_sets SET name = ? WHERE name = ?", (new, old))
    if cur.rowcount != 1:
        raise ValueError(f"No study set named {old!r}")


def cache_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT reply FROM llm_cache WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def cache_put(conn: sqlite3.Connection, key: str, study_set_id: int, purpose: str, reply: str) -> None:
    conn.execute("INSERT OR REPLACE INTO llm_cache (key, study_set_id, purpose, reply, created_at) VALUES (?, ?, ?, ?, ?)",
                 (key, study_set_id, purpose, reply, _now()))
    conn.commit()


def cache_clear(conn: sqlite3.Connection, study_set_id: int) -> None:
    conn.execute("DELETE FROM llm_cache WHERE study_set_id = ?", (study_set_id,))
