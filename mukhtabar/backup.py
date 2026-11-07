"""A dated copy of the database.

    python -m mukhtabar.backup [--db PATH] [--to DIR]

Writes DIR/<name>-<YYYY-MM-DD>.db (DIR defaults to a "backups" folder next to the
database); a second backup on the same day gets -2, -3, ... SQLite's backup API is
used, so the copy is consistent even while the database is open elsewhere.
"""

import argparse
import sqlite3
from datetime import date
from pathlib import Path

from mukhtabar import db


def backup(db_path: str | Path, to: str | Path | None = None, today: date | None = None) -> Path:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"No database at {db_path}")
    folder = Path(to) if to else db_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{db_path.stem}-{(today or date.today()).isoformat()}"
    target, n = folder / f"{stem}.db", 1
    while target.exists():
        n += 1
        target = folder / f"{stem}-{n}.db"
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(target)
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mukhtabar.backup")
    parser.add_argument("--db", default=db.DEFAULT_PATH)
    parser.add_argument("--to", help="folder for the copy (default: backups/ next to the database)")
    args = parser.parse_args(argv)
    try:
        target = backup(args.db, args.to)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    print(f"Backed up to {target}")


if __name__ == "__main__":
    main()
