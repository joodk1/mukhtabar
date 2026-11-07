from datetime import date

import pytest

from mukhtabar import backup, db


def test_backup_writes_a_dated_copy_next_to_the_database(tmp_path):
    path = tmp_path / "mukhtabar.db"
    conn = db.connect(path)
    with conn:
        db.create_study_set(conn, "Set B", "ar")
    first = backup.backup(path, today=date(2026, 11, 7))
    second = backup.backup(path, today=date(2026, 11, 7))
    assert first == tmp_path / "backups" / "mukhtabar-2026-11-07.db"
    assert second.name == "mukhtabar-2026-11-07-2.db"               # same day: never overwritten
    copy = db.connect(first)
    assert copy.execute("SELECT name FROM study_sets").fetchone()[0] == "Set B"


def test_backup_of_a_missing_database_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup.backup(tmp_path / "none.db")
    assert not (tmp_path / "backups").exists()
