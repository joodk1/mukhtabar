"""Command-line ingest, until the UI exists.

    python -m mukhtabar.ingest add --set NAME [--language ar] FILE.pdf ...
    python -m mukhtabar.ingest rechunk --set NAME [--chunk-chars 1000]
    python -m mukhtabar.ingest removed --set NAME
"""

import argparse
import json
import sys

from mukhtabar import db
from mukhtabar.ingest.chunk import TARGET_CHARS
from mukhtabar.ingest.pipeline import ingest_pdf, rechunk


def _add(conn, args) -> None:
    for path in args.files:
        print(f"{path}:", flush=True)
        r = ingest_pdf(conn, args.set, path, args.language, args.chunk_chars)
        if r.skipped:
            print("  already in this study set, skipped")
            continue
        print(f"  language {r.language}, {r.pages} pages ({r.blank_pages} blank), {r.chunks} chunks")
        print(f"  header/footer lines removed: {r.removed_lines} (list them with the 'removed' command)")
        if r.flagged_pages:
            print(f"  pages flagged (possible columns, not split): {r.flagged_pages}")
        if r.kept_repeats:
            print("  repeated at a page edge but kept (too rare to be sure it isn't content):")
            for side, text, n in r.kept_repeats:
                print(f"    {n:4d} pages, {side}: {text}")
        if r.language == "und":
            print("  could not tell the language; set it with --language")


def _rechunk(conn, args) -> None:
    for filename, n in rechunk(conn, args.set, args.chunk_chars).items():
        print(f"{filename}: {n} chunks")


def _removed(conn, args) -> None:
    study_set = db.get_study_set(conn, args.set)
    if study_set is None:
        sys.exit(f"No study set named {args.set!r}")
    for doc in db.documents(conn, study_set["id"]):
        print(f"== {doc['filename']}")
        for page in db.pages(conn, doc["id"]):
            for line in json.loads(page["removed"]):
                print(f"  p{page['page']}: {line}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mukhtabar.ingest")
    parser.add_argument("--db", default=db.DEFAULT_PATH, help=f"database file (default {db.DEFAULT_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add PDFs to a study set (created if new)")
    add.add_argument("--set", required=True, help="study set name")
    add.add_argument("--language", help="override the guessed language, e.g. ar or en")
    add.add_argument("--chunk-chars", type=int, default=TARGET_CHARS)
    add.add_argument("files", nargs="+")

    re_ = sub.add_parser("rechunk", help="rebuild chunks from stored pages")
    re_.add_argument("--set", required=True)
    re_.add_argument("--chunk-chars", type=int, default=TARGET_CHARS)

    rm = sub.add_parser("removed", help="list header/footer lines removed at ingest")
    rm.add_argument("--set", required=True)

    args = parser.parse_args(argv)
    conn = db.connect(args.db)
    {"add": _add, "rechunk": _rechunk, "removed": _removed}[args.command](conn, args)


if __name__ == "__main__":
    main()
