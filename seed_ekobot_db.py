"""
seed_ekobot_db.py
=================
Dynamically discovers and indexes every .json knowledge file into the
ChromaDB collection "ieu_knowledge_base".

Discovery order
---------------
1. The path given by --data-dir  (if provided)
2. ./data/   (if the directory exists)
3. The project root  (fallback – where the root-level JSON files live)

Structural shapes handled
--------------------------
  list[str]   – each non-empty string → one document
  list[dict]  – each dict → one or more documents
                • dicts with a "text" / "description" field use it as body
                • dicts whose value for any key is a dict-of-lists are
                  expanded into one document per inner key (e.g. one doc
                  per curriculum semester)
                • list-valued fields named "electives" → separate doc
  dict        – each top-level key → one document (used for regulation
                blocks, nested catalogs, etc.)

Usage
-----
    python seed_ekobot_db.py
    python seed_ekobot_db.py --data-dir ./data
    python seed_ekobot_db.py --reset                # wipe collection first
    python seed_ekobot_db.py --batch-size 200
    python seed_ekobot_db.py --db-dir /tmp/mydb
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import chromadb

from embeddings import SentenceTransformerEmbeddingFunction

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "ieu_knowledge_base"
DEFAULT_DB_PATH = "./db"
DEFAULT_BATCH = 100
MAX_DOC_CHARS = 8_000  # hard cap per document – keeps token budgets sane
MIN_DOC_CHARS = 10  # skip documents shorter than this

# Files that hold test / evaluation data or have no embeddable text content
SKIP_FILES = frozenset(
    {
        "eval_questions.json",  # automated test harness data
        "test_questions.json",  # automated test questions
        "course_codes.json",  # plain list of code strings – nothing to embed
        "ecobot_rules_sources.json",  # URL-only rows already covered by ecobot_rules_pages
    }
)

# Fields whose value is treated as the primary document body when present
_PRIMARY_TEXT_FIELDS = ("text", "description", "content", "body", "summary")

# Scalar fields that are safe to store as ChromaDB metadata
# (ChromaDB only accepts str | int | float | bool)
_META_KEYS = {
    "code",
    "department",
    "department_code",
    "url",
    "source_url",
    "entry_year",
    "program_name",
    "title",
    "sheet",
    "faculty_school",
    "category",
    "language",
    "doc_type",
}

# ─────────────────────────────────────────────────────────────────────────────
# Low-level utilities
# ─────────────────────────────────────────────────────────────────────────────


def _slugify(text: str, maxlen: int = 48) -> str:
    """Lower-case, strip non-alphanumeric, cap length – safe for IDs."""
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:maxlen]


def _stable_id(*parts) -> str:
    """
    Deterministic, collision-resistant document ID.
    Uses a SHA-1 prefix so re-running the seeder performs safe upserts.
    """
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _truncate(text: str) -> str:
    if len(text) <= MAX_DOC_CHARS:
        return text
    return text[:MAX_DOC_CHARS] + "\n…[truncated]"


def _label(key: str) -> str:
    """'entry_year' → 'Entry Year'"""
    return key.replace("_", " ").title()


def _is_dict_of_lists(value) -> bool:
    """
    Returns True when *value* is a non-empty dict whose values are all lists.
    This pattern appears in curriculum semesters: {semester_name: [courses]}.
    """
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(v, list) for v in value.values())
    )


# ─────────────────────────────────────────────────────────────────────────────
# Recursive value → text converter
# ─────────────────────────────────────────────────────────────────────────────


def _flatten_value(value, depth: int = 0, max_depth: int = 5) -> str:
    """
    Recursively convert any JSON value to a human-readable string.
    Returns "" for None / empty collections so callers can filter easily.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()

    if depth >= max_depth:
        return "[…]"

    if isinstance(value, list):
        parts = []
        for item in value:
            flat = _flatten_value(item, depth + 1)
            if flat:
                # Indent nested items to make structure readable
                prefix = "  - " if depth == 0 else "    "
                parts.append(f"{prefix}{flat}")
        return "\n".join(parts)

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = _flatten_value(v, depth + 1)
            if flat:
                parts.append(f"{_label(k)}: {flat}")
        return "\n".join(parts)

    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_metadata(entry: dict, extras: dict | None = None) -> dict:
    """
    Extract scalar fields that are safe to store as ChromaDB metadata.
    None values are dropped; all values are coerced to str/int/float/bool.
    """
    meta: dict = {}

    for k, v in entry.items():
        if k not in _META_KEYS:
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            meta[k] = v
        elif isinstance(v, (int, float)):
            meta[k] = v
        elif isinstance(v, str) and v.strip():
            meta[k] = v.strip()
        # Lists / dicts are intentionally omitted (not valid ChromaDB metadata)

    if extras:
        for k, v in extras.items():
            if v is None:
                continue
            if isinstance(v, bool):
                meta[k] = v
            elif isinstance(v, (int, float)):
                meta[k] = v
            elif isinstance(v, str) and v.strip():
                meta[k] = v.strip()

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Document builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_dict_body(entry: dict) -> str:
    """
    Convert a flat-ish dict to a human-readable document.

    Priority:
      1. Use the first 'text' / 'description' / … field as the main body.
      2. Prepend structured key:value header lines for scalar fields.
      3. Fall back to full flatten when nothing better is available.
    """
    primary = ""
    for field in _PRIMARY_TEXT_FIELDS:
        v = entry.get(field)
        if isinstance(v, str) and len(v.strip()) > 30:
            primary = v.strip()
            break

    header_lines = []
    for k, v in entry.items():
        if k in _PRIMARY_TEXT_FIELDS:
            continue  # will be appended separately
        if isinstance(v, (dict, list)):
            # Skip complex nested blobs in the header; they come via expansion
            flat = _flatten_value(v)
            if flat and len(flat) <= 300:
                header_lines.append(f"{_label(k)}: {flat}")
        elif v is not None:
            flat = _flatten_value(v)
            if flat:
                header_lines.append(f"{_label(k)}: {flat}")

    if primary:
        return "\n".join(header_lines + [primary])

    if header_lines:
        return "\n".join(header_lines)

    # Last-resort: flatten everything
    return _flatten_value(entry)


def _expand_dict_of_lists(
    field_name: str,
    inner_dict: dict,
    header: str,
    base_meta: dict,
    file_stem: str,
    entry_idx: int,
):
    """
    Expand a dict-of-lists field (e.g. semesters) into one document per key.
    Each value list is rendered as a bullet-point course/item list.
    """
    for inner_key, item_list in inner_dict.items():
        if not isinstance(item_list, list) or not item_list:
            continue

        bullet_lines = []
        for item in item_list:
            if isinstance(item, dict):
                parts = [str(v).strip() for k, v in item.items() if v is not None]
                bullet_lines.append("  - " + " | ".join(filter(None, parts)))
            elif item is not None:
                bullet_lines.append(f"  - {item}")

        body = f"{header}\n{_label(field_name)}: {inner_key}\nItems:\n" + "\n".join(
            bullet_lines
        )
        meta = {
            **base_meta,
            "sub_key": str(inner_key),
            "doc_type": f"{file_stem}_{_slugify(field_name)}",
        }
        doc_id = _stable_id(file_stem, entry_idx, field_name, inner_key)
        yield _truncate(body), meta, doc_id


def _docs_from_dict_entry(entry: dict, file_stem: str, idx: int):
    """
    Yield (doc_text, metadata, doc_id) from a single list-of-dicts item.

    Expansion logic (in order):
      1. Any field whose value is a dict-of-lists → expand per inner key.
         Typical example: 'semesters': {'1. Year Fall': [course, …], …}
      2. Any field literally named 'electives' that is a list of dicts →
         one additional document for the elective pool.
      3. The remaining scalar/text content → one standard document.
    """
    base_meta = _build_metadata(entry, {"doc_type": file_stem})

    # Build a header from fields that are NOT the expand target or electives
    # (reused across multiple sub-docs)
    def _header_for(exclude_key: str | None = None) -> str:
        lines = []
        for k, v in entry.items():
            if k == exclude_key:
                continue
            if k == "electives":
                continue
            if isinstance(v, (dict, list)):
                continue
            flat = _flatten_value(v)
            if flat:
                lines.append(f"{_label(k)}: {flat}")
        return "\n".join(lines)

    # ── 1. Detect dict-of-lists fields and expand them ──────────────────────
    expanded = False
    for field_key, field_val in entry.items():
        if not _is_dict_of_lists(field_val):
            continue
        expanded = True
        header = _header_for(exclude_key=field_key)
        yield from _expand_dict_of_lists(
            field_key, field_val, header, base_meta, file_stem, idx
        )

    # ── 2. Electives list → separate document ───────────────────────────────
    elec_list = entry.get("electives")
    if isinstance(elec_list, list) and elec_list:
        header = _header_for()
        bullet_lines = []
        for item in elec_list:
            if isinstance(item, dict):
                parts = [str(v).strip() for v in item.values() if v is not None]
                bullet_lines.append("  - " + " | ".join(filter(None, parts)))
            elif item is not None:
                bullet_lines.append(f"  - {item}")
        body = header + "\nElectives:\n" + "\n".join(bullet_lines)
        meta = {**base_meta, "doc_type": f"{file_stem}_electives"}
        doc_id = _stable_id(file_stem, idx, "electives")
        yield _truncate(body), meta, doc_id

    # ── 3. Main document (always produced for non-expand-only entries) ───────
    # For entries that were fully consumed by expansion, the main doc would be
    # redundant or empty – skip it only when all content came from expansion.
    if expanded:
        # Still produce a lightweight "index card" with scalar fields so the
        # department/year/program fields are discoverable independently.
        header = _header_for()
        if header.strip():
            meta = {**base_meta, "doc_type": f"{file_stem}_summary"}
            doc_id = _stable_id(file_stem, idx, "summary")
            yield _truncate(header), meta, doc_id
    else:
        body = _build_dict_body(entry)
        if body.strip():
            meta = base_meta
            doc_id = _stable_id(file_stem, idx)
            yield _truncate(body), meta, doc_id


def _docs_from_top_level_dict(data: dict, file_stem: str):
    """
    When the JSON root is itself a dict (e.g. ieu_regulations.json),
    yield one document per top-level key so each logical section is
    individually retrievable.
    """
    for section_key, section_val in data.items():
        body = f"Section: {_label(section_key)}\n{_flatten_value(section_val)}"
        if not body.strip():
            continue
        meta = {"section": section_key, "doc_type": file_stem}
        doc_id = _stable_id(file_stem, section_key)
        yield _truncate(body), meta, doc_id


def _docs_from_string_list(data: list, file_stem: str):
    """
    For JSON files that are plain lists of strings (e.g. tag lists,
    URL lists). Only items with at least MIN_DOC_CHARS are indexed.
    """
    for i, item in enumerate(data):
        if not isinstance(item, str):
            continue
        text = item.strip()
        if len(text) >= MIN_DOC_CHARS:
            meta = {"doc_type": file_stem}
            doc_id = _stable_id(file_stem, i, text)
            yield _truncate(text), meta, doc_id


# ─────────────────────────────────────────────────────────────────────────────
# File-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def generate_documents(filepath: Path):
    """
    Load one JSON file and yield (doc_text, metadata, doc_id) for every
    logical document it contains.

    Handles all structural shapes:
      - list[str]   → _docs_from_string_list
      - list[dict]  → _docs_from_dict_entry (with nested expansion)
      - dict        → _docs_from_top_level_dict
    """
    stem = _slugify(filepath.stem)

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  [WARN] {filepath.name}: invalid JSON – {exc}")
        return
    except OSError as exc:
        print(f"  [WARN] {filepath.name}: cannot read – {exc}")
        return
    except Exception as exc:
        print(f"  [WARN] {filepath.name}: unexpected error – {exc}")
        return

    if data is None:
        print(f"  [SKIP] {filepath.name}: file contains null")
        return

    # ── Dispatch by root type ─────────────────────────────────────────────────
    if isinstance(data, list):
        if not data:
            print(f"  [SKIP] {filepath.name}: empty list")
            return

        first = data[0]

        if isinstance(first, str):
            yield from _docs_from_string_list(data, stem)

        elif isinstance(first, dict):
            for idx, entry in enumerate(data):
                if not isinstance(entry, dict) or not entry:
                    continue
                try:
                    yield from _docs_from_dict_entry(entry, stem, idx)
                except Exception as exc:
                    print(f"  [WARN] {filepath.name}[{idx}]: skipped – {exc}")

        else:
            print(
                f"  [SKIP] {filepath.name}: list of {type(first).__name__} "
                f"– no handler for this element type"
            )

    elif isinstance(data, dict):
        if not data:
            print(f"  [SKIP] {filepath.name}: empty dict")
            return
        yield from _docs_from_top_level_dict(data, stem)

    else:
        print(f"  [SKIP] {filepath.name}: unsupported root type {type(data).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_collection(db_path: str, reset: bool):
    """Connect to (or create) the ChromaDB collection."""
    emb_fn = SentenceTransformerEmbeddingFunction()
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[DB] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # Collection did not exist – that's fine

    col = client.get_or_create_collection(COLLECTION_NAME, embedding_function=emb_fn)
    print(f"[DB] Collection '{COLLECTION_NAME}' ready  ({col.count()} existing docs)")
    return col


def batch_upsert(
    collection, documents: list, ids: list, metadatas: list, batch_size: int
) -> int:
    """
    Upsert documents in batches.
    On batch failure, falls back to per-item upsert so one bad record
    does not block the entire file.
    Returns the number of successfully indexed documents.
    """
    indexed = 0
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        b_docs = documents[start:end]
        b_ids = ids[start:end]
        b_metas = metadatas[start:end]
        try:
            collection.upsert(documents=b_docs, ids=b_ids, metadatas=b_metas)
            indexed += len(b_docs)
        except Exception as batch_err:
            print(f"  [WARN] Batch upsert failed ({batch_err}); retrying item-by-item…")
            for doc, did, meta in zip(b_docs, b_ids, b_metas):
                try:
                    collection.upsert(documents=[doc], ids=[did], metadatas=[meta])
                    indexed += 1
                except Exception as item_err:
                    print(f"    [ERR] Skipped id={did} – {item_err}")
    return indexed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _find_data_dir(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        p = Path(candidate)
        if p.is_dir():
            return p.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed ChromaDB 'ieu_knowledge_base' from JSON knowledge files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Directory containing .json files (default: ./data/ then project root)",
    )
    parser.add_argument(
        "--db-dir",
        default=DEFAULT_DB_PATH,
        metavar="PATH",
        help=f"ChromaDB persistence path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH,
        type=int,
        metavar="N",
        help=f"Upsert batch size (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    args = parser.parse_args()

    # ── Resolve data directory ─────────────────────────────────────────────────
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        if not data_dir.is_dir():
            print(
                f"[ERROR] --data-dir '{args.data_dir}' does not exist or is not a directory"
            )
            sys.exit(1)
    else:
        # Auto-detect: prefer ./data/, fall back to project root
        data_dir = _find_data_dir(["data", "."])
        if data_dir is None:
            print("[ERROR] Cannot locate a data directory. Run with --data-dir <path>.")
            sys.exit(1)

    print(f"[Seeder] Scanning: {data_dir}")

    # ── Discover .json files ───────────────────────────────────────────────────
    all_json = sorted(data_dir.glob("*.json"))
    if not all_json:
        print("[Seeder] No .json files found. Nothing to index.")
        return

    to_index = [f for f in all_json if f.name not in SKIP_FILES]
    skipped = [f.name for f in all_json if f.name in SKIP_FILES]

    if skipped:
        print(f"[Seeder] Skipping {len(skipped)} file(s) (test/non-knowledge data):")
        for name in skipped:
            print(f"         • {name}")

    print(f"[Seeder] Files to index ({len(to_index)}):")
    for f in to_index:
        print(f"         • {f.name}")
    print()

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    os.makedirs(args.db_dir, exist_ok=True)
    col = get_collection(args.db_dir, args.reset)
    print()

    # ── Process each file ─────────────────────────────────────────────────────
    grand_total = 0
    file_stats = {}

    for filepath in to_index:
        print(f"── {filepath.name} ──────────────────────────────────────────")
        docs, ids, metas = [], [], []

        for doc_text, metadata, doc_id in generate_documents(filepath):
            text = (doc_text or "").strip()
            if len(text) < MIN_DOC_CHARS:
                continue  # skip near-empty docs
            docs.append(text)
            ids.append(doc_id)
            metas.append(metadata if metadata else {})

        if not docs:
            print(f"  [SKIP] No indexable documents generated\n")
            file_stats[filepath.name] = 0
            continue

        print(f"  Generated : {len(docs):>6} document(s)")
        indexed = batch_upsert(col, docs, ids, metas, args.batch_size)
        grand_total += indexed
        file_stats[filepath.name] = indexed
        print(f"  Indexed   : {indexed:>6}/{len(docs)}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    bar = "=" * 54
    print(bar)
    print(f"  {'File':<36}  {'Docs':>6}")
    print(f"  {'-' * 36}  {'------':>6}")
    for name, count in file_stats.items():
        print(f"  {name:<36}  {count:>6}")
    print(f"  {'-' * 36}  {'------':>6}")
    print(f"  {'TOTAL':<36}  {grand_total:>6}")
    print(bar)
    print(f"  Collection size now: {col.count()} docs")
    print(bar)


if __name__ == "__main__":
    main()
