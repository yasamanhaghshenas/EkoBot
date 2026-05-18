import chromadb
import json
import os
import sys
import time
import hashlib
from embeddings import SentenceTransformerEmbeddingFunction
from scraper import scrape_support_pages


def normalize_metadata(course):
    url = course.get("url") or course.get("syllabus_url") or ""
    if url is None:
        url = ""
    return {
        "code": str(course.get("code", "")),
        "url": str(url),
    }


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


DATA_DB_PATH = os.environ.get("EKOBOT_DB_PATH") or "/data/ekobot_db"
INDEX_VERSION_FILE = os.path.join(DATA_DB_PATH, "index_version.txt")
COLLECTION_NAME = "ieu_courses"

try:
    print("Setting up ChromaDB...")
    os.makedirs(DATA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DATA_DB_PATH)
    EMBEDDING_FUNCTION = SentenceTransformerEmbeddingFunction()
except Exception as e:
    print(f"⚠ Warning: Failed to initialize embedding function or ChromaDB client: {e}")
    print("  ChromaDB operations may not work optimally")
    client = None
    EMBEDDING_FUNCTION = None


def get_collection():
    if client is None:
        print("⚠ Error: ChromaDB client not initialized")
        return None
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)


def safe_add(collection, documents, ids, metadatas, retries=2):
    last_error = None
    for _ in range(retries + 1):
        try:
            collection.add(documents=documents, ids=ids, metadatas=metadatas)
            return collection
        except Exception as e:
            last_error = e
            if "does not exist" in str(e).lower() or "notfound" in type(e).__name__.lower():
                collection = get_collection()
                continue
            raise
    raise last_error

def main():
    """Main indexing function - can be called safely even if setup failed."""
    if client is None:
        print("⚠ Cannot index: ChromaDB client not initialized")
        return
    start_time = time.time()

    def compute_data_hash():
        files = [
            "courses.json",
            "dept_info.json",
            "curriculum.json",
            "ecobot_rules_pages.json",
            "ieu_regulations.json",
        ]
        entries = []
        for f in files:
            try:
                size = os.path.getsize(f) if os.path.exists(f) else 0
            except Exception:
                size = 0
            entries.append(f"{f}:{size}")
        s = "\n".join(entries)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    # Ensure collection exists (create if missing)
    try:
        collection = get_collection()
        if collection is None:
            print("⚠ Failed to acquire ChromaDB collection")
            return
    except Exception as e:
        print(f"⚠ Failed to get collection: {e}")
        return

    try:
        current_hash = compute_data_hash()
        stored_hash = None
        if os.path.exists(INDEX_VERSION_FILE):
            try:
                with open(INDEX_VERSION_FILE, "r", encoding="utf-8") as fh:
                    stored_hash = fh.read().strip()
            except Exception:
                stored_hash = None

        col_count = 0
        try:
            col_count = collection.count()
        except Exception:
            col_count = 0

        # If index exists and version matches, skip indexing
        if col_count > 0 and stored_hash and stored_hash == current_hash:
            elapsed = time.time() - start_time
            print("[EkoBot] ChromaDB loaded from cache, skipping indexing")
            print(f"[EkoBot] Index loaded in {elapsed:.1f}s")
            return

        # If collection has docs but hash mismatches, rebuild by deleting collection
        if col_count > 0 and (not stored_hash or stored_hash != current_hash):
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            try:
                collection = client.create_collection(COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
            except Exception as e:
                print(f"⚠ Failed to create collection: {e}")
                return
    except Exception as e:
        print(f"⚠ Index version check failed: {e}")
        # proceed to (re)build index

    courses = []
    if os.path.exists("courses.json"):
        print("Loading courses.json...")
        with open("courses.json", "r", encoding="utf-8") as f:
            courses = json.load(f)
    else:
        print("courses.json not found. Skipping course indexing and continuing with available files...")

    print(f"Indexing {len(courses)} courses...")
    batch_size = 100
    indexed_courses = 0
    failed_courses = 0

    for group in batched(list(enumerate(courses)), batch_size):
        ids = []
        docs = []
        metas = []
        for i, course in group:
            text = course.get("text", "")
            if not text:
                failed_courses += 1
                continue
            ids.append(f"course_{i}")
            docs.append(text)
            metas.append(normalize_metadata(course))

        if not ids:
            continue

        try:
            collection = safe_add(collection, documents=docs, ids=ids, metadatas=metas)
            indexed_courses += len(ids)
            print(f"  Added batch: {indexed_courses}/{len(courses)}")
        except Exception as e:
            # Fallback to per-item add to isolate problematic records.
            print(f"  Batch failed, falling back to per-item add: {e}")
            for idx, doc, meta in zip(ids, docs, metas):
                try:
                    collection = safe_add(collection, documents=[doc], ids=[idx], metadatas=[meta])
                    indexed_courses += 1
                except Exception:
                    failed_courses += 1

    indexed_dept_docs = 0
    failed_dept_docs = 0
    if os.path.exists("dept_info.json"):
        print("\nLoading dept_info.json...")
        with open("dept_info.json", "r", encoding="utf-8") as f:
            dept_info = json.load(f)

        print(f"Indexing {len(dept_info)} department requirement records...")
        for i, item in enumerate(dept_info):
            dept = item.get("department", "")
            text = item.get("text", "")
            url = item.get("url", "")
            if not dept or not text:
                continue

            doc = f"Department: {dept}\nDocument Type: Graduation Requirements\n{text}"
            try:
                collection = safe_add(
                    collection,
                    documents=[doc],
                    ids=[f"dept_req_{i}"],
                    metadatas=[
                        {
                            "code": f"DEPT {dept}",
                            "department": str(dept),
                            "doc_type": "dept_requirements",
                            "url": str(url or ""),
                        }
                    ],
                )
                indexed_dept_docs += 1
            except Exception:
                failed_dept_docs += 1

    indexed_web_docs = 0
    failed_web_docs = 0
    indexed_support_docs = 0
    failed_support_docs = 0
    indexed_curriculum_docs = 0
    failed_curriculum_docs = 0
    if os.path.exists("curriculum.json"):
        print("\nLoading curriculum.json...")
        with open("curriculum.json", "r", encoding="utf-8") as f:
            curriculum = json.load(f)

        print(f"Indexing curriculum semesters from {len(curriculum)} rows...")
        for i, row in enumerate(curriculum):
            dept = (row.get("department") or "").strip()
            entry_year = row.get("entry_year")
            semesters = row.get("semesters") or {}
            if not dept or not isinstance(semesters, dict):
                continue

            for sem_name, courses in semesters.items():
                if not isinstance(courses, list):
                    continue
                lines = []
                for course in courses:
                    code = (course.get("code") or "").strip()
                    name = (course.get("name") or "").strip()
                    if code and name:
                        lines.append(f"- {code}: {name}")

                if not lines:
                    continue

                doc = (
                    f"Department: {dept}\n"
                    f"Entry Year: {entry_year}\n"
                    f"Semester: {sem_name}\n"
                    "Courses:\n" + "\n".join(lines)
                )
                try:
                    collection = safe_add(
                        collection,
                        documents=[doc],
                        ids=[f"curriculum_{i}_{sem_name}"],
                        metadatas=[
                            {
                                "doc_type": "curriculum_semester",
                                "department": dept,
                                "entry_year": str(entry_year or ""),
                                "semester": str(sem_name),
                            }
                        ],
                    )
                    indexed_curriculum_docs += 1
                except Exception:
                    failed_curriculum_docs += 1

    if os.path.exists("ecobot_rules_pages.json"):
        print("\nLoading ecobot_rules_pages.json...")
        with open("ecobot_rules_pages.json", "r", encoding="utf-8") as f:
            web_pages = json.load(f)

        print(f"Indexing {len(web_pages)} web info records...")
        for i, item in enumerate(web_pages):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            description = (item.get("description") or "").strip()
            text = (item.get("text") or "").strip()
            sheet = (item.get("sheet") or "").strip()

            if not (title or description or text):
                continue

            doc_parts = []
            if title:
                doc_parts.append(f"Title: {title}")
            if sheet:
                doc_parts.append(f"Category: {sheet}")
            if description:
                doc_parts.append(f"Description: {description}")
            if url:
                doc_parts.append(f"URL: {url}")
            if text:
                doc_parts.append("\n" + text)
            doc = "\n".join(doc_parts).strip()

            try:
                collection = safe_add(
                    collection,
                    documents=[doc],
                    ids=[f"web_info_{i}"],
                    metadatas=[
                        {
                            "doc_type": "web_info",
                            "title": title,
                            "url": url,
                            "sheet": sheet,
                        }
                    ],
                )
                indexed_web_docs += 1
            except Exception:
                failed_web_docs += 1

    print("\nScraping and indexing support service pages...")
    support_docs = scrape_support_pages(verify_ssl=True)
    print(f"Support chunks collected: {len(support_docs)}")
    for i, item in enumerate(support_docs):
        text = (item.get("text") or "").strip()
        metadata = item.get("metadata") or {}
        if not text:
            continue
        try:
            collection = safe_add(
                collection,
                documents=[text],
                ids=[f"support_page_{i}"],
                metadatas=[
                    {
                        "doc_type": "support_page",
                        "category": str(metadata.get("category") or ""),
                        "language": str(metadata.get("language") or ""),
                        "source_url": str(metadata.get("source_url") or ""),
                        "url": str(metadata.get("source_url") or ""),
                    }
                ],
            )
            indexed_support_docs += 1
        except Exception:
            failed_support_docs += 1

    indexed_reg_docs = 0
    failed_reg_docs = 0
    if os.path.exists("ieu_regulations.json"):
        print("\nLoading ieu_regulations.json...")
        try:
            with open("ieu_regulations.json", "r", encoding="utf-8") as f:
                regs = json.load(f)

            doc = json.dumps(regs, ensure_ascii=False)
            collection = safe_add(
                collection,
                documents=[doc],
                ids=["regulations_json"],
                metadatas=[{"doc_type": "regulations"}],
            )
            indexed_reg_docs = 1
        except Exception:
            failed_reg_docs = 1

    print("\nDone indexing.")
    print(f"Course docs indexed: {indexed_courses}")
    print(f"Course docs failed : {failed_courses}")
    print(f"Dept docs indexed  : {indexed_dept_docs}")
    print(f"Dept docs failed   : {failed_dept_docs}")
    print(f"Curriculum docs indexed: {indexed_curriculum_docs}")
    print(f"Curriculum docs failed : {failed_curriculum_docs}")
    print(f"Web docs indexed   : {indexed_web_docs}")
    print(f"Web docs failed    : {failed_web_docs}")
    print(f"Support docs indexed: {indexed_support_docs}")
    print(f"Support docs failed : {failed_support_docs}")
    print(f"Reg docs indexed   : {indexed_reg_docs}")
    print(f"Reg docs failed    : {failed_reg_docs}")
    print(f"Collection total   : {collection.count()}")

    # Write/update index version file to speed up future warm starts
    try:
        try:
            current_hash = None
            def compute_data_hash_local():
                files = [
                    "courses.json",
                    "dept_info.json",
                    "curriculum.json",
                    "ecobot_rules_pages.json",
                    "ieu_regulations.json",
                ]
                entries = []
                for f in files:
                    try:
                        size = os.path.getsize(f) if os.path.exists(f) else 0
                    except Exception:
                        size = 0
                    entries.append(f"{f}:{size}")
                s = "\n".join(entries)
                return hashlib.sha256(s.encode("utf-8")).hexdigest()

            current_hash = compute_data_hash_local()
            with open(INDEX_VERSION_FILE, "w", encoding="utf-8") as fh:
                fh.write(current_hash)
        except Exception:
            pass
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"[EkoBot] Index built in {elapsed:.1f}s")

# Call main() for indexing (safe to call even if initialization failed)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"⚠ Indexing failed: {e}")
        print("⚠ But the application can still run without indexing (using fallback embeddings)")