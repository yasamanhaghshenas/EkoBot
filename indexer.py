import chromadb
import json
import os
from embeddings import SimpleHashEmbeddingFunction


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


print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./db")
COLLECTION_NAME = "ieu_courses"
EMBEDDING_FUNCTION = SimpleHashEmbeddingFunction()


def get_collection():
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

try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

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

if os.path.exists("dept_info.json"):
    print("\nLoading dept_info.json...")
    with open("dept_info.json", "r", encoding="utf-8") as f:
        dept_info = json.load(f)

    print(f"Indexing {len(dept_info)} department requirement records...")
    indexed_dept_docs = 0
    failed_dept_docs = 0
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

    print("\nDone indexing.")
    print(f"Course docs indexed: {indexed_courses}")
    print(f"Course docs failed : {failed_courses}")
    print(f"Dept docs indexed  : {indexed_dept_docs}")
    print(f"Dept docs failed   : {failed_dept_docs}")
    print(f"Collection total   : {collection.count()}")