import os
import json
import chromadb
from embeddings import get_embedding_function


def _pick_db_path():
    return "/tmp/ekobot_db"


READY_FLAG = "/tmp/ekobot_ready.flag"


def build_db_if_empty():
    print("Checking ChromaDB...")
    db_path = "/tmp/ekobot_db"

    # Check if already indexed successfully
    if os.path.exists(READY_FLAG):
        try:
            client = chromadb.PersistentClient(path=db_path)
            col = client.get_collection("ieu_courses")
            count = col.count()
            if count > 0:
                print(f"ChromaDB ready with {count} docs. Skipping.")
                return
        except:
            pass
        # flag exists but db is bad, remove flag and rebuild
        try:
            os.remove(READY_FLAG)
        except:
            pass

    # rebuild
    try:
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)
    except:
        pass

    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection("ieu_courses")
    except:
        pass

    collection = client.create_collection(
        "ieu_courses",
        embedding_function=get_embedding_function()
    )

    print("Loading courses.json...")
    with open("courses.json", "r", encoding="utf-8") as f:
        courses = json.load(f)

    print(f"Indexing {len(courses)} courses...")
    batch_docs, batch_ids, batch_meta = [], [], []

    for i, course in enumerate(courses):
        batch_docs.append(course["text"])
        batch_ids.append(f"course_{i}")
        batch_meta.append({
            "code": course.get("code", ""),
            "url": course.get("url", "")
        })

        if len(batch_docs) == 100:
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta
            )
            print(f"  Indexed {i+1}/{len(courses)}")
            batch_docs, batch_ids, batch_meta = [], [], []

    if batch_docs:
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_meta
        )

    # write success flag
    with open(READY_FLAG, "w") as f:
        f.write(str(len(courses)))

    print(f"=== Done! {len(courses)} courses indexed ===")

if __name__ == "__main__":
    build_db_if_empty()
