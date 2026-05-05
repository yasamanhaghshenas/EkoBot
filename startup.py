import os
import json
import chromadb
from embeddings import get_embedding_function

def build_db_if_empty():
    print("Checking ChromaDB...")
    client = chromadb.PersistentClient(path="/data/db")

    try:
        existing = client.get_collection(
            "ieu_courses",
            embedding_function=get_embedding_function()
        )
        count = existing.count()
        if count > 0:
            print(f"ChromaDB already has {count} docs. Skipping.")
            return
    except:
        pass

    try:
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
        batch_docs = []
        batch_ids = []
        batch_meta = []

        for i, course in enumerate(courses):
            batch_docs.append(course["text"])
            batch_ids.append(f"course_{i}")
            batch_meta.append({
                "code": course.get("code", ""),
                "url": course.get("url", "")
            })

            if len(batch_docs) == 50:
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_meta
                )
                print(f"  Indexed {i+1}/{len(courses)}")
                batch_docs = []
                batch_ids = []
                batch_meta = []

        if batch_docs:
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta
            )

        print(f"=== Done! {len(courses)} courses indexed ===")

    except Exception as e:
        print(f"Startup error: {e}")

if __name__ == "__main__":
    build_db_if_empty()
