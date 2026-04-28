import os
import json
import chromadb

def build_db_if_empty():
    print("Checking ChromaDB...")
    client = chromadb.PersistentClient(path="./db")
    
    try:
        collection = client.get_or_create_collection("ieu_courses")
        if collection.count() > 0:
            print(f"ChromaDB already has {collection.count()} courses. Skipping.")
            return
    except:
        collection = client.get_or_create_collection("ieu_courses")

    if not os.path.exists("courses.json"):
        print("courses.json not found. Cannot build DB.")
        return

    print("Building ChromaDB from courses.json...")
    with open("courses.json", "r", encoding="utf-8") as f:
        courses = json.load(f)

    for i, course in enumerate(courses):
        collection.add(
            documents=[course["text"]],
            ids=[f"course_{i}"],
            metadatas=[{
                "code": course.get("code", ""),
                "url": course.get("url", "")
            }]
        )
        if i % 100 == 0:
            print(f"  Indexed {i}/{len(courses)}...")

    print(f"Done! {len(courses)} courses indexed.")

if __name__ == "__main__":
    build_db_if_empty()
