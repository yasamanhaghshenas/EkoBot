import chromadb
import json

print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./db")

try:
    client.delete_collection("ieu_courses")
except:
    pass

collection = client.create_collection("ieu_courses")

print("Loading courses.json...")
with open("courses.json", "r", encoding="utf-8") as f:
    courses = json.load(f)

print(f"Indexing {len(courses)} courses...")
for i, course in enumerate(courses):
    collection.add(
        documents=[course["text"]],
        ids=[f"course_{i}"],
        metadatas=[{"code": course["code"], "url": course.get("url") or course.get("syllabus_url", "")}]
    )
    print(f"  Added: {course['code']}")

print(f"\nDone! {len(courses)} courses are now in the database.")