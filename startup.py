import os
import json
import chromadb

def build_db_if_empty():
    print("Checking ChromaDB...")
    client = chromadb.PersistentClient(path="./db")
    
    try:
        # delete and recreate to avoid embedding conflicts
        try:
            client.delete_collection("ieu_courses")
            print("Deleted old collection to avoid embedding conflicts.")
        except:
            pass
        
        collection = client.create_collection("ieu_courses")
        
        if not os.path.exists("courses.json"):
            print("courses.json not found.")
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
        
    except Exception as e:
        print(f"Startup error: {e}")

if __name__ == "__main__":
    build_db_if_empty()
