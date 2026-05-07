import os
import json
import chromadb
from embeddings import get_embedding_function


def _pick_db_path():
    env_path = os.environ.get("CHROMA_DB_PATH")
    if env_path:
        print(f"Using env path: {env_path}")
        return env_path

    data_path = "/data/db"
    print(f"Checking if /data/db is writable...")
    try:
        os.makedirs(data_path, exist_ok=True)
        test_file = os.path.join(data_path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"SUCCESS: Using persistent path: {data_path}")
        return data_path
    except Exception as e:
        print(f"FAILED /data/db not writable: {e}")
        fallback = "/tmp/ekobot_db"
        print(f"Falling back to: {fallback}")
        return fallback

def build_db_if_empty():
    print("Checking ChromaDB...")
    import shutil
    db_path = _pick_db_path()
    print(f"Using ChromaDB path: {db_path}")

    # Test if existing db is healthy before indexing.
    try:
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection("ieu_courses")
        count = col.count()
        if count > 0:
            print(f"ChromaDB already has {count} docs. Skipping.")
            return
    except Exception as e:
        err_text = str(e)
        print(f"ChromaDB corrupted or empty: {err_text}")

        # Missing collection is normal for first run; don't delete folder for that case.
        if "does not exist" in err_text.lower():
            print("Collection does not exist yet. Building new index.")
        else:
            print("Deleting corrupted db and rebuilding...")
            try:
                shutil.rmtree(db_path)
                print("Deleted corrupted db.")
            except Exception:
                pass

    # Rebuild from scratch.
    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Primary DB init failed at {db_path}: {e}")
        fallback = "/tmp/ekobot_db"
        os.makedirs(fallback, exist_ok=True)
        client = chromadb.PersistentClient(path=fallback)
        db_path = fallback
        print(f"Switched ChromaDB path to fallback: {db_path}")

    try:
        client.delete_collection("ieu_courses")
    except Exception:
        pass

    try:
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
        corruption_detected = False

        for i, course in enumerate(courses):
            batch_docs.append(course["text"])
            batch_ids.append(f"course_{i}")
            batch_meta.append({
                "code": course.get("code", ""),
                "url": course.get("url", "")
            })

            if len(batch_docs) == 50:
                try:
                    collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_meta
                    )
                    print(f"  Indexed {i+1}/{len(courses)}")
                except Exception as batch_err:
                    err_text = str(batch_err)
                    print(f"Batch add failed at {i+1}/{len(courses)}: {err_text}")

                    if (
                        "database disk image is malformed" in err_text.lower()
                        or "readonly" in err_text.lower()
                    ):
                        print("Detected DB corruption during indexing. Switching to fallback DB...")
                        corruption_detected = True
                        try:
                            client = None
                            collection = None
                            shutil.rmtree(db_path)
                            print(f"Deleted corrupted {db_path}")
                        except Exception:
                            pass

                        fallback = "/tmp/ekobot_db"
                        os.makedirs(fallback, exist_ok=True)
                        try:
                            client = chromadb.PersistentClient(path=fallback)
                            client.delete_collection("ieu_courses")
                        except Exception:
                            pass

                        collection = client.create_collection(
                            "ieu_courses",
                            embedding_function=get_embedding_function()
                        )
                        db_path = fallback
                        print(f"Switched to {db_path}. Retrying batch...")

                        try:
                            collection.add(
                                documents=batch_docs,
                                ids=batch_ids,
                                metadatas=batch_meta
                            )
                            print(f"  Indexed {i+1}/{len(courses)} (recovered)")
                        except Exception as retry_err:
                            print(f"Retry failed: {retry_err}")
                    else:
                        raise

                batch_docs = []
                batch_ids = []
                batch_meta = []

        if batch_docs:
            try:
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_meta
                )
            except Exception as final_batch_err:
                err_text = str(final_batch_err)
                print(f"Final batch failed: {err_text}")

                if (
                    "database disk image is malformed" in err_text.lower()
                    or "readonly" in err_text.lower()
                ) and not corruption_detected:
                    print("Detected DB corruption on final batch. Switching to fallback...")
                    try:
                        shutil.rmtree(db_path)
                    except Exception:
                        pass

                    fallback = "/tmp/ekobot_db"
                    os.makedirs(fallback, exist_ok=True)
                    try:
                        client = chromadb.PersistentClient(path=fallback)
                        client.delete_collection("ieu_courses")
                    except Exception:
                        pass

                    collection = client.create_collection(
                        "ieu_courses",
                        embedding_function=get_embedding_function()
                    )

                    collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_meta
                    )
                    print(f"Final batch indexed after fallback")
                else:
                    raise

        print(f"=== Done! {len(courses)} courses indexed ===")

    except Exception as e:
        print(f"Startup error: {e}")

if __name__ == "__main__":
    build_db_if_empty()
