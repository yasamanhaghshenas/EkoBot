import chromadb

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("ieu_courses")

# test what ChromaDB actually finds for your question
results = collection.query(
    query_texts=["prerequisites for SE 116"],
    n_results=3
)

print("Top 3 results found:\n")
for i, doc in enumerate(results["documents"][0]):
    print(f"--- Result {i+1} ---")
    print(doc[:500])  # show first 500 characters
    print()