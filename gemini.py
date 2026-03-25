from groq import Groq
import chromadb
import re

GROQ_API_KEY = "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" 

client_groq = Groq(api_key=GROQ_API_KEY)

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("ieu_courses")

def extract_course_codes(text):
    pattern = r'\b([A-Z]{2,5})\s*(\d{3})\b'
    matches = re.findall(pattern, text.upper())
    return list(set([f"{m[0]} {m[1]}" for m in matches]))

def get_context_for_codes(codes):
    contexts = []
    for code in codes:
        found = False

        # first try exact match
        try:
            results = collection.query(
                query_texts=[code],
                n_results=1,
                where={"code": code}
            )
            if results["documents"][0]:
                contexts.append(results["documents"][0][0])
                found = True
        except:
            pass

        # if not found, try semantic search
        # this handles typos like EEEE 242 -> EEE 242
        if not found:
            print(f"  '{code}' not found exactly, trying fuzzy search...")
            results = collection.query(
                query_texts=[code],
                n_results=1
            )
            if results["documents"][0]:
                doc = results["documents"][0][0]
                match = re.search(r'Course Code: ([A-Z]{2,5} \d{3})', doc)
                if match:
                    found_code = match.group(1)
                    print(f"  Fuzzy matched '{code}' -> '{found_code}'")
                    contexts.append(doc)

    return contexts

def ask(question, history=[]):
    print(f"\nSearching database for: {question}")

    # extract codes from current question only for multi-course detection
    current_codes = extract_course_codes(question)

    # also extract from history for memory/context
    history_text = " ".join([msg["content"] for msg in history[-4:]])
    history_codes = extract_course_codes(history_text)

    contexts = []

    if current_codes:
        print(f"  Detected course codes in question: {current_codes}")

        # fetch each course separately so we never miss one
        contexts = get_context_for_codes(current_codes)

        # check which codes were actually found
        found_codes = []
        for ctx in contexts:
            for code in current_codes:
                if code in ctx:
                    found_codes.append(code)

        # for missing codes try semantic search as fallback
        missing_codes = [c for c in current_codes if c not in found_codes]
        if missing_codes:
            print(f"  Trying semantic search for missing: {missing_codes}")
            for code in missing_codes:
                results = collection.query(
                    query_texts=[code],
                    n_results=1
                )
                if results["documents"][0]:
                    contexts.append(results["documents"][0][0])

    elif history_codes:
        # no codes in current question but there are codes in history
        # e.g. "what about se 116?" after asking about CE 342
        print(f"  No codes in question, using history codes: {history_codes}")
        contexts = get_context_for_codes(history_codes)

    if not contexts:
        # no codes anywhere - do general semantic search
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        contexts = results["documents"][0]

    if not contexts:
        return "I don't have information about that in the IEU course database."

    context = "\n\n---\n\n".join(contexts)
    print(f"  Found {len(contexts)} relevant course(s), asking AI...")

    # build messages with history
    messages = [
        {
            "role": "system",
            "content": """You are EkoBot, a helpful academic assistant for İzmir University of Economics (IEU).
Answer student questions about IEU courses accurately based only on the context provided.
When the student asks about multiple courses in one message, answer each course separately with a clear label like "MATH 154:", "MATH 240:", etc.
If a follow-up question refers to a course mentioned earlier in the conversation, use that context.
If the answer is not in the context for a specific course, say so for that course only and do not guess.
Always answer in the same language the student used — Turkish or English.
Be concise and direct. Do not make up information.
If a course code seems like a typo (e.g. EEEE 242 instead of EEE 242), answer based on the closest match found and mention the correction to the student."""
        }
    ]

    # add conversation history (last 6 messages)
    for msg in history[-6:]:
        messages.append(msg)

    # add current question with context
    messages.append({
        "role": "user",
        "content": f"Context from IEU course database:\n{context}\n\nStudent question: {question}"
    })

    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return response.choices[0].message.content


# terminal test with memory
if __name__ == "__main__":
    print("EkoBot Test — type your question (or 'quit' to exit)")
    print("-" * 50)

    conversation_history = []

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        answer = ask(question, conversation_history)
        print(f"\nAnswer: {answer}")

        # save to history
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})