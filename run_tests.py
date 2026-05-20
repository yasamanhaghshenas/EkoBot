import json
import re
import sys

import requests


CHAT_URL = "https://timo1382-ekobot.hf.space/chat"
FALLBACK_MARKERS = [
    "sorry, i can't help",
]
STOPWORDS = {
    "a", "about", "an", "and", "answer", "any", "as", "brief", "can",
    "contains", "content", "course", "correctly", "direct", "expected",
    "flow", "for", "friendly", "from", "human", "in", "info", "is",
    "mention", "needs", "no", "not", "or", "response", "should", "system",
    "the", "to", "turkish", "with",
}


def load_cases(path="test_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_keywords(expected):
    text = expected or ""
    phrases = []

    for piece in re.split(r"[,;]", text):
        piece = piece.strip()
        if piece and len(piece) > 2:
            phrases.append(piece.lower())

    tokens = []
    for token in re.findall(r"[\w.%+-]+", text.lower(), flags=re.UNICODE):
        if len(token) < 3 and not re.search(r"\d", token):
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)

    return phrases + tokens


def first_100(text):
    return re.sub(r"\s+", " ", text or "").strip()[:100]


def ask(question):
    response = requests.post(
        CHAT_URL,
        json={"question": question, "history": []},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("answer", "")


def mark_result(answer, expected):
    answer_text = answer or ""
    answer_lower = answer_text.lower()
    if not answer_text.strip():
        return "FAIL"
    if any(marker in answer_lower for marker in FALLBACK_MARKERS):
        return "FAIL"

    keywords = expected_keywords(expected)
    if any(keyword in answer_lower for keyword in keywords):
        return "PASS"
    return "MANUAL"


def main():
    try:
        cases = load_cases()
    except Exception as exc:
        print(f"Could not read test_questions.json: {exc}")
        return 2

    counts = {"PASS": 0, "FAIL": 0, "MANUAL": 0}

    for case in cases:
        answer = ""
        try:
            answer = ask(case.get("question", ""))
            result = mark_result(answer, case.get("expected", ""))
        except Exception as exc:
            answer = f"ERROR: {exc}"
            result = "FAIL"

        counts[result] += 1
        print(
            f"[{result}] ID: {case.get('id')} | "
            f"Category: {case.get('category')} | "
            f"Q: {case.get('question')} | "
            f"A: {first_100(answer)}"
        )

    print(
        f"\nSummary: {counts['PASS']} passed, "
        f"{counts['FAIL']} failed, "
        f"{counts['MANUAL']} need manual review"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
