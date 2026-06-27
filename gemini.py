"""
EkoBot – Context-driven RAG pipeline
LLM: OpenRouter  |  Vector store: ChromaDB  |  Collection: ieu_knowledge_base
"""

import os

import chromadb
from openai import OpenAI

from catalog import get_ieu_faculty_options
from embeddings import SentenceTransformerEmbeddingFunction
from language_utils import detect_language, get_fallback_response

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_MODEL = "meta-llama/llama-3-8b-instruct:free"
_COLLECTION_NAME = "ieu_knowledge_base"
_CONTEXT_DOCS = 4  # top-k chunks retrieved from ChromaDB
_MAX_CONTEXT_CHARS = 12_000
_MAX_HISTORY_TURNS = 6  # message pairs kept in the LLM prompt

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

_embedding_fn = SentenceTransformerEmbeddingFunction()

try:
    _chroma = chromadb.PersistentClient(path="./db")
    collection = _chroma.get_or_create_collection(
        _COLLECTION_NAME, embedding_function=_embedding_fn
    )
    print(f"[ChromaDB] Connected to '{_COLLECTION_NAME}' ({collection.count()} docs)")
except Exception as _e:
    print(f"[ChromaDB] Warning – could not connect: {_e}")
    collection = None

try:
    _llm = OpenAI(
        base_url=_OPENROUTER_BASE,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    print("[OpenRouter] Client ready")
except Exception as _e:
    print(f"[OpenRouter] Warning – could not initialise client: {_e}")
    _llm = None


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────


def _query_context(question: str, n_results: int = _CONTEXT_DOCS) -> list:
    """Return the top-n document texts from ChromaDB for *question*."""
    if collection is None:
        return []
    try:
        results = collection.query(query_texts=[question], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception as e:
        print(f"[ChromaDB] Query error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────


def _build_system_prompt(
    context_docs: list,
    session_language: str,
    client_type: str,
) -> str:
    context = "\n\n---\n\n".join(context_docs)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n...[truncated]"

    lang_name = "Turkish" if session_language == "tr" else "English"
    mobile_note = (
        "\n- Mobile clients: use plain text only — no markdown headers or bullet points."
        if (client_type or "").strip().lower() == "mobile"
        else ""
    )

    return f"""You are EkoBot, an AI academic assistant for İzmir University of Economics (IEU).

Relevant IEU database documents are provided below as context. Your job is to answer the student's question STRICTLY from that context.

STRICT RULES:
1. Answer ONLY from the provided context. NEVER use outside knowledge or hallucinate facts.
2. If the answer is not in the context respond exactly with:
   - Turkish: "Bu konuda bilgim yok."
   - English: "I don't have information on that."
3. NEVER invent course names, codes, credits, instructors, descriptions, prerequisites, or academic rules.
4. ALWAYS respond in {lang_name} — match the language of the student's latest message exactly. Never mix languages.
5. Be concise, friendly, and direct — like a helpful academic advisor.
6. Curriculum/semester planning questions require: faculty, department, year of study, semester (Fall/Spring), and enrollment year. If any of these are missing, ask for them and prefix your ENTIRE response with the literal token FOLLOWUP_NEEDED: (no space before the colon, nothing before the prefix).
7. Short continuation replies ('yes', 'go on', 'evet', 'devam et', 'and?') — continue the last topic in conversation history; do NOT start a new topic.
8. For course code lookups, report department info exactly as shown in the context. Do not correct or guess department prefixes.
9. When asked about grading or evaluation, clearly list the evaluation breakdown (Midterm %, Final %, etc.) from the course data if available.{mobile_note}

=== IEU DATABASE CONTEXT ===
{context}
=== END CONTEXT ===
"""


def _call_llm(
    question: str,
    context_docs: list,
    history: list,
    client_type: str = None,
    session_language: str = "tr",
) -> str:
    if _llm is None:
        return get_fallback_response(session_language)

    system_prompt = _build_system_prompt(context_docs, session_language, client_type)

    messages = [{"role": "system", "content": system_prompt}]
    # Include recent conversation turns for continuity
    for msg in (history or [])[-_MAX_HISTORY_TURNS:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    print(f"  Calling LLM ({_MODEL}) with {len(context_docs)} context doc(s)...")
    try:
        resp = _llm.chat.completions.create(
            model=_MODEL,
            messages=messages,
            max_tokens=1000,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return answer if answer else get_fallback_response(session_language)
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return get_fallback_response(session_language)


# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP OPTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _t(en: str, tr: str, prefer_tr: bool) -> str:
    return tr if prefer_tr else en


def _detect_followup_type(answer_lower: str) -> str | None:
    """
    Infer which student profile field the LLM is asking about from its
    response text so the UI can render the right quick-reply buttons.
    """
    if any(
        p in answer_lower
        for p in [
            "which faculty",
            "hangi fakülte",
            "hangi fakülteye",
            "hangi okul",
            "which school",
            "fakülteniz",
        ]
    ):
        return "faculty"

    if any(
        p in answer_lower
        for p in [
            "which department",
            "hangi bölüm",
            "hangi bolüm",
            "bölümünüz",
            "bolumunuz",
            "which dept",
        ]
    ):
        return "dept"

    if any(
        p in answer_lower
        for p in [
            "entry year",
            "starting year",
            "enrollment year",
            "hangi yıl",
            "hangi yil",
            "hangi yılda kayıt",
            "hangi yilda kayit",
            "universiteye hangi yıl",
            "curriculum version",
        ]
    ):
        return "start_year"

    if any(
        p in answer_lower
        for p in [
            "which semester are you in",
            "fall or spring",
            "hangi dönem",
            "hangi donem",
            "güz veya bahar",
            "guz veya bahar",
        ]
    ):
        return "semester"

    if any(
        p in answer_lower
        for p in [
            "which year of your program",
            "which year are you in",
            "what year are you in",
            "hangi sınıf",
            "hangi sinif",
            "kacinci sinif",
            "kaçıncı sınıf",
        ]
    ):
        return "year_term"

    return None


def _get_followup_options(followup_type: str, prefer_tr: bool) -> list | None:
    """Return quick-reply option dicts ``{label, value}`` for *followup_type*."""
    if followup_type == "faculty":
        return get_ieu_faculty_options(prefer_tr=prefer_tr)

    if followup_type == "semester":
        return [
            {"label": _t("Fall", "Güz", prefer_tr), "value": "Fall"},
            {"label": _t("Spring", "Bahar", prefer_tr), "value": "Spring"},
        ]

    if followup_type == "year_term":
        return [
            {"label": _t("Year 1", "1. Sınıf", prefer_tr), "value": "1"},
            {"label": _t("Year 2", "2. Sınıf", prefer_tr), "value": "2"},
            {"label": _t("Year 3", "3. Sınıf", prefer_tr), "value": "3"},
            {"label": _t("Year 4", "4. Sınıf", prefer_tr), "value": "4"},
        ]

    # dept and start_year cannot be enumerated without knowing faculty/dept;
    # return None so the UI falls back to free-text input.
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def ask_with_options(
    question: str,
    history: list = None,
    client_type: str = None,
    session_language: str = "tr",
) -> dict:
    """
    Main entry point consumed by main.py.

    Returns
    -------
    dict with keys:
        answer       – bot's text response (str)
        is_followup  – True when the bot needs more info from the student (bool)
        options      – list of {label, value} quick-reply dicts, or None
        followup_mode – 'replace' | None
    """
    history = history or []
    session_language = session_language or detect_language(question)
    prefer_tr = session_language == "tr"

    safe_q = (question or "").encode("ascii", "backslashreplace").decode("ascii")
    print(f"\n[EkoBot] Question: {safe_q}")

    # ── Step 1: retrieve relevant context from ChromaDB ──────────────────────
    context_docs = _query_context(question)
    print(f"  Retrieved {len(context_docs)} context doc(s) from ChromaDB")

    # ── Step 2: call the LLM with injected context ───────────────────────────
    raw_answer = _call_llm(
        question,
        context_docs,
        history,
        client_type=client_type,
        session_language=session_language,
    )

    # ── Step 3: parse the LLM response ───────────────────────────────────────
    _PREFIX = "FOLLOWUP_NEEDED:"
    is_followup = raw_answer.startswith(_PREFIX)
    clean_answer = raw_answer[len(_PREFIX) :].strip() if is_followup else raw_answer

    # ── Step 4: generate quick-reply buttons when a follow-up is needed ──────
    options = None
    if is_followup:
        ft = _detect_followup_type(clean_answer.lower())
        if ft:
            options = _get_followup_options(ft, prefer_tr)

    return {
        "answer": clean_answer,
        "is_followup": is_followup,
        "options": options,
        "followup_mode": "replace" if is_followup else None,
    }
