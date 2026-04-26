from groq import Groq
import chromadb
import json
import os
import re

from embeddings import SimpleHashEmbeddingFunction
from parsing import (
    VALID_DEPTS,
    _dedupe_keep_order,
    _normalize_term_value,
    detect_department,
    detect_department_from_context,
    extract_course_codes,
    extract_failed_courses_from_context,
    extract_gpa_from_context,
    extract_starting_year,
    extract_starting_year_from_history,
    extract_starting_year_loose,
    extract_year_and_term,
    extract_year_and_term_from_history,
    extract_year_only,
    extract_term_only,
    get_credit_limit_by_gpa,
    get_last_bot_question_type,
    get_latest_planning_tuple_from_history,
    get_latest_substantive_user_question,
    is_course_lookup_question,
    is_graduation_question,
    is_regulations_question,
    is_semester_question,
    normalize_department_code,
    wants_both_semesters,
    is_out_of_scope_question,
)
from catalog import (
    _semester_sort_index,
    get_available_entry_years,
    get_credit_load_rules,
    get_curriculum_course_positions,
    get_curriculum_courses,
    get_electives,
    get_graduation_text,
    get_regulations,
    has_multiple_curriculum_variants,
)
from planning import (
    _safe_alternative_courses_for_term,
    build_ects_aware_recovery_schedule,
    build_suggested_recovery_order,
    get_blocked_dependency_chain,
    get_scoped_offered_terms,
    get_scoped_prerequisite_maps,
)
from retrieval import format_course_direct_response, get_context_for_codes

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    client_groq = Groq(api_key=GROQ_API_KEY)
else:
    client_groq = None
    print("WARNING: GROQ_API_KEY is not set. LLM fallback disabled.")

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("ieu_courses", embedding_function=SimpleHashEmbeddingFunction())


def _extract_json_object(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def parse_request_slots(question, history, last_bot_q):
    """
    Parse the user's request into intent and slot values.
    Uses heuristics first, then an LLM fallback for messy natural language.
    """
    anchor_question = None
    if last_bot_q in ["dept", "year_term", "start_year"]:
        anchor_question = get_latest_substantive_user_question(history)

    analysis_question = question
    if anchor_question:
        analysis_question = anchor_question + " " + question

    result = {
        "analysis_question": analysis_question,
        "intent": None,
        "department": None,
        "year": None,
        "term": None,
        "start_year": None,
    }

    # Heuristic pass
    dept = detect_department_from_context(
        analysis_question,
        history,
        strict=True,
        allow_short_code_reply=(last_bot_q == "dept"),
    )
    if dept:
        result["department"] = dept

    year, term = extract_year_and_term(analysis_question)
    if not year or not term:
        h_year, h_term = extract_year_and_term_from_history(history)
        year = year or h_year
        term = term or h_term
    result["year"] = year
    result["term"] = term

    # Check the raw reply first so answers like "2018" are not obscured by the
    # anchor question we may have prepended to `analysis_question`.
    start_year = extract_starting_year_loose(question)
    if not start_year:
        start_year = extract_starting_year_loose(analysis_question)
    if not start_year:
        start_year = extract_starting_year_from_history(history)
    result["start_year"] = start_year

    q_lower = analysis_question.lower()
    if is_graduation_question(analysis_question):
        result["intent"] = "graduation"
    elif is_regulations_question(analysis_question):
        result["intent"] = "regulations"
    elif is_semester_question(analysis_question) or any(k in q_lower for k in ["which course", "which courses", "what course", "what courses", "should i take", "i should take", "hangi ders"]):
        result["intent"] = "planning"

    # LLM normalization for likely academic questions.
    likely_academic = any([
        result["intent"] in ["planning", "graduation", "regulations"],
        last_bot_q in ["dept", "year_term", "start_year"],
        bool(re.search(r"\b(course|courses|semester|semester\b|graduation|graduate|curriculum|curriculum\b|department|major|requirements)\b", q_lower)),
    ])

    needs_help = (
        result["intent"] in ["planning", "graduation"] and not result["department"]
    ) or (
        result["intent"] == "planning" and result["department"] and (not result["year"] or not result["term"])
    ) or (
        result["intent"] == "planning" and result["department"] and (has_multiple_curriculum_variants(result["department"]) or len(get_available_entry_years(result["department"])) > 1) and not result["start_year"]
    )

    if likely_academic and client_groq is not None:
        try:
            parser_prompt = f"""You are an IEU student-request parser.
Return ONLY valid JSON, no markdown, no explanation.

Possible intents: planning, graduation, regulations, course_lookup, other
Possible departments: {", ".join(VALID_DEPTS)}

Rules:

Return JSON with exactly these keys:
intent, department, year, term, start_year, confidence

Current user message: {question}
Latest substantive user question: {anchor_question or "null"}
Last bot question type: {last_bot_q or "null"}
Conversation history (recent): {json.dumps(history[-12:], ensure_ascii=False)}
"""
            response = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,
                messages=[
                    {"role": "system", "content": parser_prompt},
                    {"role": "user", "content": "Parse the request now."},
                ],
            )
            parsed = _extract_json_object(response.choices[0].message.content)
            if isinstance(parsed, dict):
                intent = (parsed.get("intent") or "").strip().lower() or None
                if intent in {"planning", "graduation", "regulations", "course_lookup", "other"}:
                    result["intent"] = intent

                parsed_dept = normalize_department_code(parsed.get("department"))
                if parsed_dept and parsed_dept in VALID_DEPTS + ["HEM"]:
                    result["department"] = parsed_dept

                if parsed.get("year") is not None:
                    try:
                        result["year"] = int(parsed.get("year"))
                    except Exception:
                        pass

                parsed_term = _normalize_term_value(parsed.get("term"))
                if parsed_term:
                    result["term"] = parsed_term

                if parsed.get("start_year") is not None:
                    try:
                        result["start_year"] = int(parsed.get("start_year"))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  request parser fallback error: {e}")

    # Final normalization.
    if result["department"]:
        result["department"] = normalize_department_code(result["department"])
    result["term"] = _normalize_term_value(result["term"])
    return result


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(question, context_docs, history):
    if client_groq is None:
        return "I don't have that information in the IEU database."

    context = "\n\n---\n\n".join(context_docs)
    # truncate context to avoid token limit (max ~8000 chars)
    if len(context) > 8000:
        context = context[:8000] + "\n...[truncated]"
    print(f"  Calling LLM with {len(context_docs)} context doc(s)...")

    messages = [
        {
            "role": "system",
            "content": """You are EkoBot, an academic assistant for İzmir University of Economics (IEU).

You are EkoBot. You have been given relevant IEU data as context.
Reason over ALL the context provided to give the best possible answer.
If the student's question combines multiple things (GPA + department +
year + semester), answer ALL parts of the question together naturally.
If you need information not in the context, ask ONE follow-up question.
Never say you don't have information if the answer can be reasoned
from the context provided.

STRICT RULES:
1. Answer ONLY using the context provided. Never use your own knowledge.
2. If the answer is not in the context, say: 'I don't have that information in the IEU database.'
3. NEVER make up course names, codes, credits, instructors, or descriptions.
4. If multiple courses are asked about, answer each one separately with a clear label.
5. If a course code looks like a typo, mention the correction.
6. Always reply in the same language the student used (Turkish or English).
7. Be concise, friendly, and direct — like a helpful academic advisor.
8. For direct course code requests (e.g. 'tell me about CE 326'), treat it as a direct lookup.
    Do NOT argue with the code prefix or claim it "should" belong to another department.
    Report the department exactly as shown in the provided course record.
9. When the user gives a short follow-up reply like 'yes', 'sure',
   'tell me more', 'go ahead', or 'and?' — look at the conversation
   history to find what course or topic was being discussed, and
   continue answering about THAT same topic. Never search for new
   courses when the user is clearly continuing the same conversation.
   Example: if the bot just explained SE 350 and asked 'would you
   like to know more?' and user says 'yes' — answer more about
   SE 350, do not search for something new.

IEU DEPARTMENT CODES:
- CE = Computer Engineering
- SE = Software Engineering
- EEE = Electrical & Electronics Engineering
- IE = Industrial Engineering
- BME = Biomedical Engineering
- ME = Mechanical Engineering
- CIVE = Civil Engineering
- MCE = Mechatronics Engineering

ELECTIVE SLOTS:
- ELEC 001-010 are placeholder slots in the curriculum, not real course names.
- When elective slots appear, explain naturally: 'You have X elective slots this semester.
    Would you like to see what courses you can choose for those slots?'

GPA AND CREDIT LOAD RULES:

Normal semester load is typically around 32 ECTS.
If GPA is between 2.00 and 2.49, student may take up to +6 extra ECTS (up to 38 ECTS).
If GPA is 2.50 or above, student may take up to +12 extra ECTS (up to 44 ECTS).
If GPA < 2.00, student takes normal course load with maximum 32 ECTS.
Always be specific and helpful — do not just say you don't have the information.""",
        }
    ]

    for msg in history[-12:]:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": (
            f"=== IEU DATABASE CONTEXT (use ONLY this) ===\n{context}\n"
            f"=== END CONTEXT ===\n\n"
            f"Student question: {question}\n\n"
            f"Answer ONLY from the context above."
        ),
    })

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# CONVERSATION STATE & PROFILING HELPERS
# ─────────────────────────────────────────────

def extract_profile_from_history(history):
    """
    Extract what we know about the user from conversation history.
    Returns: {department, year, term, start_year, gpa, erasmus, failed_courses}
    """
    profile = {
        "department": None,
        "year": None,
        "term": None,
        "start_year": None,
        "gpa": None,
        "erasmus": None,
        "failed_courses": None,
    }

    # Scan history for extracted values
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")

        # Department
        if not profile["department"]:
            profile["department"] = detect_department_from_context(content, history, strict=True)

        # Year and term
        if not profile["year"] or not profile["term"]:
            y, t = extract_year_and_term(content)
            y = y or extract_year_only(content)
            t = t or extract_term_only(content)
            if y and not profile["year"]:
                profile["year"] = y
            if t and not profile["term"]:
                profile["term"] = t

        # Start year
        if not profile["start_year"]:
            profile["start_year"] = extract_starting_year_loose(content)

        # GPA
        if not profile["gpa"]:
            profile["gpa"] = extract_gpa_from_context(content, history)

        # Erasmus/international status
        if not profile["erasmus"]:
            content_lower = content.lower()
            if any(word in content_lower for word in ["erasmus", "exchange", "international", "international student"]):
                profile["erasmus"] = any(word in content_lower for word in ["erasmus", "exchange"])

        # Failed courses
        if not profile["failed_courses"]:
            profile["failed_courses"] = extract_failed_courses_from_context(content, history, None)

    return profile


def get_next_profiling_question(profile, question_lower):
    """
    Determine which profiling question should be asked next.
    Returns: (question_string, next_question_type) or (None, None) if profiling complete
    """
    # Required: department first
    if not profile["department"]:
        return (
            "Hi! 👋 I'm EkoBot, your IEU academic assistant.\n\n"
            "To help you find the right courses and graduation requirements, I need to understand your situation.\n\n"
            "What department are you in? (e.g., CE, SE, EEE, IE, BME, ME, NURSING, etc.)",
            "department",
        )

    # Required: year and semester
    if not profile["year"] or not profile["term"]:
        return (
            "Got it! 👍\n\n"
            "What year and semester are you currently in?\n"
            "(e.g., '2nd year spring', 'third year fall', etc.)",
            "year_term",
        )

    # If curriculum has variants, ask for start year
    dept = profile["department"]
    if (len(get_available_entry_years(dept)) > 1 or has_multiple_curriculum_variants(dept)) and not profile["start_year"]:
        return (
            "Which year did you start university? (e.g., 2022, 2023)\n"
            "This helps me show you the right curriculum version.",
            "start_year",
        )

    # All required profiling is done
    return (None, None)


def _profiling_followup_message(question_type):
    if question_type == "department":
        return "FOLLOWUP_NEEDED: Which department are you in? (major) (e.g. CE, SE, EEE, IE, BME, ME, Nursing)"
    if question_type == "year_term":
        return "FOLLOWUP_NEEDED: Which year and semester are you in? (e.g. 2nd year spring)"
    if question_type == "start_year":
        return "FOLLOWUP_NEEDED: Which year did you start university? (e.g. 2023)"
    return None


def _is_short_ack(text):
    t = (text or "").strip().lower()
    if not t:
        return False
    if len(t.split()) > 3:
        return False
    return t in {
        "yes", "yeah", "yep", "ok", "okay", "sure", "please",
        "go ahead", "continue", "more", "and", "and?",
    }


def _maybe_elective_followup(question, history):
    if not _is_short_ack(question):
        return None

    last_assistant = None
    for msg in reversed(history[-12:]):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    if not last_assistant:
        return None

    content_lower = last_assistant.lower()
    if "elective slot" not in content_lower and "courses you can choose" not in content_lower:
        return None

    latest_plan = get_latest_planning_tuple_from_history(history)
    if not latest_plan:
        return None

    dept, _, _ = latest_plan
    electives = get_electives(dept)
    if not electives:
        return "I don't have elective options for that department in the IEU database."

    lines = [f"Available electives for {dept}:"]
    for item in electives[:20]:
        lines.append(f"- {item['code']}: {item['name']}")
    return "\n".join(lines)


def _format_profile_context_ack(profile):
    if not profile:
        return None

    parts = []
    dept = profile.get("department")
    year = profile.get("year")
    term = profile.get("term")
    start_year = profile.get("start_year")
    failed_courses = _dedupe_keep_order(profile.get("failed_courses") or [])

    if dept:
        parts.append(dept)
    if year and term:
        parts.append(f"Year {year} {term}")
    elif year:
        parts.append(f"Year {year}")
    elif term:
        parts.append(term)
    if start_year:
        parts.append(f"started {start_year}")
    if failed_courses:
        parts.append("failed " + ", ".join(failed_courses))

    if not parts:
        return None

    return (
        "Got it - I've noted your profile: "
        + ", ".join(parts)
        + ". I'll keep that in mind for future course planning."
    )


def _looks_like_profile_update_only(question, history, last_bot_q, profile=None):
    if last_bot_q in {"dept", "year_term", "start_year", "failed_courses"}:
        return False

    q = (question or "").strip().lower()
    if not q:
        return False

    if "?" in q:
        return False

    if re.search(r"\b(what|which|how|why|when|where|who|tell|show|can|should|do|does|is|are)\b", q):
        return False

    if is_course_lookup_question(question) or is_semester_question(question) or is_regulations_question(question) or is_graduation_question(question):
        return False

    if profile is None:
        profile = extract_profile_from_history(list(history or []) + [{"role": "user", "content": question}])

    return any(
        profile.get(field)
        for field in ["department", "year", "term", "start_year", "failed_courses"]
    )


def _looks_turkish(text):
    t = text or ""
    tl = t.lower()
    return any(w in tl for w in ["kredi", "ders", "not", "ortalama", "gano", "bahar", "guz", "mezun", "kural", "yonetmelik"])


def _t(en_text, tr_text, prefer_tr):
    return tr_text if prefer_tr else en_text


def _extract_loose_gpa_value(text, allow_integer=False):
    """
    Extract a GPA-like float (0.00-4.00) from short follow-up replies such as:
    - "1.9"
    - "what about 1.9"
    - "3" when `allow_integer` is enabled for a GPA/credit follow-up context.

    We intentionally require a decimal separator unless the user explicitly
    mentions GPA/CGPA/GANO/ortalama, or we are already in a GPA follow-up
    context, to avoid confusing "3rd" (year) with "3.0" (GPA).
    """
    if not text:
        return None
    t = (text or "").strip().lower()

    m = re.search(r"\b([0-4](?:[.,]\d{1,2})?)\b", t)
    if not m:
        return None

    raw = m.group(1)
    has_decimal = "." in raw or "," in raw
    if not has_decimal and not allow_integer and not any(k in t for k in ["gpa", "cgpa", "gano", "ortalama"]):
        return None

    try:
        v = float(raw.replace(",", "."))
    except Exception:
        return None
    return v


def _dedupe_texts(items):
    seen = set()
    out = []
    for item in items or []:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _extract_recent_course_codes_from_history(history, limit=12):
    for msg in reversed((history or [])[-limit:]):
        content = msg.get("content") or ""
        codes = extract_course_codes(content)
        if codes:
            return _dedupe_keep_order(codes)
    return []


def _looks_like_contextual_followup(question):
    q = (question or "").strip().lower()
    if not q:
        return False
    if _is_short_ack(question):
        return True

    cues = [
        "what about it",
        "what about that",
        "how about it",
        "how about that",
        "more details",
        "tell me more",
        "details about",
        "prerequisites?",
        "prereq?",
        "and the",
        "and what",
    ]
    return any(cue in q for cue in cues)


def _build_semantic_query_texts(question, full_question, parsed, history, codes, failed_courses, gpa_value):
    queries = []

    def add(value):
        text = " ".join(str(value or "").split()).strip()
        if text:
            queries.append(text)

    add(question)
    if full_question != question:
        add(full_question)

    latest = get_latest_substantive_user_question(history)
    if latest and latest not in {question, full_question}:
        add(latest)

    dept = parsed.get("department")
    year = parsed.get("year")
    term = parsed.get("term")
    start_year = parsed.get("start_year")
    intent = parsed.get("intent")

    if dept and year and term:
        add(f"{dept} Year {year} {term} semester curriculum courses")
    elif dept:
        add(f"{dept} curriculum courses")

    if dept:
        add(f"{dept} graduation requirements")
        add(f"{dept} electives")

    if start_year:
        add(f"{dept or ''} start year {start_year} curriculum")

    if codes:
        add(" ".join(codes) + " course prerequisites credits description")

    if failed_courses:
        add(" ".join(failed_courses) + " failed courses retake prerequisites alternatives")

    if gpa_value is not None:
        add(f"GPA {gpa_value:.2f} credit limit rules")

    if intent:
        add(f"{intent} IEU academic assistant question")

    return _dedupe_texts(queries)[:6]


def _query_relevant_context(collection_obj, query_texts, n_results=3, max_docs=6):
    if not query_texts:
        return []

    try:
        results = collection_obj.query(query_texts=query_texts, n_results=n_results)
    except Exception as e:
        print(f"  semantic context query error: {e}")
        return []

    docs = []
    for batch in results.get("documents") or []:
        for doc in batch or []:
            if not doc:
                continue
            if doc in docs:
                continue
            docs.append(doc)
            if len(docs) >= max_docs:
                return docs
    return docs


def _build_conversation_snapshot(question, parsed, history, codes, failed_courses, gpa_value, last_bot_q):
    lines = []
    dept = parsed.get("department")
    year = parsed.get("year")
    term = parsed.get("term")
    start_year = parsed.get("start_year")
    intent = parsed.get("intent")

    if dept:
        lines.append(f"Department: {dept}")
    if year and term:
        lines.append(f"Current semester: Year {year} {term}")
    elif year:
        lines.append(f"Current year: {year}")
    elif term:
        lines.append(f"Current term: {term}")
    if start_year:
        lines.append(f"Starting year: {start_year}")
    if gpa_value is not None:
        lines.append(f"GPA: {gpa_value:.2f}")
    if codes:
        lines.append("Course codes mentioned: " + ", ".join(codes))
    if failed_courses:
        lines.append("Failed courses: " + ", ".join(failed_courses))
    latest = get_latest_substantive_user_question(history)
    if latest and latest.strip().lower() != (question or "").strip().lower():
        lines.append(f"Latest substantive user question: {latest}")
    if intent:
        lines.append(f"Detected intent: {intent}")
    if last_bot_q:
        lines.append(f"Last bot follow-up: {last_bot_q}")

    if not lines:
        return None
    return "=== CONVERSATION SNAPSHOT ===\n" + "\n".join(lines)

    if 0.0 <= v <= 4.0:
        return v
    return None


def _is_credit_load_question(text):
    t = (text or "").lower()
    return any(k in t for k in [
        "how many credits", "credits can i take", "credit limit", "credit load", "ects",
        "kredi", "ne kadar kredi", "kac kredi", "kaç kredi",
    ])


def _format_regulations_answer(question, history, gpa_value):
    regs = get_regulations()
    if not regs:
        return None

    prefer_tr = _looks_turkish(question)
    t = (question or "").lower()

    # Credit load / ECTS limits
    if _is_credit_load_question(question):
        if gpa_value is None:
            return _t(
                "FOLLOWUP_NEEDED: What is your CGPA/GPA (0.00-4.00)?",
                "FOLLOWUP_NEEDED: GANO/CGPA ortalaman kac? (0.00-4.00)",
                prefer_tr,
            )

        max_ects = get_credit_limit_by_gpa(gpa_value)
        rules = (regs.get("credit_load") or {}).get("rules_by_gpa") or {}
        if gpa_value >= 2.5:
            rule = rules.get("2.50_and_above") or ""
        elif gpa_value >= 2.0:
            rule = rules.get("2.00_to_2.49") or ""
        else:
            rule = rules.get("below_threshold") or ""

        lines = [
            _t(
                f"With GPA {gpa_value:.2f}, your maximum course load is {max_ects} ECTS per semester.",
                f"GPA {gpa_value:.2f} ile donemlik maksimum ders yukunuz {max_ects} ECTS.",
                prefer_tr,
            )
        ]
        if rule:
            lines.append(_t(f"Rule: {rule}", f"Kural: {rule}", prefer_tr))
        return "\n".join(lines)

    # Graduation minima / academic standing
    if any(k in t for k in ["probation", "academic standing", "basari kosulu", "basari", "akademik"]):
        standing = regs.get("academic_standing") or {}
        thresholds = standing.get("probation_thresholds") or {}
        lines = [_t("Academic standing (probation) thresholds:", "Akademik durum (probation) esikleri:", prefer_tr)]
        for k in ["end_of_year_1", "end_of_year_2", "end_of_year_3"]:
            if thresholds.get(k):
                lines.append(f"- {thresholds[k]}")
        return "\n".join(lines) if len(lines) > 1 else None

    if any(k in t for k in ["graduate", "graduation", "mezun", "mezuniyet", "finish", "complete"]):
        standing = regs.get("academic_standing") or {}
        min_gpa = standing.get("graduation_min_gpa")
        ects = standing.get("graduation_ects") or {}
        lines = []
        if min_gpa is not None:
            lines.append(_t(f"Minimum CGPA to graduate: {min_gpa:.2f}/4.00.", f"Mezuniyet icin minimum CGPA: {min_gpa:.2f}/4.00.", prefer_tr))
        if ects:
            if ects.get("undergraduate_4year") is not None:
                lines.append(_t(f"Typical 4-year undergraduate ECTS requirement: {ects['undergraduate_4year']} ECTS.", f"Tipik 4 yillik lisans ECTS gereksinimi: {ects['undergraduate_4year']} ECTS.", prefer_tr))
        return "\n".join(lines) if lines else None

    # Withdrawal (W) rules
    if any(k in t for k in ["withdraw", "withdrawal", " w ", "w grade", "w?", "ders birak", "birakma"]):
        w = regs.get("withdrawal") or {}
        lines = [_t("Withdrawal (W) limits:", "Ders birakma (W) limitleri:", prefer_tr)]
        if w.get("max_per_semester") is not None:
            lines.append(_t(f"- Max per semester: {w['max_per_semester']}", f"- Donem basina maksimum: {w['max_per_semester']}", prefer_tr))
        if w.get("max_during_undergraduate") is not None:
            lines.append(_t(f"- Max during undergraduate: {w['max_during_undergraduate']}", f"- Lisans boyunca maksimum: {w['max_during_undergraduate']}", prefer_tr))
        if w.get("restrictions"):
            lines.append(_t(f"- Restrictions: {w['restrictions']}", f"- Kisitlar: {w['restrictions']}", prefer_tr))
        return "\n".join(lines) if len(lines) > 1 else None

    # Grading scale / special grades
    if any(k in t for k in ["grade", "grading", "letter grade", "coefficient", "harf", "not sistemi", "aa", "ba", "bb", "cb", "cc", "dc", "dd", "fd", "ff", "na", "ex", "ni", "incomplete"]):
        grading = regs.get("grading") or {}
        scale = grading.get("scale") or []
        special = grading.get("special_grades") or {}

        lines = [_t("IEU letter grade scale (points -> coefficient):", "IEU harf notu sistemi (puan -> katsayi):", prefer_tr)]
        for item in scale:
            try:
                coeff = float(item.get("coefficient"))
                coeff_str = f"{coeff:.2f}"
            except Exception:
                coeff_str = str(item.get("coefficient"))
            lines.append(f"- {item.get('grade')}: {item.get('points')} -> {coeff_str} ({item.get('status')})")

        # Only add special grades if asked about them explicitly (keeps answers short).
        if any(k in t for k in ["na", "ex", "ni", "incomplete", "withdraw", "w", "special"]):
            for k in ["NA", "W", "I", "EX", "NI", "S", "U"]:
                if k in special:
                    lines.append(f"- {k}: {special[k]}")
        return "\n".join(lines) if len(lines) > 1 else None

    return None


def _maybe_regulations_followup(question, history):
    """
    Handle short follow-ups like "what about 1.9" after a credit-limit answer.
    """
    # Accept bare replies like "3" here as well, because this function only
    # runs when the last assistant turn is already about GPA/credit load.
    gpa_loose = _extract_loose_gpa_value(question, allow_integer=True)
    if gpa_loose is None:
        return None

    last_assistant = None
    for msg in reversed(history[-12:]):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    if not last_assistant:
        return None

    la = last_assistant.lower()
    credit_context = any(k in la for k in [
        "maximum course load", "course load", "ects", "credits", "credit limit",
        "kredi",
    ]) or ("what is your cgpa" in la) or ("gano/cgpa" in la)
    if not credit_context:
        return None

    synthetic = f"My GPA is {gpa_loose:.2f}. How many credits can I take this semester?"
    return _format_regulations_answer(synthetic, history, gpa_loose)


def _profiling_year_term_followup(question):
    y, t = extract_year_and_term(question)
    y = y or extract_year_only(question)
    t = t or extract_term_only(question)

    if y and not t:
        return "FOLLOWUP_NEEDED: Which semester are you in? (Fall or Spring)"
    if t and not y:
        return "FOLLOWUP_NEEDED: Which year are you in? (e.g. 1st, 2nd, 3rd, 4th)"
    if not y and not t:
        return "FOLLOWUP_NEEDED: Which year and semester are you in? (e.g. 2nd year spring)"
    return None


def is_greeting_or_small_talk(text):
    """
    Check if the message is just a greeting or small talk, not a substantive question.
    """
    t = text.lower().strip()
    greeting_patterns = [
        "hello", "hi", "hey", "greetings",
        "how are you", "what's up", "sup",
        "good morning", "good afternoon", "good evening",
        "merhaba", "selam", "nasılsın", "nasilsin",
    ]
    return any(pattern in t for pattern in greeting_patterns) and len(t) < 50


# ─────────────────────────────────────────────
# MAIN ASK FUNCTION
# ─────────────────────────────────────────────

def ask(question, history=[]):
    # Some eval harnesses redirect stdout to non-UTF8 files on Windows.
    # Keep debug logging ASCII-only to avoid encode errors.
    safe_question = (question or "").encode("ascii", "backslashreplace").decode("ascii")
    print(f"\n[EkoBot] Question: {safe_question}")

    question_lower = question.lower().strip()

    if is_out_of_scope_question(question):
        return "I don't have that information in the IEU database."

    last_bot_q = get_last_bot_question_type(history)
    if _is_short_ack(question) and last_bot_q in {"dept", "year_term", "start_year", "failed_courses"}:
        if last_bot_q == "failed_courses":
            return "FOLLOWUP_NEEDED: Please list the failed course codes (e.g. CE 201, MATH 153)."
        followup = _profiling_followup_message({
            "dept": "department",
            "year_term": "year_term",
            "start_year": "start_year",
        }.get(last_bot_q))
        if followup:
            return followup

    elective_followup = _maybe_elective_followup(question, history)
    if elective_followup:
        return elective_followup

    regs_followup = _maybe_regulations_followup(question, history)
    if regs_followup:
        return regs_followup

    year_hint, term_hint = extract_year_and_term(question)
    academic_hint = any([
        is_course_lookup_question(question),
        is_semester_question(question),
        is_regulations_question(question),
        is_graduation_question(question),
        bool(extract_course_codes(question)),
        bool(year_hint and term_hint),
        bool(extract_starting_year_loose(question)),
    ])

    # If the bot previously asked a profiling follow-up (dept/year-term/start-year/etc),
    # treat the user's reply as part of the academic flow *when* the last substantive
    # user question was academic. This prevents replies like "nursing" from being
    # mis-routed into onboarding profiling.
    if not academic_hint and last_bot_q in {"dept", "year_term", "start_year", "failed_courses"}:
        anchor_question = get_latest_substantive_user_question(history)
        if anchor_question:
            a_year, a_term = extract_year_and_term(anchor_question)
            anchor_academic = any([
                is_course_lookup_question(anchor_question),
                is_semester_question(anchor_question),
                is_regulations_question(anchor_question),
                is_graduation_question(anchor_question),
                bool(extract_course_codes(anchor_question)),
                bool(a_year and a_term),
                bool(extract_starting_year_loose(anchor_question)),
            ])
            if anchor_academic:
                academic_hint = True

    # ── PROFILING FLOW ───────────────────────────────────────────────
    # Check if we need to complete user profiling first
    if not academic_hint:
        # Include the current user message when building the profile. Clients vary in
        # whether they include the latest message in `history`, so do it here to keep
        # behavior consistent.
        history_with_current = list(history or []) + [{"role": "user", "content": question}]
        profile = extract_profile_from_history(history_with_current)
        next_question, question_type = get_next_profiling_question(profile, question_lower)

        # If the user is just saying hello/greeting and we haven't profiled yet,
        # respond with a greeting + first profiling question
        if is_greeting_or_small_talk(question) and next_question:
            return next_question

        # If profiling is not complete, ask for the next required info
        # (unless they're answering a follow-up profiling question)
        if next_question:
            last_bot_msg = None
            if history:
                for msg in reversed(history[-3:]):
                    if msg.get("role") == "assistant":
                        last_bot_msg = msg.get("content", "")
                        break

            # Decide whether the user is answering the *next* profiling question
            # (not merely reacting to whatever the last assistant message said).
            is_answering_profiling = False
            if question_type == "department":
                is_answering_profiling = bool(detect_department(question))
            elif question_type == "year_term":
                is_answering_profiling = bool(
                    extract_year_only(question) or extract_term_only(question)
                    or any(extract_year_and_term(question))
                )
            elif question_type == "start_year":
                is_answering_profiling = bool(extract_starting_year_loose(question))
            elif question_type == "failed_courses":
                is_answering_profiling = bool(extract_failed_courses_from_context(question, history, None))

            if is_answering_profiling and question_type == "year_term":
                y_now = extract_year_only(question) or extract_year_and_term(question)[0]
                t_now = extract_term_only(question) or extract_year_and_term(question)[1]
                y_hist, t_hist = extract_year_and_term_from_history(history)
                year_final = y_now or y_hist
                term_final = t_now or t_hist

                if year_final and not term_final:
                    return "FOLLOWUP_NEEDED: Which semester are you in? (Fall or Spring)"
                if term_final and not year_final:
                    return "FOLLOWUP_NEEDED: Which year are you in? (e.g. 1st, 2nd, 3rd, 4th)"
                if not year_final and not term_final:
                    return "FOLLOWUP_NEEDED: Which year and semester are you in? (e.g. 2nd year spring)"

            # If they didn't seem to be answering the profiling question, ask again
            if not is_answering_profiling:
                followup = _profiling_followup_message(question_type)
                return followup or next_question

    # ── ACADEMIC QUESTION HANDLING ───────────────────────────────────────────────
    # From here on, proceed with the existing logic
    if _looks_like_profile_update_only(question, history, last_bot_q):
        history_with_current = list(history or []) + [{"role": "user", "content": question}]
        profile = extract_profile_from_history(history_with_current)
        next_question, question_type = get_next_profiling_question(profile, question_lower)
        if next_question:
            return _profiling_followup_message(question_type) or next_question
        profile_ack = _format_profile_context_ack(profile)
        if profile_ack:
            return profile_ack

    parsed = parse_request_slots(question, history, last_bot_q)
    full_question = parsed["analysis_question"]

    # ── gather ALL potentially relevant context ───────────────────────
    context_parts = []

    # 1. specific course codes
    # Prefer codes explicitly mentioned in the current question. Only fall back to
    # recent user history when the user didn't provide any code (e.g., "yes",
    # "what about it", "and?"). This avoids echoing previously discussed codes
    # when the student is asking about a new course.
    codes = extract_course_codes(full_question)
    if not codes and _looks_like_contextual_followup(question):
        codes = _extract_recent_course_codes_from_history(history)
    if not codes:
        all_text = full_question + " " + " ".join(
            m["content"] for m in history[-12:] if m.get("role") == "user")
        codes = extract_course_codes(all_text)
    course_contexts = []
    missing_codes = []
    if codes:
        course_contexts, missing_codes = get_context_for_codes(collection, codes)
        if course_contexts:
            ql = full_question.lower()
            if len(codes) == 1 and any(p in ql for p in ["tell me about", "more details", "details about", "information about", "what is ", "what about", "how about"]):
                return format_course_direct_response(course_contexts[0])
            context_parts.extend(course_contexts)

    # 2. department detection
    dept = parsed["department"]
    year = parsed["year"]
    term = parsed["term"]
    start_year = parsed["start_year"]
    failed_courses = extract_failed_courses_from_context(question, history, last_bot_q)
    gpa_value = extract_gpa_from_context(full_question, history)
    conversation_snapshot = _build_conversation_snapshot(
        question,
        parsed,
        history,
        codes,
        failed_courses,
        gpa_value,
        last_bot_q,
    )

    q_lower = full_question.lower()
    course_lookup_intent = is_course_lookup_question(full_question)
    planning_keywords = any(k in q_lower for k in [
        "which course", "which courses", "what course", "what courses",
        "should i take", "i should take", "hangi ders",
    ])
    graduation_intent = parsed["intent"] == "graduation" or is_graduation_question(full_question)
    regulations_intent = (
        parsed["intent"] == "regulations" or is_regulations_question(full_question)
    )
    planning_intent = (
        parsed["intent"] == "planning"
        or is_semester_question(full_question)
        or ((year and term) and planning_keywords)
        or planning_keywords
    ) and not course_lookup_intent
    both_semesters_intent = planning_intent and wants_both_semesters(full_question)

    if course_lookup_intent and not planning_intent and not graduation_intent and not regulations_intent:
        if course_contexts:
            formatted = [format_course_direct_response(doc) for doc in course_contexts]
            if missing_codes:
                missing_list = ", ".join(missing_codes)
                formatted.append(f"I don't have that information in the IEU database for {missing_list}.")
            return "\n\n".join(formatted)
        if missing_codes:
            missing_list = ", ".join(missing_codes)
            return f"I don't have that information in the IEU database for {missing_list}."

    # If user shares failed course codes after a planning answer, reuse the latest
    # planning tuple and return a retake-prioritized recommendation.
    if failed_courses and not planning_intent and not graduation_intent and not regulations_intent:
        latest_plan = get_latest_planning_tuple_from_history(history)
        if latest_plan:
            h_dept, h_year, h_term = latest_plan
            h_courses = get_curriculum_courses(h_dept, h_year, h_term, start_year=start_year)
            if h_courses:
                return _format_semester_courses(
                    h_dept,
                    h_year,
                    h_term,
                    h_courses,
                    failed_courses=failed_courses,
                    start_year=start_year,
                    gpa_value=gpa_value,
                )

    # Graduation requirement questions should not fall back to generic course search.
    if graduation_intent and not dept:
        return "FOLLOWUP_NEEDED: Which department are you in? (major) (e.g. CE, SE, EEE, IE, BME, ME, Nursing)"

    if graduation_intent and dept:
        grad_text = get_graduation_text(dept, collection=collection)
        if not grad_text:
            return "I don't have graduation requirements for that department in my database yet."
        if client_groq is None:
            return grad_text[:2000]
        context_parts.append(grad_text[:2000])

    # Deterministic regulations answers (works even without LLM).
    if regulations_intent and (not planning_intent) and (not graduation_intent):
        regs_answer = _format_regulations_answer(full_question, history, gpa_value)
        if regs_answer:
            return regs_answer

    # For course-planning questions, require missing details via follow-up
    # instead of falling into semantic search (which can hallucinate).
    # Also handle the case where user is directly answering department follow-up
    # with a short reply like "ce".
    if (not graduation_intent) and (planning_intent or (last_bot_q == "dept" and parsed["department"])):
        if not dept:
            return "FOLLOWUP_NEEDED: Which department are you in? (major) (e.g. CE, SE, EEE, IE, BME, ME, Nursing)"
        if not year or not term:
            needs_start_year = len(get_available_entry_years(dept)) > 1 or has_multiple_curriculum_variants(dept)
            return "FOLLOWUP_NEEDED: Which year and semester are you in? (e.g. 2nd year spring)"
        needs_start_year = len(get_available_entry_years(dept)) > 1 or has_multiple_curriculum_variants(dept)
        if needs_start_year and not start_year:
            return "FOLLOWUP_NEEDED: Which year did you start university? (e.g. 2023)"

    # Deterministic combined answer when the user asks both semesters of a year.
    if (not graduation_intent) and both_semesters_intent and dept and year:
        needs_start_year = len(get_available_entry_years(dept)) > 1 or has_multiple_curriculum_variants(dept)
        if needs_start_year and not start_year:
            return "FOLLOWUP_NEEDED: Which year did you start university? (e.g. 2023)"

        fall_courses = get_curriculum_courses(dept, year, "Fall", start_year=start_year)
        spring_courses = get_curriculum_courses(dept, year, "Spring", start_year=start_year)

        parts = []
        if fall_courses:
            parts.append(_format_semester_courses(dept, year, "Fall", fall_courses, failed_courses=failed_courses, start_year=start_year, gpa_value=gpa_value))
        if spring_courses:
            parts.append(_format_semester_courses(dept, year, "Spring", spring_courses, failed_courses=failed_courses, start_year=start_year, gpa_value=gpa_value))

        if parts:
            answer = "\n\n".join(parts)
            if regulations_intent and _is_credit_load_question(full_question):
                regs_answer = _format_regulations_answer(full_question, history, gpa_value)
                if regs_answer:
                    answer = answer + "\n\n" + regs_answer.replace("FOLLOWUP_NEEDED: ", "")
            return answer

    # Deterministic semester answer once dept + year + term are known.
    if (not graduation_intent) and (planning_intent or last_bot_q in ["year_term", "start_year"]) and dept and year and term:
        courses = get_curriculum_courses(dept, year, term, start_year=start_year)
        if not courses:
            if last_bot_q in ["year_term", "start_year"]:
                # Try an LLM-guided retry before giving up on curriculum matching.
                parsed_retry = parse_request_slots(full_question, history, last_bot_q)
                dept = parsed_retry["department"] or dept
                year = parsed_retry["year"] or year
                term = parsed_retry["term"] or term
                start_year = parsed_retry["start_year"] or start_year
                courses = get_curriculum_courses(dept, year, term, start_year=start_year)
        if not courses:
            return "I don't have that semester's curriculum data in my database."
        answer = _format_semester_courses(dept, year, term, courses, failed_courses=failed_courses, start_year=start_year, gpa_value=gpa_value)
        if regulations_intent and _is_credit_load_question(full_question):
            regs_answer = _format_regulations_answer(full_question, history, gpa_value)
            if regs_answer:
                answer = answer + "\n\n" + regs_answer.replace("FOLLOWUP_NEEDED: ", "")
        return answer

    if dept and year and term:
        courses = get_curriculum_courses(dept, year, term, start_year=start_year)
        if courses:
            elec_count = sum(1 for c in courses
                           if c.get("code", "").startswith("ELEC"))
            regular = [f"- {c['code']}: {c['name']}"
                      for c in courses
                      if not c.get("code", "").startswith("ELEC")]
            curr_text = (f"{dept} Year {year} {term} Semester courses:\n"
                        + "\n".join(regular))
            if elec_count:
                curr_text += (f"\n\nPlus {elec_count} elective slot(s). "
                             f"Available electives for {dept}:")
                electives = get_electives(dept)
                for e in electives[:20]:
                    curr_text += f"\n- {e['code']}: {e['name']}"
            context_parts.append(curr_text)

    # 4. graduation requirements if dept mentioned
    if (not graduation_intent) and dept and any(k in full_question.lower() for k in
                   ["graduate", "graduation", "finish", "complete",
                    "mezun", "bitir"]):
        grad_text = get_graduation_text(dept, collection=collection)
        if grad_text:
            context_parts.append(grad_text[:1000])

    # 5. regulations if GPA/credits/grades mentioned (allow mixed planning+rules questions)
    if regulations_intent or any(k in full_question.lower() for k in
        ["gpa", "credit", "grade", "kredi", "not", "ortalama",
         "how many", "can i take", "withdraw", "probation", "letter grade", "harf"]):
        context_parts.append(get_credit_load_rules())
        regs = get_regulations()
        if regs:
            # Keep this compact; the LLM can still reason over deterministic outputs above.
            context_parts.append(json.dumps(regs.get("grading", {}), ensure_ascii=False))
            context_parts.append(json.dumps(regs.get("credit_load", {}), ensure_ascii=False))
            context_parts.append(json.dumps(regs.get("withdrawal", {}), ensure_ascii=False))

    # 6. semantic search fallback
    if not context_parts:
        query_texts = _build_semantic_query_texts(
            question,
            full_question,
            parsed,
            history,
            codes,
            failed_courses,
            gpa_value,
        )
        context_parts = _query_relevant_context(collection, query_texts, n_results=3, max_docs=6)

    if not context_parts:
        return ("I don't have enough information to answer that. "
                "Could you give me more details?")

    # ── ask LLM to reason over ALL context ───────────────────────────
    # truncate to avoid token limit
    if conversation_snapshot:
        context_parts.insert(0, conversation_snapshot)

    combined = "\n\n---\n\n".join(context_parts)
    if len(combined) > 8000:
        combined = combined[:8000]

    return call_llm(full_question, [combined], history)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _format_semester_courses(dept, year, term, courses, failed_courses=None, start_year=None, gpa_value=None):
    elec_count = sum(1 for c in courses if c.get("code", "").startswith("ELEC"))
    lines = [f"{dept} - Year {year} {term} Semester:\n"]
    failed_courses = _dedupe_keep_order(failed_courses or [])
    blocked_by_code = {}

    scoped_prereq_map = {}
    scoped_reverse_graph = {}
    offered_term_by_code = {}
    positions = {}
    if failed_courses:
        scoped_prereq_map, scoped_reverse_graph = get_scoped_prerequisite_maps(
            dept,
            start_year=start_year,
        )
        offered_term_by_code = get_scoped_offered_terms(dept, start_year=start_year)
        positions = get_curriculum_course_positions(dept, start_year=start_year)
        blocked_chain_preview = get_blocked_dependency_chain(
            failed_courses,
            dept,
            year,
            term,
            start_year=start_year,
            reverse_graph=scoped_reverse_graph,
        )
        blocked_by_code = {
            item.get("code"): item.get("blocked_by")
            for item in blocked_chain_preview
            if item.get("code") and item.get("blocked_by")
        }

    for c in courses:
        code = c.get("code", "").strip()
        name = c.get("name", "").strip()
        if not code.startswith("ELEC"):
            if code in blocked_by_code:
                lines.append(
                    f"- {code}: {name} (blocked now by failed prerequisite chain via {blocked_by_code[code]})"
                )
            else:
                lines.append(f"- {code}: {name}")

    if elec_count > 0:
        lines.append(f"\nYou also have {elec_count} elective slot(s) this semester.")
        lines.append("Would you like to see what courses you can choose for those slots?")

    if failed_courses:
        current_codes = {(c.get("code") or "").strip().upper() for c in courses}
        lines.append("\nRetake priority (failed courses you mentioned):")
        for code in failed_courses:
            offered_term = (
                (positions.get(code, {}).get("term") or offered_term_by_code.get(code) or "")
            ).strip()
            offered_note = f"; offered in {offered_term}" if offered_term else ""
            if code in current_codes:
                lines.append(f"- {code} (already in this semester curriculum; prioritize retaking this course{offered_note})")
            else:
                lines.append(f"- {code} (not listed in this semester curriculum; add it in its offered term{offered_note} if your credit load allows)")

        blocked_chain = get_blocked_dependency_chain(
            failed_courses,
            dept,
            year,
            term,
            start_year=start_year,
            reverse_graph=scoped_reverse_graph,
        )
        if blocked_chain:
            lines.append("\nBlocked dependency chain (following semesters):")
            for item in blocked_chain[:20]:
                title = f"{item['code']}"
                if item.get("name"):
                    title += f" - {item['name']}"
                lines.append(
                    f"- {title} (Year {item['year']} {item['term']}) blocked by {item['blocked_by']}"
                )
            if len(blocked_chain) > 20:
                lines.append(f"- ...and {len(blocked_chain) - 20} more downstream blocked course(s).")
        else:
            lines.append("\nNo downstream prerequisite blockers were found in the available curriculum data.")

        blocked_codes = set(failed_courses)
        blocked_codes.update(item.get("code") for item in blocked_chain if item.get("code"))
        alternatives = _safe_alternative_courses_for_term(
            dept,
            year,
            term,
            current_codes,
            blocked_codes,
            positions,
            scoped_prereq_map,
            limit=8,
        )
        if alternatives:
            lines.append("\nSafe alternatives for this term (from later years, excluding courses blocked by your failed-chain):")
            for item in alternatives:
                title = f"{item['code']}"
                if item.get("name"):
                    title += f" - {item['name']}"
                lines.append(f"- {title} (Year {item['year']} {item['term']})")
        else:
            lines.append("\nNo safe same-term alternatives were found from later semesters in the available curriculum data.")

        recovery_order = build_suggested_recovery_order(
            failed_courses,
            blocked_chain,
            dept,
            start_year=start_year,
        )
        if recovery_order:
            lines.append("\nSuggested recovery order:")
            for i, item in enumerate(recovery_order[:20], start=1):
                title = item["code"]
                if item.get("name"):
                    title += f" - {item['name']}"
                semester_part = ""
                if item.get("year") and item.get("term"):
                    semester_part = f" (Year {item['year']} {item['term']})"
                lines.append(f"{i}. {title}{semester_part} [{item.get('reason', 'dependency order')}]"
)
            if len(recovery_order) > 20:
                lines.append(f"...and {len(recovery_order) - 20} more item(s).")

            current_idx = _semester_sort_index(int(year), term)
            schedule, max_ects = build_ects_aware_recovery_schedule(
                recovery_order,
                gpa_value,
                current_idx,
                prereq_map=scoped_prereq_map,
                offered_term_by_code=offered_term_by_code,
            )
            if schedule:
                gpa_note = f"GPA {gpa_value:.2f}" if gpa_value is not None else "GPA not provided"
                lines.append(f"\nECTS-aware recovery schedule ({gpa_note}, max {max_ects} ECTS/semester):")
                lines.append("Strict prerequisite sequencing is applied (dependent courses are scheduled after unmet failed prerequisites and only in their offered Fall/Spring term).")
                for sem in schedule[:8]:
                    header = f"- Year {sem['year']} {sem['term']}: {sem['total_ects']:.0f}/{max_ects} ECTS"
                    lines.append(header)
                    for item in sem["items"]:
                        title = item["code"]
                        if item.get("name"):
                            title += f" - {item['name']}"
                        ects_txt = f"{item['ects']:.0f} ECTS" if item.get("ects", 0) > 0 else "ECTS unknown"
                        lines.append(f"  - {title} [{ects_txt}; {item.get('reason', 'dependency order')}]"
)
                if len(schedule) > 8:
                    lines.append(f"- ...and {len(schedule) - 8} more semester(s) in the recovery schedule.")
    else:
        lines.append("\nIf you have any failed courses to retake, share their course codes (e.g. CE 201, MATH 153) and I will prioritize them in your plan.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# TERMINAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("EkoBot — type your question (or 'quit' to exit)")
    print("-" * 50)
    conversation_history = []

    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        answer = ask(question, conversation_history)
        print(f"\nEkoBot: {answer}")

        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})
