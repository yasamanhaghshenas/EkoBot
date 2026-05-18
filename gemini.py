from groq import Groq
import chromadb
import json
import os
import re

from embeddings import SentenceTransformerEmbeddingFunction
from language_utils import detect_language, get_fallback_response
from parsing import (
    VALID_DEPTS,
    _dedupe_keep_order,
    _normalize_term_value,
    detect_department,
    detect_department_from_context,
    detect_faculty_choice,
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
    is_gpa_credit_limit_question,
    is_regulations_question,
    is_web_info_question,
    is_semester_question,
    normalize_department_code,
    wants_both_semesters,
    is_out_of_scope_question,
)
from catalog import (
    _semester_sort_index,
    get_available_entry_years,
    get_curriculum_course_positions,
    get_curriculum_courses,
    get_curriculum_year_options,
    get_electives,
    get_ieu_department_options,
    get_ieu_faculty_options,
    get_curriculum_variant_options,
    has_multiple_curriculum_variants,
    resolve_ieu_department,
    resolve_ieu_faculty,
)
from planning import (
    _safe_alternative_courses_for_term,
    build_ects_aware_recovery_schedule,
    build_suggested_recovery_order,
    get_blocked_dependency_chain,
    get_scoped_offered_terms,
    get_scoped_prerequisite_maps,
)
from retrieval import (
    extract_course_specific_response,
    format_course_direct_response,
    get_context_for_codes,
    get_certificate_context,
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
if not os.environ.get("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set. LLM fallback disabled.")

db_client = chromadb.PersistentClient(path="./db")
collection = db_client.get_or_create_collection("ieu_courses", embedding_function=SentenceTransformerEmbeddingFunction())

_REGULATIONS_CACHE = None


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


def _get_curriculum_url_for_start_year(resolved_dept, start_year):
    """
    Pick the correct curriculum URL from resolved_dept based on start_year.
    If start_year is provided, searches curriculum.json for a matching entry_year.
    Falls back to the default curriculum_url if no match found.
    """
    if not resolved_dept or start_year is None:
        return (resolved_dept.get("curriculum_url") or "").strip() or None
    
    # Try to find the correct URL in curriculum.json for this start_year
    try:
        import json, os
        start_year_int = int(start_year)
        
        if os.path.exists("curriculum.json"):
            with open("curriculum.json", "r", encoding="utf-8") as f:
                curriculum = json.load(f)
            
            dept_code = (resolved_dept.get("department_code") or resolved_dept.get("department") or "").strip().upper()
            program_name = resolved_dept.get("program_name") or resolved_dept.get("label") or ""
            
            for item in curriculum:
                try:
                    if int(item.get("entry_year") or 0) == start_year_int:
                        # Check if department matches
                        item_dept = (item.get("department") or "").strip().upper()
                        if item_dept != dept_code:
                            continue
                        # Check if program name matches (roughly)
                        item_program = (item.get("program_name") or "").strip().lower()
                        if item_program and program_name and item_program != program_name.lower():
                            continue
                        # Found a match - return its URL
                        url = (item.get("url") or item.get("curriculum_url") or "").strip()
                        if url:
                            return url
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    
    # Fall back to default
    return (resolved_dept.get("curriculum_url") or "").strip() or None


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
        "program_name": None,
        "curriculum_url": None,
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
    # If still missing, try individual extractors (for replies like "4th spring" without "year")
    if not year:
        year = extract_year_only(analysis_question)
    if not term:
        term = extract_term_only(analysis_question)
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
    elif is_gpa_credit_limit_question(analysis_question, history):
        result["intent"] = "credit_limit"
    elif is_regulations_question(analysis_question):
        result["intent"] = "regulations"
    elif is_web_info_question(analysis_question):
        result["intent"] = "web_info"
    elif is_semester_question(analysis_question) or any(k in q_lower for k in ["which course", "which courses", "what course", "what courses", "should i take", "i should take", "hangi ders"]):
        result["intent"] = "planning"

    # LLM normalization for likely academic questions.
    likely_academic = any([
        result["intent"] in ["planning", "graduation", "regulations", "credit_limit"],
        last_bot_q in ["dept", "year_term", "start_year"],
        bool(re.search(r"\b(course|courses|semester|semester\b|graduation|graduate|curriculum|curriculum\b|department|major|requirements)\b", q_lower)),
        bool(is_web_info_question(analysis_question)),
    ])

    heuristic_department = result["department"]
    heuristic_year = result["year"]
    heuristic_term = result["term"]
    heuristic_start_year = result["start_year"]

    needs_help = (
        result["intent"] in ["planning", "graduation", "credit_limit"] and not result["department"]
    ) or (
        result["intent"] == "planning" and result["department"] and (not result["year"] or not result["term"])
    ) or (
        result["intent"] == "planning" and result["department"] and has_multiple_curriculum_variants(result["department"]) and not result["start_year"]
    )

    if likely_academic and client is not None:
        try:
            parser_prompt = f"""You are an IEU student-request parser.
Return ONLY valid JSON, no markdown, no explanation.

Possible intents: planning, graduation, regulations, credit_limit, course_lookup, web_info, other
Possible departments: {", ".join(VALID_DEPTS)}

Rules:

Return JSON with exactly these keys:
intent, department, year, term, start_year, confidence

Important:
- `year` is the student's year of study and must be 1-6.
- `start_year` is the enrollment year, such as 2020.

Current user message: {question}
Latest substantive user question: {anchor_question or "null"}
Last bot question type: {last_bot_q or "null"}
Conversation history (recent): {json.dumps(history[-12:], ensure_ascii=False)}
"""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,
                messages=[
                    {"role": "system", "content": parser_prompt},
                    {"role": "user", "content": "Parse the request now."},
                ],
                max_tokens=1000,
            )
            parsed = _extract_json_object(response.choices[0].message.content)
            if isinstance(parsed, dict):
                intent = (parsed.get("intent") or "").strip().lower() or None
                if intent in {"planning", "graduation", "regulations", "credit_limit", "course_lookup", "web_info", "other"}:
                    result["intent"] = intent

                parsed_dept = normalize_department_code(parsed.get("department"))
                if parsed_dept and parsed_dept in VALID_DEPTS + ["HEM"] and not heuristic_department:
                    result["department"] = parsed_dept

                if parsed.get("year") is not None:
                    try:
                        parsed_year = int(parsed.get("year"))
                        # Study year must be 1-6. Ignore enrollment years like 2020
                        # that can leak in from the follow-up context.
                        if 1 <= parsed_year <= 6 and heuristic_year is None:
                            result["year"] = parsed_year
                    except Exception:
                        pass

                parsed_term = _normalize_term_value(parsed.get("term"))
                if parsed_term and not heuristic_term:
                    result["term"] = parsed_term

                if parsed.get("start_year") is not None:
                    try:
                        parsed_start_year = int(parsed.get("start_year"))
                        if heuristic_start_year is None:
                            result["start_year"] = parsed_start_year
                    except Exception:
                        pass
        except Exception as e:
            print(f"  request parser fallback error: {e}")

    # Final normalization.
    if result["department"]:
        result["department"] = normalize_department_code(result["department"])
        resolved_dept = resolve_ieu_department(result["department"])
        if resolved_dept:
            result["program_name"] = (resolved_dept.get("program_name") or resolved_dept.get("label") or "").strip() or None
            result["curriculum_url"] = _get_curriculum_url_for_start_year(resolved_dept, result.get("start_year"))
    result["term"] = _normalize_term_value(result["term"])
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LLM CALL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def call_llm(question, context_docs, history, client_type=None, session_language=None):
    session_language = session_language or detect_language(question)
    prefer_tr = session_language == "tr"
    if client is None:
        return _soft_fallback_message(session_language)

    context = "\n\n---\n\n".join(context_docs)
    # truncate context to avoid token limit (increased to 15000 to preserve evaluation system)
    if len(context) > 15000:
        context = context[:15000] + "\n...[truncated]"
    print(f"  Calling LLM with {len(context_docs)} context doc(s)...")

    client_type_norm = (client_type or "").strip().lower()
    mobile_rule = ""
    if client_type_norm == "mobile":
        mobile_rule = "\n10. For mobile clients, do not use markdown headers or bullet points. Reply in plain text."
    language_rule = f"\n11. ABSOLUTE RULE: Your response language must be {session_language}. Never switch languages. Even for greetings, thanks, or one-word replies — always use {session_language}. If you find yourself writing in the other language, stop and rewrite."

    system_prompt = """You are EkoBot, an academic assistant for ؤ°zmir University of Economics (IEU).

You are EkoBot. You have been given relevant IEU data as context.
Reason over ALL the context provided to give the best possible answer.
If the student's question combines multiple things (GPA + department +
year + semester), answer ALL parts of the question together naturally.
If you need information not in the context, ask ONE follow-up question.
Never say you don't have information if the answer can be reasoned
from the context provided.

STRICT RULES:
1. Answer ONLY using the context provided. Never use your own knowledge.
2. If the answer is not in the context, say: ''
3. NEVER make up course names, codes, credits, instructors, or descriptions.
4. If multiple courses are asked about, answer each one separately with a clear label.
5. If a course code looks like a typo, mention the correction.
6. CRITICAL: Always respond in the EXACT same language the student used in their MOST RECENT message.
    If they wrote in Turkish, respond entirely in Turkish. If English, respond entirely in English.
    Never mix languages in a single response.
    "tell me about electives" = English -> respond in English.
    "seçmeli dersler" = Turkish -> respond in Turkish.
7. Be concise, friendly, and direct â€” like a helpful academic advisor.
8. For direct course code requests (e.g. 'tell me about CE 326'), treat it as a direct lookup.
    Do NOT argue with the code prefix or claim it "should" belong to another department.
    Report the department exactly as shown in the provided course record.
9. When the user gives a short follow-up reply like 'yes', 'sure',
   'tell me more', 'go ahead', or 'and?' â€” look at the conversation
   history to find what course or topic was being discussed, and
   continue answering about THAT same topic. Never search for new
   courses when the user is clearly continuing the same conversation.
    Example: if the bot just explained SE 350 and asked 'would you
    like to know more?' and user says 'yes' â€” answer more about
    SE 350, do not search for something new.

Always be specific and helpful â€” do not just say you don't have the information.
10. EVALUATION/GRADING SYSTEM RULE: If the student asks about grading, assessment, evaluation method,
    exam weights, or how they will be graded in a course, extract and clearly present the
    Evaluation System section from the course data (e.g., Midterm 40%, Final 60%, etc.).
    Present grading breakdown as a clear list if available.
""" + language_rule
    if mobile_rule:
        system_prompt += mobile_rule

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for msg in history[-6:]:
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

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content
        if not answer or answer.strip() == "" or answer.strip() == "''":
            return get_fallback_response(session_language)
        return answer
    except Exception as e:
        print(f"LLM error: {e}")
        return _soft_fallback_message(session_language)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONVERSATION STATE & PROFILING HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _curriculum_entry_year_followup(clarify_invalid_reply=False, prefer_tr=False):
    lines = [
        _t(
            "FOLLOWUP_NEEDED: Which year did you start university?",
            "FOLLOWUP_NEEDED: Hangi yılda başladın üniversiteye?",
            prefer_tr,
        ),
    ]
    if clarify_invalid_reply:
        lines.append(_t(
            "I still need your entry year, not your current class year/semester.",
            "Hâlâ giriş yılına ihtiyacım var, şu anki sınıf/dönem bilgine değil.",
            prefer_tr,
        ))
    lines.append(_t(
        "Please choose one of the options below.",
        "Aşağıdaki seçeneklerden birini seçebilirsin.",
        prefer_tr,
    ))
    return "\n".join(lines)


def _curriculum_year_followup(prefer_tr=False):
    return _t(
        "FOLLOWUP_NEEDED: Which year of your program are you in?",
        "FOLLOWUP_NEEDED: Kaçıncı sınıftasın?",
        prefer_tr,
    )


def _curriculum_semester_followup(prefer_tr=False):
    return _t(
        "FOLLOWUP_NEEDED: Are you in the Fall or Spring semester?",
        "FOLLOWUP_NEEDED: Hangi yarıyıldasın, Güz mü Bahar mı?",
        prefer_tr,
    )


def _looks_like_profile_correction(text):
    q = _normalize_chatty_text(text)
    if not q:
        return False

    correction_cues = (
        "sorry i meant",
        "sorry, i meant",
        "i meant",
        "i mean",
        "meant to say",
        "actually",
        "no wait",
        "wait no",
        "correction",
        "instead",
        "rather",
        "my bad",
        "wrong",
    )
    return any(cue in q for cue in correction_cues)


def _profile_fields_changed(previous, current):
    fields = ["faculty", "department", "program_name", "year", "term", "start_year"]
    previous = previous or {}
    current = current or {}
    return any(previous.get(field) != current.get(field) for field in fields)


def extract_profile_from_history(history):
    """
    Extract what we know about the user from conversation history.
    Returns: {department, year, term, start_year, gpa, erasmus, failed_courses}
    """
    profile = {
        "faculty": None,
        "department": None,
        "program_name": None,
        "curriculum_url": None,
        "year": None,
        "term": None,
        "start_year": None,
        "gpa": None,
        "erasmus": None,
        "failed_courses": None,
    }

    def _clear_lower_fields(level):
        if level == "faculty":
            profile["department"] = None
            profile["program_name"] = None
            profile["curriculum_url"] = None
            profile["year"] = None
            profile["term"] = None
            profile["start_year"] = None
        elif level == "department":
            profile["program_name"] = None
            profile["curriculum_url"] = None
            profile["year"] = None
            profile["term"] = None
            profile["start_year"] = None
        elif level == "year":
            profile["term"] = None

    def _set_profile_value(key, value, clear_level=None):
        if value is None:
            return False
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return False
        if profile.get(key) == value:
            return False
        profile[key] = value
        if clear_level:
            _clear_lower_fields(clear_level)
        return True

    # Scan history in order so later explicit answers override earlier wrong clicks.
    for msg in history or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "") or ""

        faculty = detect_faculty_choice(content)
        if faculty:
            # Higher-level choices reset the dependent lower-level profile data.
            _set_profile_value("faculty", faculty, clear_level="faculty")

        resolved_dept = resolve_ieu_department(content, faculty_name=profile["faculty"])
        if not resolved_dept:
            resolved_dept = resolve_ieu_department(content)

        if resolved_dept:
            dept_code = (resolved_dept.get("department_code") or "").strip().upper()
            if dept_code:
                _set_profile_value("department", normalize_department_code(dept_code), clear_level="department")
            dept_name = (resolved_dept.get("program_name") or resolved_dept.get("label") or "").strip()
            if dept_name:
                profile["program_name"] = dept_name
            new_faculty = (resolved_dept.get("faculty_label") or resolved_dept.get("faculty_sheet") or "").strip() or None
            if new_faculty:
                profile["faculty"] = new_faculty
            curriculum_url = _get_curriculum_url_for_start_year(resolved_dept, profile.get("start_year"))
            if curriculum_url:
                profile["curriculum_url"] = curriculum_url
        else:
            dept_code = detect_department(content)
            if dept_code:
                _set_profile_value("department", normalize_department_code(dept_code), clear_level="department")
                code_resolved = resolve_ieu_department(profile["department"], faculty_name=profile["faculty"])
                if not code_resolved:
                    code_resolved = resolve_ieu_department(profile["department"])
                if code_resolved:
                    dept_name = (code_resolved.get("program_name") or code_resolved.get("label") or "").strip()
                    if dept_name:
                        profile["program_name"] = dept_name
                    new_faculty = (code_resolved.get("faculty_label") or code_resolved.get("faculty_sheet") or "").strip() or None
                    if new_faculty:
                        profile["faculty"] = new_faculty
                    curriculum_url = _get_curriculum_url_for_start_year(code_resolved, profile.get("start_year"))
                    if curriculum_url:
                        profile["curriculum_url"] = curriculum_url

        if profile["department"] and not profile["program_name"]:
            code_resolved = resolve_ieu_department(profile["department"], faculty_name=profile["faculty"])
            if not code_resolved:
                code_resolved = resolve_ieu_department(profile["department"])
            if code_resolved:
                dept_name = (code_resolved.get("program_name") or code_resolved.get("label") or "").strip()
                if dept_name:
                    profile["program_name"] = dept_name
                new_faculty = (code_resolved.get("faculty_label") or code_resolved.get("faculty_sheet") or "").strip() or None
                if new_faculty:
                    profile["faculty"] = new_faculty
                curriculum_url = _get_curriculum_url_for_start_year(code_resolved, profile.get("start_year"))
                if curriculum_url:
                    profile["curriculum_url"] = curriculum_url

        # Year and term
        y, t = extract_year_and_term(content)
        y = y or extract_year_only(content)
        t = t or extract_term_only(content)
        if y is not None:
            _set_profile_value("year", y, clear_level="year")
        if t is not None:
            profile["term"] = t

        # Start year
        start_year = extract_starting_year_loose(content)
        if start_year is not None:
            profile["start_year"] = start_year
            if profile["department"]:
                resolved_dept = resolve_ieu_department(profile["department"], faculty_name=profile["faculty"])
                if not resolved_dept:
                    resolved_dept = resolve_ieu_department(profile["department"])
                if resolved_dept:
                    curriculum_url = _get_curriculum_url_for_start_year(resolved_dept, profile["start_year"])
                    if curriculum_url:
                        profile["curriculum_url"] = curriculum_url

        # GPA
        gpa_value = extract_gpa_from_context(content, history)
        if gpa_value is not None:
            profile["gpa"] = gpa_value

        # Erasmus/international status
        content_lower = content.lower()
        if any(word in content_lower for word in ["erasmus", "exchange", "international", "international student"]):
            profile["erasmus"] = any(word in content_lower for word in ["erasmus", "exchange"])

        # Failed courses
        failed_courses = extract_failed_courses_from_context(content, history, None)
        if failed_courses:
            profile["failed_courses"] = failed_courses

    # Refine curriculum_url based on start_year if we have both
    if profile["start_year"] and profile["department"]:
        resolved_dept = resolve_ieu_department(profile["department"], faculty_name=profile["faculty"])
        if not resolved_dept:
            resolved_dept = resolve_ieu_department(profile["department"])
        if resolved_dept:
            profile["curriculum_url"] = _get_curriculum_url_for_start_year(resolved_dept, profile["start_year"])

    return profile


def get_next_profiling_question(profile, question_lower, require_curriculum_details=True, prefer_tr=False):
    """
    Determine which profiling question should be asked next.
    Returns: (question_string, next_question_type) or (None, None) if profiling complete
    """
    # Required: faculty first
    if not profile["faculty"]:
        return (
            _t(
                "Hi! I can help you plan courses accurately.\n\n"
                "Could you tell me which faculty you're in? Please choose one of the options below.",
                "Merhaba! Ders planını doğru hazırlayabilmem için küçük bir bilgiye ihtiyacım var.\n\n"
                "Hangi fakültede okuyorsun? Aşağıdaki seçeneklerden birini seçebilirsin.",
                prefer_tr,
            ),
            "faculty",
        )

    # Required: department next
    if not profile["department"] or not profile["program_name"]:
        return (
            _t(
                "Great, thanks. Could you tell me which department you're in? Please choose one of the options below.",
                "Harika, teşekkürler. Hangi bölümde okuyorsun? Aşağıdaki seçeneklerden birini seçebilirsin.",
                prefer_tr,
            ),
            "dept",
        )

    if require_curriculum_details:
        # Required: curriculum entry year only when the curriculum changed.
        # Ask this before year/semester so the user can pick the curriculum bucket first.
        if not profile["start_year"] and has_multiple_curriculum_variants(
            profile["department"],
            program_name=profile.get("program_name"),
        ):
            return (
                _curriculum_entry_year_followup(prefer_tr=prefer_tr),
                "start_year",
            )

        # Required: year first, then semester
        if not profile["year"] or not profile["term"]:
            if profile["year"] and not profile["term"]:
                return (
                    _curriculum_semester_followup(prefer_tr=prefer_tr),
                    "year_term",
                )
            return (
                _curriculum_year_followup(prefer_tr=prefer_tr),
                "year_term",
            )

    # All required profiling is done
    return (None, None)


def _requires_curriculum_details(question, history):
    anchor = get_latest_substantive_user_question(history or [])
    anchor_text = anchor or question or ""
    return not is_graduation_question(anchor_text)


def _with_followup_prefix(message, prefer_tr=False, correction=False):
    if not correction or not message:
        return message
    prefix = _t("No problem. ", "Sorun degil. ", prefer_tr)
    return message.replace("FOLLOWUP_NEEDED: ", f"FOLLOWUP_NEEDED: {prefix}", 1)


def _profiling_followup_message(question_type, profile=None, prefer_tr=False, correction=False):
    if question_type == "faculty":
        return _with_followup_prefix(_t(
            "FOLLOWUP_NEEDED: Which faculty are you in? Please choose one of the options below.",
            "FOLLOWUP_NEEDED: Hangi fakültede okuyorsun?",
            prefer_tr,
        ), prefer_tr=prefer_tr, correction=correction)
    if question_type == "dept":
        return _with_followup_prefix(_t(
            "FOLLOWUP_NEEDED: Which department are you in? Please choose one of the options below.",
            "FOLLOWUP_NEEDED: Hangin bölümdesin?",
            prefer_tr,
        ), prefer_tr=prefer_tr, correction=correction)
    if question_type == "year_term":
        if profile and profile.get("year") and not profile.get("term"):
            return _with_followup_prefix(_curriculum_semester_followup(prefer_tr=prefer_tr), prefer_tr=prefer_tr, correction=correction)
        return _with_followup_prefix(_t(
            "FOLLOWUP_NEEDED: Which year of your program are you in?",
            "FOLLOWUP_NEEDED: Kaçıncı sınıftasın?",
            prefer_tr,
        ), prefer_tr=prefer_tr, correction=correction)
    if question_type == "start_year":
        return _with_followup_prefix(_curriculum_entry_year_followup(prefer_tr=prefer_tr), prefer_tr=prefer_tr, correction=correction)
    return None


def _profiling_followup_options(question_type, profile, prefer_tr=False):
    if question_type == "faculty":
        return get_ieu_faculty_options(prefer_tr=prefer_tr)
    if question_type == "dept":
        faculty = (profile or {}).get("faculty")
        if faculty:
            return get_ieu_department_options(faculty, prefer_tr=prefer_tr)
    if question_type == "year_term":
        profile = profile or {}
        dept = profile.get("department")
        if dept:
            options = get_curriculum_year_options(
                dept,
                program_name=profile.get("program_name"),
                curriculum_url=profile.get("curriculum_url"),
                prefer_tr=prefer_tr,
            )
            if options:
                return options
    if question_type == "semester":
        return [
            {"label": _t("Fall", "Guz", prefer_tr), "value": "Fall"},
            {"label": _t("Spring", "Bahar", prefer_tr), "value": "Spring"},
        ]
    if question_type == "start_year":
        profile = profile or {}
        dept = profile.get("department")
        if dept:
            options = get_curriculum_variant_options(
                dept,
                program_name=profile.get("program_name"),
            )
            if options:
                return options
            years = get_available_entry_years(
                dept,
                program_name=profile.get("program_name"),
            )
            if len(years) > 1:
                return [{"label": str(year), "value": str(year)} for year in years]
    return None


def ask_with_options(question, history=None, client_type=None, session_language=None):
    history = history or []
    answer = ask(question, history, client_type=client_type, session_language=session_language)
    is_followup = answer.startswith("FOLLOWUP_NEEDED:")
    clean_answer = answer.replace("FOLLOWUP_NEEDED:", "").strip() if is_followup else answer
    if session_language is None:
        session_language = detect_language(question)
    prefer_tr = session_language == "tr"
    is_correction = _looks_like_profile_correction(question)

    options = None
    if is_followup:
        history_with_current = list(history or []) + [{"role": "user", "content": question}]
        profile = extract_profile_from_history(history_with_current)
        lowered = clean_answer.lower()
        if any(phrase in lowered for phrase in [
            "which faculty",
            "hangi fakulte",
            "hangi fakülte",
            "hangi okul",
            "which school",
        ]):
            options = _profiling_followup_options("faculty", profile, prefer_tr=prefer_tr)
        elif any(phrase in lowered for phrase in [
            "which department",
            "hangi bolum",
            "hangi bölüm",
            "bolumunuz",
            "bölümünüz",
        ]):
            options = _profiling_followup_options("dept", profile, prefer_tr=prefer_tr)
        elif any(
            phrase in lowered
            for phrase in [
                "which year did you enroll",
                "what year did you enroll",
                "entry year",
                "starting year",
                "curriculum version",
                "hangi yil",
                "hangi yıl",
                "hangi yilda kayit",
                "hangi yılda kayıt",
                "universiteye hangi yil",
                "universiteye hangi yıl",
            ]
        ):
            options = _profiling_followup_options("start_year", profile, prefer_tr=prefer_tr)
        elif any(
            phrase in lowered
            for phrase in [
                "which semester are you in",
                "what semester are you in",
                "which semester",
                "what semester",
                "fall or spring",
                "hangi donem",
                "hangi dönem",
                "guz veya bahar",
                "güz veya bahar",
            ]
        ):
            options = _profiling_followup_options("semester", profile, prefer_tr=prefer_tr)
        elif any(
            phrase in lowered
            for phrase in [
                "which year are you in",
                "what year are you in",
                "which year and semester",
                "what year and semester",
                "hangi sinif",
                "hangi sınıf",
                "kacinci sinif",
                "kaçıncı sınıf",
            ]
        ):
            options = _profiling_followup_options("year_term", profile, prefer_tr=prefer_tr)

    return {
        "answer": clean_answer,
        "is_followup": is_followup,
        "options": options or None,
        "followup_mode": "append" if is_followup and is_correction else ("replace" if is_followup else None),
    }


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


def _is_elective_followup_request(question):
    q = _normalize_chatty_text(question)
    if not q:
        return False

    if _is_short_ack(question):
        return True

    return any(phrase in q for phrase in [
        "tell me about electives",
        "show electives",
        "electives",
        "elective",
        "secmeli",
        "seçmeli",
        "secmeli ders",
        "seçmeli ders",
        "evet",
    ])


def _assistant_mentions_elective_context(text):
    content_lower = (text or "").lower()
    return any(phrase in content_lower for phrase in [
        "elective slot",
        "elective slots",
        "courses you can choose",
        "elective",
        "seçmeli",
        "secmeli",
    ])


def _is_pure_credit_load_question(question):
    q = _normalize_chatty_text(question)
    if not q:
        return False

    has_credit_phrase = any(phrase in q for phrase in [
        "how many credits can i take",
        "how many credits",
        "credits can i take",
        "credit limit",
        "kaç kredi",
        "kac kredi",
        "credits",
        "kredi",
        "ects",
    ])
    if not has_credit_phrase:
        return False

    # If the user explicitly mentions courses or schedule, treat as planning.
    if any(phrase in q for phrase in ["courses", "course", "schedule", "which course", "which courses"]):
        return False

    return True


def _maybe_elective_followup(question, history, session_language=None):
    if not _is_elective_followup_request(question):
        return None

    last_assistant = None
    for msg in reversed(history[-12:]):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    if not last_assistant:
        return None

    if not _assistant_mentions_elective_context(last_assistant):
        return None

    latest_plan = get_latest_planning_tuple_from_history(history)
    profile = extract_profile_from_history(history)
    dept = None
    if latest_plan:
        dept, _, _ = latest_plan
    if not dept:
        dept = profile.get("department")
    if not dept:
        return None

    elective_lines = _format_elective_option_lines(
        dept,
        start_year=profile.get("start_year"),
        program_name=profile.get("program_name"),
        curriculum_url=profile.get("curriculum_url"),
        prefer_tr=_looks_turkish(question),
    )
    if not elective_lines:
        return get_fallback_response(session_language or detect_language(question))

    return "\n".join(elective_lines)


def _format_elective_option_lines(dept, start_year=None, program_name=None, curriculum_url=None, limit=20, prefer_tr=False):
    electives = get_electives(
        dept,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
    if not electives:
        return []

    lines = [_t(f"Available electives for {dept}:", f"{dept} icin mevcut secmeli dersler:", prefer_tr)]
    for item in electives[:limit]:
        code = (item.get("code") or "").strip()
        name = (item.get("name") or "").strip()
        if code and name:
            lines.append(f"- {code}: {name}")
        elif code:
            lines.append(f"- {code}")
        elif name:
            lines.append(f"- {name}")

    if len(electives) > limit:
        lines.append(_t(f"- ...and {len(electives) - limit} more elective option(s).", f"- ...ve {len(electives) - limit} adet daha secmeli ders var.", prefer_tr))

    return lines


def _is_medicine_department(dept=None, program_name=None, faculty=None):
    text = " ".join(part for part in [dept, program_name, faculty] if part).casefold()
    return any(token in text for token in ["medicine", "tıp", "tip", "medical faculty", "faculty of medicine"])


def _format_graduation_requirements(prefer_tr=False, is_medicine=False):
    ects = 360 if is_medicine else 240
    en_lines = [
        "In order to graduate from this undergraduate program, the students are required:",
        "- to succeed in all of the courses listed in the curriculum of the program by getting the grade of at least DD/S",
        f"- to complete a minimum of {ects} ECTS",
        "- to have a Cumulative Grade Point Average (CGPA) of 2.00 out of 4.00",
        "- to complete their compulsory internship in a specified duration and quality.",
    ]
    tr_lines = [
        "Bu lisans programından mezun olabilmek icin ogrencilerin:",
        "- programin ders planinda yer alan tum dersleri en az DD/S notu ile basarmalari",
        f"- en az {ects} ECTS tamamlamalari",
        "- 4.00 uzerinden en az 2.00 Genel Not Ortalamasi (CGPA) sahibi olmalari",
        "- zorunlu stajlarini belirtilen sure ve nitelikte tamamlamalari",
    ]
    return "\n".join(tr_lines if prefer_tr else en_lines)


def _format_profile_context_ack(profile, prefer_tr=False):
    if not profile:
        return None

    parts = []
    faculty = profile.get("faculty")
    dept = profile.get("department")
    program_name = profile.get("program_name")
    year = profile.get("year")
    term = profile.get("term")
    start_year = profile.get("start_year")
    failed_courses = _dedupe_keep_order(profile.get("failed_courses") or [])

    if faculty:
        parts.append(faculty)
    if dept:
        parts.append(dept)
    if program_name and program_name != dept:
        parts.append(program_name)
    if year and term:
        parts.append(f"Year {year} {term}")
    elif year:
        parts.append(f"Year {year}")
    elif term:
        parts.append(term)
    if start_year:
        parts.append(f"enrolled {start_year}")
    if failed_courses:
        parts.append("failed " + ", ".join(failed_courses))

    if not parts:
        return None

    return _t(
        "Got it - I've noted your profile: "
        + ", ".join(parts)
        + ". I'll keep that in mind for future course planning.",
        "Tamam - profilinizi not ettim: "
        + ", ".join(parts)
        + ". Bunu sonraki ders planlamasında aklımda tutacağım.",
        prefer_tr,
    )


def _should_resume_planning_after_profile_completion(question, history):
    """
    If the latest substantive user question was a planning / curriculum request,
    do not stop at a profile acknowledgment once the profile is complete.
    """
    anchor = get_latest_substantive_user_question(history or [])
    candidate = anchor or question or ""
    if not candidate:
        return False

    candidate_lower = candidate.lower()
    if _looks_like_curriculum_lookup(candidate):
        return True
    if is_semester_question(candidate):
        return True
    if is_course_lookup_question(candidate):
        return True
    if is_graduation_question(candidate):
        return True
    if is_regulations_question(candidate):
        return True

    return any(k in candidate_lower for k in [
        "which course", "which courses", "what course", "what courses",
        "what classes", "which classes", "should i take", "i should take",
        "hangi ders", "hangi dersleri",
    ])


def _looks_like_profile_update_only(question, history, last_bot_q, profile=None, allow_pending=False):
    if not allow_pending and last_bot_q in {"dept", "year_term", "start_year", "failed_courses"}:
        return False

    q = (question or "").strip().lower()
    if not q:
        return False

    if _looks_like_contextual_followup(question):
        return False
    if _looks_like_curriculum_followup(question) or _looks_like_invalid_curriculum_term(question):
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
        for field in ["faculty", "department", "year", "term", "start_year", "failed_courses"]
    )


def _looks_turkish(text):
    t = (text or "").casefold()
    return any(w in t for w in [
        "kredi", "ders", "not", "ortalama", "gano", "bahar", "guz", "mezun",
        "kural", "yonetmelik", "merhaba", "selam", "slm", "naber", "nasilsin",
        "nasılsın", "tamam", "tmm", "boş", "bos", "neyse", "iyiyim", "hocam",
        "abi", "abla", "kanka", "hangi", "universite", "üniversite", "fakulte",
        "fakülte", "bolum", "bölüm", "sinif", "sınıf", "donem", "dönem",
    ])


def _conversation_prefers_turkish(question, history=None, assistant_text=None):
    if _looks_turkish(question):
        return True
    if assistant_text and _looks_turkish(assistant_text):
        return True

    for msg in reversed(history or []):
        if msg.get("role") != "assistant":
            continue
        if _looks_turkish(msg.get("content", "")):
            return True
        break

    latest_substantive = get_latest_substantive_user_question(history or [])
    return _looks_turkish(latest_substantive)


def _t(en_text, tr_text, prefer_tr):
    return tr_text if prefer_tr else en_text


def _soft_fallback_message(session_language="tr"):
    return get_fallback_response(session_language)


def _normalize_chatty_text(text):
    t = (text or "").strip().lower()
    if not t:
        return ""
    # Collapse stretched-out chat text like "hiiii" or "helloooo".
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _word_matches_chatty_variant(word, candidates):
    if word in candidates:
        return True
    compact = re.sub(r"(.)\1+", r"\1", word)
    return compact in candidates


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


def _get_curriculum_courses_robust(
    dept,
    year,
    term,
    start_year=None,
    program_name=None,
    curriculum_url=None,
    faculty=None,
):
    """
    Robust curriculum lookup that retries with normalized/fallback selectors.
    This avoids false "no curriculum data" responses when one selector is noisy.
    """
    attempts = []
    seen = set()

    def add_attempt(d, sy, pn, cu):
        d_norm = normalize_department_code(d) if d else None
        if not d_norm:
            return
        key = (
            d_norm,
            int(sy) if sy is not None else None,
            (pn or "").strip() or None,
            (cu or "").strip() or None,
        )
        if key in seen:
            return
        seen.add(key)
        attempts.append(key)

    add_attempt(dept, start_year, program_name, curriculum_url)
    add_attempt(dept, start_year, None, None)

    if start_year is not None:
        add_attempt(dept, None, program_name, curriculum_url)
        add_attempt(dept, None, None, None)

    if program_name:
        resolved = resolve_ieu_department(program_name, faculty_name=faculty)
        if not resolved:
            resolved = resolve_ieu_department(program_name)
        if resolved:
            add_attempt(
                resolved.get("department_code") or dept,
                start_year,
                resolved.get("program_name") or program_name,
                resolved.get("curriculum_url") or curriculum_url,
            )
            add_attempt(resolved.get("department_code") or dept, start_year, None, None)
            if start_year is not None:
                add_attempt(
                    resolved.get("department_code") or dept,
                    None,
                    resolved.get("program_name") or program_name,
                    resolved.get("curriculum_url") or curriculum_url,
                )
                add_attempt(resolved.get("department_code") or dept, None, None, None)

    for d_norm, sy, pn, cu in attempts:
        courses = get_curriculum_courses(
            d_norm,
            year,
            term,
            start_year=sy,
            program_name=pn,
            curriculum_url=cu,
        )
        if courses:
            return courses

    return []


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
        "kredi", "ne kadar kredi", "kac kredi", "kaأ§ kredi",
    ])


def _get_regulations_from_db(collection_obj):
    global _REGULATIONS_CACHE
    if _REGULATIONS_CACHE is not None:
        return _REGULATIONS_CACHE

    try:
        results = collection_obj.get(
            where={"doc_type": "regulations"},
            include=["documents"],
        )
        docs = results.get("documents") or []
        if docs:
            doc = docs[0]
            if isinstance(doc, list):
                doc = doc[0] if doc else None
            if doc:
                _REGULATIONS_CACHE = json.loads(doc)
                return _REGULATIONS_CACHE
    except Exception as e:
        print(f"  regulations lookup error: {e}")

    return None


def _query_web_info_context(collection_obj, question, max_docs=3):
    try:
        results = collection_obj.query(
            query_texts=[question],
            n_results=max_docs,
            where={"doc_type": "web_info"},
            include=["documents"],
        )
        docs = (results.get("documents") or [[]])[0]
        return [doc for doc in docs if doc]
    except Exception as e:
        print(f"  web info query error: {e}")
        return []


def _parse_web_info_doc(doc):
    meta = {
        "title": None,
        "category": None,
        "description": None,
        "url": None,
    }
    if not doc:
        return meta, []

    for field in ["Title", "Category", "Description", "URL"]:
        m = re.search(rf"^{re.escape(field)}:\s*(.+)$", doc, re.IGNORECASE | re.MULTILINE)
        if m:
            meta[field.lower()] = m.group(1).strip()

    body = ""
    if "\n\n" in doc:
        body = doc.split("\n\n", 1)[1]
    else:
        lines = doc.splitlines()
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("url:"):
                body_start = i + 1
        body = "\n".join(lines[body_start:])

    body_lines = [line.strip() for line in body.splitlines() if line.strip()]
    return meta, body_lines


def _extract_web_info_snippet(body_lines, question_terms, max_lines=6):
    if not body_lines:
        return []

    scored = []
    for idx, line in enumerate(body_lines):
        line_lower = line.lower()
        score = 0
        for term in question_terms:
            if term in line_lower:
                score += 2 if len(term) > 4 else 1
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return body_lines[:max_lines]

    chosen = []
    seen = set()
    for _, idx in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]:
        for j in range(max(0, idx - 1), min(len(body_lines), idx + 2)):
            if j in seen:
                continue
            seen.add(j)
            chosen.append(body_lines[j])
            if len(chosen) >= max_lines:
                return chosen

    return chosen[:max_lines]


def _score_web_info_doc(question_terms, meta, body_lines):
    if not question_terms:
        return 0

    score = 0
    title_desc = " ".join(
        part for part in [
            meta.get("title") or "",
            meta.get("category") or "",
            meta.get("description") or "",
        ]
        if part
    ).lower()
    body_sample = " ".join(body_lines[:20]).lower()

    for term in question_terms:
        if term in title_desc:
            score += 3
        elif term in body_sample:
            score += 1
    return score


def _format_web_info_answer(question, docs, max_docs=2):
    if not docs:
        return None

    prefer_tr = _conversation_prefers_turkish(question)

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for",
        "from", "how", "i", "is", "it", "me", "my", "of", "on", "or", "please",
        "tell", "the", "this", "to", "what", "when", "where", "who", "with",
        "you", "your",
    }
    question_terms = []
    for raw in re.findall(r"[a-z0-9]+", (question or "").lower()):
        if len(raw) < 3 or raw in stopwords:
            continue
        if raw not in question_terms:
            question_terms.append(raw)

    scored_docs = []
    for order, doc in enumerate(docs[:max_docs]):
        meta, body_lines = _parse_web_info_doc(doc)
        score = _score_web_info_doc(question_terms, meta, body_lines)
        scored_docs.append((score, order, meta, body_lines))

    positive_docs = [item for item in scored_docs if item[0] > 0]
    if positive_docs:
        scored_docs = positive_docs
    else:
        scored_docs = scored_docs[:1]

    parts = []
    for _, _, meta, body_lines in sorted(scored_docs, key=lambda item: (-item[0], item[1])):
        lines = []
        if meta["title"]:
            lines.append(_t(f"Title: {meta['title']}", f"Baslik: {meta['title']}", prefer_tr))
        if meta["category"]:
            lines.append(_t(f"Category: {meta['category']}", f"Kategori: {meta['category']}", prefer_tr))
        if meta["description"]:
            lines.append(_t(f"Description: {meta['description']}", f"Aciklama: {meta['description']}", prefer_tr))
        if meta["url"]:
            lines.append(_t(f"URL: {meta['url']}", f"URL: {meta['url']}", prefer_tr))

        snippet = _extract_web_info_snippet(body_lines, question_terms)
        if snippet:
            if lines:
                lines.append("")
            lines.extend(snippet)

        if lines:
            parts.append("\n".join(lines).strip())

    answer = "\n\n---\n\n".join(parts).strip()
    if len(answer) > 1800:
        answer = answer[:1800].rstrip()
        if not answer.endswith("..."):
            answer += "..."
    return answer or None


def _format_course_correction_note(corrections, prefer_tr):
    if not corrections:
        return None
    lines = []
    for original, corrected in corrections.items():
        lines.append(_t(
            f"Note: I could not find {original} in the IEU database. Showing the closest match {corrected}.",
            f"Not: IEU veritabaninda {original} bulunamadi. En yakin eslesme {corrected} gosteriliyor.",
            prefer_tr,
        ))
    return "\n".join(lines)


def _maybe_entry_year_note(dept, start_year, prefer_tr, program_name=None, curriculum_url=None):
    if not dept or start_year is None:
        return None
    years = get_available_entry_years(dept, program_name=program_name)
    if not years or start_year in years:
        return None
    closest = min(years, key=lambda y: abs(y - start_year))
    return _t(
        f"Note: I could not find curriculum data for entry year {start_year}. Using the closest available entry year {closest}.",
        f"Not: {start_year} giris yili icin mufredat bulunamadi. En yakin yil {closest} kullaniliyor.",
        prefer_tr,
    )


def _extract_max_ects_from_rule(rule_text):
    if not rule_text:
        return None
    m = re.search(r"up to\s*(\d+)\s*ects", rule_text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"maximum\s*(\d+)\s*ects", rule_text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _format_regulations_answer(question, history, gpa_value):
    regs = _get_regulations_from_db(collection)
    if not regs:
        return None

    prefer_tr = _conversation_prefers_turkish(question, history)
    t = (question or "").lower()

    # Credit load / ECTS limits
    if _is_credit_load_question(question):
        if gpa_value is None:
            return _t(
                "FOLLOWUP_NEEDED: What is your current GPA/CGPA (0.00-4.00)?",
                "FOLLOWUP_NEEDED: Güncel GANO/CGPA'n kaç? (0.00-4.00)",
                prefer_tr,
            )

        rules = (regs.get("credit_load") or {}).get("rules_by_gpa") or {}
        if gpa_value >= 2.5:
            rule = rules.get("2.50_and_above") or ""
        elif gpa_value >= 2.0:
            rule = rules.get("2.00_to_2.49") or ""
        else:
            rule = rules.get("below_threshold") or ""

        max_ects = _extract_max_ects_from_rule(rule)
        if max_ects is None:
            max_ects = get_credit_limit_by_gpa(gpa_value)

        lines = [
            _t(
                f"With GPA {gpa_value:.2f}, your maximum is {max_ects} ECTS.",
                f"GPA {gpa_value:.2f} ile maksimum {max_ects} ECTS alabilirsiniz.",
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


def _profiling_year_term_followup(question, prefer_tr=False):
    y, t = extract_year_and_term(question)
    y = y or extract_year_only(question)
    t = t or extract_term_only(question)

    if y and not t:
        return _curriculum_semester_followup(prefer_tr=prefer_tr)
    if t and not y:
        return _curriculum_year_followup(prefer_tr=prefer_tr)
    return _curriculum_year_followup(prefer_tr=prefer_tr)


def is_greeting_or_small_talk(text):
    """
    Check if the message is just a greeting or small talk, not a substantive question.
    """
    raw = (text or "").strip()
    if not raw:
        return False

    t = _normalize_chatty_text(raw)
    if not t:
        return False

    words = re.findall(r"[a-z0-9']+", t)
    if not words:
        return False

    raw_lower = raw.casefold()
    has_repeated_letters = bool(re.search(r"([a-z])\1{2,}", raw_lower))

    greeting_words = {
        "hi", "hello", "hey", "greetings", "merhaba", "selam", "sup", "yo",
        "nasilsin", "nas\u0131ls\u0131n", "hiya", "hai", "hewwo", "helo",
        "slm", "naber", "tamam", "tmm",
    }
    greeting_phrases = {
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "what's up",
        "whats up",
        "what s up",
        "hi there",
        "hello there",
        "hey there",
        "ne haber",
        "bos ver",
        "boş ver",
        "neyse",
        "iyiyim",
    }
    filler_words = {
        "lol", "lmao", "haha", "ok", "okay", "k", "kk", "bruh", "bro",
        "ugh", "hmm", "umm", "idk", "tmm",
    }
    frustrated_phrases = {
        "you're useless",
        "you are useless",
        "this is stupid",
        "whatever",
        "nevermind",
        "nvm",
        "forget it",
        "sacma",
        "saçma",
        "ise yaramaz",
        "işe yaramaz",
        "salak",
    }
    keyboard_mash_patterns = [r"asdf", r"asdfjkl", r"qwerty", r"zxcvbn"]
    phrase = " ".join(words)
    if phrase in greeting_phrases:
        return True

    if len(words) <= 3 and has_repeated_letters and (
        any(_word_matches_chatty_variant(word, greeting_words) for word in words)
        or any(_word_matches_chatty_variant(word, filler_words) for word in words)
    ):
        return True

    if len(words) <= 3 and any(_word_matches_chatty_variant(word, filler_words) for word in words):
        return True

    if len(words) <= 4 and any(re.fullmatch(pattern, phrase) for pattern in keyboard_mash_patterns):
        return True

    if any(cue in raw_lower for cue in frustrated_phrases):
        return True

    # Only treat very short messages as greetings when they are made up of
    # obvious greeting words. This keeps "which internship" and similar phrases
    # from being misread as small talk.
    if len(words) <= 3 and any(_word_matches_chatty_variant(word, greeting_words) for word in words):
        return True
    return False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN ASK FUNCTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _looks_like_curriculum_lookup(text):
    """
    Detect questions that are asking for a study plan / curriculum.
    These are the only questions that should force the profile-gathering flow.
    """
    t = (text or "").lower()
    if is_semester_question(text):
        return True

    curriculum_cues = [
        "curriculum",
        "curricula",
        "mufredat",
        "study plan",
        "course plan",
        "program plan",
        "programme plan",
        "degree plan",
    ]
    return any(cue in t for cue in curriculum_cues)


def _looks_like_curriculum_followup(text):
    """
    Detect short semester follow-ups like "what about 3rd fall".
    These should be handled deterministically, not routed to the LLM fallback.
    """
    q = _normalize_chatty_text(text)
    if not q:
        return False
    if is_course_lookup_question(text) or is_graduation_question(text) or is_regulations_question(text):
        return False

    year = extract_year_only(text)
    if not year:
        year, _ = extract_year_and_term(text)
    term = extract_term_only(text)
    if not term:
        _, term = extract_year_and_term(text)
    if not year or not term:
        return False

    followup_prefixes = (
        "what about",
        "how about",
        "and",
        "also",
        "for",
        "in",
        "what if",
        "then",
        "so",
    )
    correction_cues = (
        "i meant",
        "meant",
        "just kidding",
        "joking",
        "jk",
        "j/k",
        "sorry",
        "instead",
        "no wait",
        "wait no",
        "correction",
        "i mean",
        "meant to say",
    )

    if re.match(r"^(what about|how about|what if|and|also|for|in|then|so)\b", q):
        return True
    if any(cue in q for cue in correction_cues):
        return True

    if len(q.split()) <= 4:
        return True

    return False


def _looks_like_invalid_curriculum_term(text):
    q = _normalize_chatty_text(text)
    if not q:
        return False

    invalid_terms = [
        "summer",
        "winter",
        "break",
        "vacation",
        "intersession",
        "inter-session",
    ]
    return any(re.search(rf"\b{re.escape(term)}\b", q) for term in invalid_terms)


def reset_flow_state(history, question):
    """
    Heuristic to decide whether an active profiling/planning flow should be
    interrupted and reset because the user asked an unrelated question or the
    flow has stalled.

    Returns True when the flow should be reset (caller should clear any
    pending flow state like `last_bot_q`).
    """
    try:
        last_bot_q = get_last_bot_question_type(history)
    except Exception:
        last_bot_q = None

    if not last_bot_q:
        return False

    q = (question or "").strip()
    q_lower = q.lower()

    # Determine the new message's primary intent (if any)
    new_intent = None
    if is_semester_question(q):
        new_intent = "planning"
    elif is_regulations_question(q):
        new_intent = "regulations"
    elif is_gpa_credit_limit_question(q, history):
        new_intent = "credit_limit"
    elif is_web_info_question(q):
        new_intent = "web_info"
    elif is_graduation_question(q):
        new_intent = "graduation"
    elif is_course_lookup_question(q):
        new_intent = "course_lookup"

    # If the user is mid-profiling (faculty/dept/year/start_year/failed_courses)
    # and the new message clearly maps to a non-academic or different intent,
    # treat it as an interrupt.
    profiling_tokens = {"faculty", "dept", "year_term", "start_year", "failed_courses"}
    if last_bot_q in profiling_tokens:
        # If the new intent is unrelated to profiling/planning, reset.
        if new_intent is None or new_intent == "web_info" or new_intent == "course_lookup":
            return True

        # If the new intent is academic but not planning (e.g., regulations/credit),
        # we still allow it to proceed without clearing profile state. Only reset
        # when it's a clearly unrelated topic.

    # Flow timeout: if there have been many recent messages (stall) assume stale.
    recent = (history or [])[-12:]
    if last_bot_q in profiling_tokens and len(recent) >= 6:
        return True

    return False


TOPIC_SIGNALS = {
    "internship": ["internship", "staj"],
    "password": ["password", "şifre", "sifre", "forgot", "reset"],
    "certificate": ["certificate", "belge", "öğrenci belgesi"],
    "attendance": ["attendance", "devam", "missed class"],
    "graduation": ["graduation", "mezuniyet"],
    "gpa": ["gpa", "gano", "credit limit", "kredi limiti"],
    "transcript": ["transcript", "transkript"],
}


def is_topic_change(question: str, history: list) -> bool:
    if len(history) < 4:
        return False
    q_lower = question.lower()
    last_two_assistant = [m["content"].lower() for m in history[-4:] if m["role"] == "assistant"]
    combined_history = " ".join(last_two_assistant)
    for topic, keywords in TOPIC_SIGNALS.items():
        q_match = any(kw in q_lower for kw in keywords)
        history_match = any(kw in combined_history for kw in keywords)
        if q_match and not history_match:
            print(f"[INTENT] Topic change detected: {topic}")
            return True
    return False


SELF_IDENTITY_QUESTIONS = [
    "what are you", "who are you", "sen kimsin", "ne sin", "nedir ekobot",
    "what is ekobot", "tell me about yourself", "kendini tanıt",
]

USER_IDENTITY_QUESTIONS = [
    "who am i", "ben kimim", "do you know me", "beni tanıyor musun",
    "what do you know about me",
]


def is_self_identity_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(phrase in q_lower for phrase in SELF_IDENTITY_QUESTIONS)


def is_user_identity_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(phrase in q_lower for phrase in USER_IDENTITY_QUESTIONS)


def _format_user_identity_response(profile, prefer_tr=False):
    if not profile:
        profile = {}

    parts = []
    gpa_value = profile.get("gpa")
    department = profile.get("program_name") or profile.get("department")
    year = profile.get("year")

    if gpa_value is not None:
        parts.append(_t(f"a GPA of {gpa_value}", f"bir GANO {gpa_value}", prefer_tr))
    if department:
        parts.append(_t(f"you're in {department}", f"{department} bölümündesin", prefer_tr))
    if year:
        parts.append(_t(f"you're in year {year}", f"{year}. sınıftasın", prefer_tr))

    if not parts:
        return _t(
            "I don't have your personal information — I can only remember what you tell me during our conversation!",
            "Kişisel bilgilerin bende yok — sadece bu konuşma sırasında bana söylediklerini hatırlayabilirim!",
            prefer_tr,
        )

    if len(parts) == 1:
        details = parts[0]
    elif len(parts) == 2:
        details = f"{parts[0]} and {parts[1]}" if not prefer_tr else f"{parts[0]} ve {parts[1]}"
    else:
        if prefer_tr:
            details = ", ".join(parts[:-1]) + f" ve {parts[-1]}"
        else:
            details = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return _t(
        f"Based on our conversation, you mentioned {details}.",
        f"Konuşmamıza göre {details}.",
        prefer_tr,
    )


CONVERSATIONAL_PHRASES = [
    "thanks", "thank you", "teşekkür", "tamam", "okay", "ok", "sure",
    "got it", "cool", "great", "nice", "bye", "goodbye", "görüşürüz",
    "lol", "haha", "wtf", "hmm", "oh", "wow", "really", "interesting",
    "who am i", "what are you", "who are you", "sen kimsin", "ben kimim",
]

IEU_ACRONYMS = {
    "ybs": "Yönetim Bilişim Sistemleri (Management Information Systems) — a department at IEU Faculty of Business.",
    "obs": "Öğrenci Bilgi Sistemi — IEU's Student Information System, accessible at obs.ieu.edu.tr",
    "sys": "Staj Yönetim Sistemi — IEU's Internship Management System at staj.izmirekonomi.edu.tr",
    "tlc": "Teaching and Learning Center — IEU's academic support center. See: https://www.ieu.edu.tr/tlc/en/for-students",
    "kyk": "Kredi ve Yurtlar Kurumu — Government student dormitory and loan institution",
    "ects": "European Credit Transfer System — the credit system used at IEU",
    "cgpa": "Cumulative Grade Point Average — your overall GPA across all semesters",
    "gano": "Genel Ağırlıklı Not Ortalaması — Turkish term for CGPA",
    "bano": "Bütünleme Ağırlıklı Not Ortalaması — weighted GPA for resit exams",
}

ACADEMIC_KEYWORDS = [
    "course", "ders", "gpa", "gano", "ects", "kredi", "internship", "staj",
    "graduation", "mezuniyet", "curriculum", "müfredat", "syllabus",
    "department", "bölüm", "faculty", "fakülte", "registration", "kayıt",
    "exam", "sınav", "grade", "not", "certificate", "belge", "attendance",
    "devam", "prerequisite", "ön koşul", "elective", "seçmeli", "advisor",
    "danışman", "transcript", "transkript", "password", "şifre", "apply",
    "başvuru", "scholarship", "burs", "erasmus", "exchange", "double major",
    "çift anadal", "minor", "yan dal",
]


def classify_message_type(question: str, history: list) -> str:
    q_lower = question.lower().strip()

    # Check academic keywords first
    if any(kw in q_lower for kw in ACADEMIC_KEYWORDS):
        return "academic"

    # Check conversational phrases
    if any(phrase in q_lower for phrase in CONVERSATIONAL_PHRASES):
        return "conversational"

    # Short message with no keywords
    if len(q_lower) < 15:
        if not history:
            return "conversational"
        # Check if last bot message asked a question (flow continuation)
        last_bot_msgs = [m["content"] for m in history if m["role"] == "assistant"]
        if last_bot_msgs:
            last = last_bot_msgs[-1].lower()
            if "?" in last or "please" in last or "lütfen" in last or "seç" in last:
                return "ambiguous"
        return "conversational"

    return "academic"


def _extract_known_acronym_definition(question: str):
    q_lower = (question or "").lower().strip()
    definition_cues = ["what is", "what does", "what's", "nedir", "ne demek"]

    if not any(cue in q_lower for cue in definition_cues):
        return None, None

    for acronym, definition in IEU_ACRONYMS.items():
        if re.search(rf"\b{re.escape(acronym)}\b", q_lower):
            return acronym, definition
    return None, None


def ask(question, history=[], client_type=None, session_language=None):
    # Some eval harnesses redirect stdout to non-UTF8 files on Windows.
    # Keep debug logging ASCII-only to avoid encode errors.
    safe_question = (question or "").encode("ascii", "backslashreplace").decode("ascii")
    print(f"\n[EkoBot] Question: {safe_question}")

    session_language = session_language or detect_language(question)
    prefer_tr = session_language == "tr"
    history_profile = extract_profile_from_history(history or [])

    if is_self_identity_question(question):
        return _t(
            "I'm EkoBot, an AI-powered academic assistant for İzmir University of Economics (IEU). I can help you with courses, curriculum planning, graduation requirements, GPA rules, internships, and university regulations. What would you like to know?",
            "Ben EkoBot, İzmir Ekonomi Üniversitesi (İEU) öğrencileri için geliştirilmiş bir yapay zeka akademik asistanım. Dersler, müfredat, mezuniyet koşulları, GPA kuralları, staj ve üniversite yönetmelikleri konularında yardımcı olabilirim. Ne öğrenmek istersin?",
            prefer_tr,
        )

    if is_user_identity_question(question):
        return _format_user_identity_response(history_profile, prefer_tr=prefer_tr)

    msg_type = classify_message_type(question, history or [])
    print(f"[ROUTE] Message type: {msg_type}")

    acronym, acronym_definition = _extract_known_acronym_definition(question)
    if acronym_definition:
        print(f"[ROUTE] Acronym match: {acronym}")
        return acronym_definition

    if msg_type == "conversational":
        if client is None:
            return get_fallback_response(session_language)
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are EkoBot, a friendly assistant for IEU. Respond naturally and briefly in "
                            f"{session_language}. Max 2 sentences. Don't dump academic info unless asked."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=1000,
            )
            answer = response.choices[0].message.content
            if not answer or answer.strip() == "" or answer.strip() == "''":
                return get_fallback_response(session_language)
            return answer
        except Exception:
            return get_fallback_response(session_language)

    if msg_type == "ambiguous":
        pass

    question_lower = question.lower().strip()
    last_bot_q = get_last_bot_question_type(history)

    if is_topic_change(question, history or []):
        last_bot_q = None

    # Interrupt detection: if the user asks a clearly unrelated question while
    # a profiling/planning flow is active, reset the pending flow state so the
    # new request is handled independently.
    try:
        if reset_flow_state(history, question):
            last_bot_q = None
    except Exception:
        pass

    if is_greeting_or_small_talk(question):
        if last_bot_q in {"faculty", "dept", "year_term", "start_year", "failed_courses"}:
            pending_map = {
                "faculty": "faculty",
                "dept": "dept",
                "year_term": "year_term",
                "start_year": "start_year",
                "failed_courses": "failed_courses",
            }
            followup = _profiling_followup_message(pending_map.get(last_bot_q), history_profile, prefer_tr=prefer_tr)
            if followup:
                return followup
        return _t(
            "Hi! How can I help you with IEU information today?",
            "Merhaba! IEU bilgileriyle ilgili size nasil yardimci olabilirim?",
            prefer_tr,
        )

    if is_out_of_scope_question(question) and not is_web_info_question(question):
        return get_fallback_response(session_language)

    if _is_short_ack(question) and last_bot_q in {"faculty", "dept", "year_term", "start_year", "failed_courses"}:
        if last_bot_q == "failed_courses":
            return _t(
                "FOLLOWUP_NEEDED: Which course codes did you fail? (e.g. CE 201, MATH 153)",
                "FOLLOWUP_NEEDED: Kaldığın derslerin kodları neler? (örn. CE 201, MATH 153)",
                prefer_tr,
            )
        followup = _profiling_followup_message({
            "faculty": "faculty",
            "dept": "dept",
            "year_term": "year_term",
            "start_year": "start_year",
        }.get(last_bot_q), None, prefer_tr=prefer_tr)
        if followup:
            return followup

    elective_followup = _maybe_elective_followup(question, history, session_language=session_language)
    if elective_followup:
        return elective_followup

    regs_followup = _maybe_regulations_followup(question, history)
    if regs_followup:
        return regs_followup

    if _is_pure_credit_load_question(question):
        gpa_value = extract_gpa_from_context(question, history)
        regs_answer = _format_regulations_answer(question, history, gpa_value)
        if regs_answer:
            return regs_answer

    latest_substantive_question = get_latest_substantive_user_question(history or [])
    if _looks_like_invalid_curriculum_term(question):
        if (
            (history_profile.get("department") and history_profile.get("year") and history_profile.get("term"))
            or _looks_like_curriculum_lookup(latest_substantive_question or "")
        ):
            return _t(
                "I only understand Fall or Spring for semester questions. If you meant a semester, try something like '2nd fall' or '2nd spring'.",
                "Ders donemi sorularinda sadece Fall veya Spring anlarim. Bir donemi kastediyorsaniz '2nd fall' veya '2nd spring' gibi yazin.",
                prefer_tr,
            )

    history_with_current = list(history or []) + [{"role": "user", "content": question}]
    current_profile = extract_profile_from_history(history_with_current)
    if _profile_fields_changed(history_profile, current_profile) and _looks_like_profile_update_only(
        question,
        history,
        last_bot_q,
        profile=current_profile,
        allow_pending=True,
    ):
        next_question, question_type = get_next_profiling_question(
            current_profile,
            question_lower,
            require_curriculum_details=_requires_curriculum_details(question, history),
            prefer_tr=prefer_tr,
        )
        if next_question:
            return _profiling_followup_message(
                question_type,
                current_profile,
                prefer_tr=prefer_tr,
                correction=_looks_like_profile_correction(question),
            ) or next_question
        profile_ack = _format_profile_context_ack(current_profile, prefer_tr=prefer_tr)
        if profile_ack and not _should_resume_planning_after_profile_completion(question, history):
            return profile_ack

    regulations_question = is_regulations_question(question)

    profile_flow_trigger = _looks_like_curriculum_lookup(question) and not regulations_question

    pythonDIRECT_RAG_KEYWORDS = [
        "internship", "staj", "staj başvuru", "apply for internship",
        "password", "şifre", "sifre", "forgot password",
        "student certificate", "degree certificate", "diploma", "mezuniyet belgesi", "diplomam", "öğrenci belgesi",
        "transcript", "transkript",
        "apply for graduation", "mezuniyet başvuru",
    ]
    if any(kw in question_lower for kw in pythonDIRECT_RAG_KEYWORDS):
        # skip curriculum flow, go straight to RAG
        profile_flow_trigger = False

    if not regulations_question and not profile_flow_trigger and last_bot_q in {"faculty", "dept", "year_term", "start_year", "failed_courses"}:
        if last_bot_q == "faculty":
            profile_flow_trigger = bool(detect_faculty_choice(question) or detect_department(question))
        elif last_bot_q == "dept":
            profile_flow_trigger = bool(detect_department(question))
        elif last_bot_q == "year_term":
            profile_flow_trigger = bool(
                extract_year_only(question) or extract_term_only(question)
                or any(extract_year_and_term(question))
            )
        elif last_bot_q == "start_year":
            if extract_starting_year_loose(question):
                profile_flow_trigger = True
            elif (
                extract_year_only(question)
                or extract_term_only(question)
                or any(extract_year_and_term(question))
            ):
                return _curriculum_entry_year_followup(clarify_invalid_reply=True, prefer_tr=prefer_tr)
        elif last_bot_q == "failed_courses":
            profile_flow_trigger = bool(extract_failed_courses_from_context(question, history, None))

    # PROFILING FLOW
    # Only ask for profile details when the current message is curriculum-related
    # or it is clearly answering a pending curriculum-profile follow-up.
    if profile_flow_trigger:
        # Include the current user message when building the profile. Clients vary in
        # whether they include the latest message in `history`, so do it here to keep
        # behavior consistent.
        history_with_current = list(history or []) + [{"role": "user", "content": question}]
        profile = extract_profile_from_history(history_with_current)
        next_question, question_type = get_next_profiling_question(
            profile,
            question_lower,
            require_curriculum_details=_requires_curriculum_details(question, history),
            prefer_tr=prefer_tr,
        )

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
            if question_type == "faculty":
                is_answering_profiling = bool(detect_faculty_choice(question) or detect_department(question))
            elif question_type == "dept":
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
                    return _curriculum_semester_followup(prefer_tr=prefer_tr)
                if term_final and not year_final:
                    return _curriculum_year_followup(prefer_tr=prefer_tr)
                if not year_final and not term_final:
                    return _curriculum_year_followup(prefer_tr=prefer_tr)

            if is_answering_profiling and question_type != "year_term" and next_question:
                return next_question

            # If they didn't seem to be answering the profiling question, ask again
            if not is_answering_profiling:
                followup = _profiling_followup_message(
                    question_type,
                    profile,
                    prefer_tr=prefer_tr,
                    correction=_looks_like_profile_correction(question),
                )
                return followup or next_question

    # â”€â”€ ACADEMIC QUESTION HANDLING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # From here on, proceed with the existing logic
    if _looks_like_profile_update_only(question, history, last_bot_q):
        history_with_current = list(history or []) + [{"role": "user", "content": question}]
        profile = extract_profile_from_history(history_with_current)
        next_question, question_type = get_next_profiling_question(
            profile,
            question_lower,
            require_curriculum_details=_requires_curriculum_details(question, history),
            prefer_tr=prefer_tr,
        )
        if next_question:
            return _profiling_followup_message(
                question_type,
                profile,
                prefer_tr=prefer_tr,
                correction=_looks_like_profile_correction(question),
            ) or next_question
        profile_ack = _format_profile_context_ack(profile, prefer_tr=prefer_tr)
        if profile_ack and not _should_resume_planning_after_profile_completion(question, history):
            return profile_ack

    parsed = parse_request_slots(question, history, last_bot_q)
    full_question = parsed["analysis_question"]
    prefer_tr = prefer_tr or _looks_turkish(full_question)
    history_with_current = list(history or []) + [{"role": "user", "content": question}]
    profile = extract_profile_from_history(history_with_current)

    # â”€â”€ gather ALL potentially relevant context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    corrections = {}
    low_confidence_score = None
    if codes:
        course_contexts, missing_codes, corrections, low_confidence_score = get_context_for_codes(collection, codes)
        if low_confidence_score is not None and not course_contexts:
            print(f"[RETRIEVAL] Low confidence ({low_confidence_score:.2f}), falling back")
            return get_fallback_response(session_language)
        if course_contexts:
            specific_response = extract_course_specific_response(course_contexts, full_question)
            if specific_response:
                return specific_response

            ql = full_question.lower()
            if len(codes) == 1 and any(p in ql for p in ["tell me about", "more details", "details about", "information about", "what is ", "what about", "how about"]):
                correction_note = _format_course_correction_note(corrections, _looks_turkish(full_question))
                if correction_note:
                    return correction_note + "\n\n" + format_course_direct_response(course_contexts[0])
                return format_course_direct_response(course_contexts[0])
            context_parts.extend(course_contexts)
            correction_note = _format_course_correction_note(corrections, prefer_tr)
            if correction_note:
                context_parts.append(correction_note)

    # 2. department detection
    dept = parsed["department"] or profile.get("department")
    year = parsed["year"]
    term = parsed["term"]
    start_year = parsed["start_year"]
    failed_courses = extract_failed_courses_from_context(question, history, last_bot_q)
    gpa_value = extract_gpa_from_context(full_question, history)
    program_name = profile.get("program_name")
    curriculum_url = profile.get("curriculum_url")
    faculty = profile.get("faculty")
    if dept and start_year:
        resolved = resolve_ieu_department(dept, faculty_name=faculty)
        if resolved:
            for row in (resolved.get("rows") or []):
                if row.get("entry_year") == start_year:
                    curriculum_url = row.get("curriculum_url") or curriculum_url
                    break
    entry_year_note = _maybe_entry_year_note(dept, start_year, prefer_tr, program_name=program_name, curriculum_url=curriculum_url)
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
    credit_limit_intent = parsed["intent"] == "credit_limit" or is_gpa_credit_limit_question(full_question, history)
    regulations_intent = (
        parsed["intent"] == "regulations" or is_regulations_question(full_question)
    )
    curriculum_followup_intent = _looks_like_curriculum_followup(question)

    if credit_limit_intent:
        regs_answer = _format_regulations_answer(full_question, history, gpa_value)
        if regs_answer:
            return regs_answer

    planning_intent = (
        parsed["intent"] == "planning"
        or is_semester_question(full_question)
        or curriculum_followup_intent
        or ((year and term) and planning_keywords)
        or planning_keywords
    ) and not course_lookup_intent
    both_semesters_intent = planning_intent and wants_both_semesters(full_question)
    web_info_intent = (
        parsed["intent"] == "web_info" or is_web_info_question(full_question)
    ) and not (regulations_intent or planning_intent or graduation_intent or course_lookup_intent)
    web_docs = _query_web_info_context(collection, full_question) if web_info_intent else []

    if web_docs:
        if web_info_intent and client is None:
            return _format_web_info_answer(full_question, web_docs) or web_docs[0]
        if client is None and not (course_lookup_intent or planning_intent or graduation_intent):
            return _format_web_info_answer(full_question, web_docs) or web_docs[0]
        context_parts.extend(web_docs)
    elif web_info_intent:
        return get_fallback_response(session_language)

    if course_lookup_intent and not planning_intent and not graduation_intent and not regulations_intent:
        if course_contexts:
            # Otherwise return full course information
            formatted = [format_course_direct_response(doc) for doc in course_contexts]
            correction_note = _format_course_correction_note(corrections, _looks_turkish(full_question))
            if correction_note:
                formatted.insert(0, correction_note)
            if missing_codes:
                missing_list = ", ".join(missing_codes)
                formatted.append(get_fallback_response(session_language))
            return "\n\n".join(formatted)
        if missing_codes:
            missing_list = ", ".join(missing_codes)
            return get_fallback_response(session_language)

    # If user shares failed course codes after a planning answer, reuse the latest
    # planning tuple and return a retake-prioritized recommendation.
    if failed_courses and not planning_intent and not graduation_intent and not regulations_intent:
        latest_plan = get_latest_planning_tuple_from_history(history)
        if latest_plan:
            h_dept, h_year, h_term = latest_plan
            h_courses = get_curriculum_courses(
                h_dept,
                h_year,
                h_term,
                start_year=start_year,
                program_name=program_name,
                curriculum_url=curriculum_url,
            )
            if h_courses:
                retry_entry_year_note = _maybe_entry_year_note(
                    h_dept,
                    start_year,
                    prefer_tr,
                    program_name=program_name,
                    curriculum_url=curriculum_url,
                )
                return _format_semester_courses(
                    h_dept,
                    h_year,
                    h_term,
                    h_courses,
                    failed_courses=failed_courses,
                    start_year=start_year,
                    program_name=program_name,
                    curriculum_url=curriculum_url,
                    gpa_value=gpa_value,
                    entry_year_note=retry_entry_year_note,
                    prefer_tr=prefer_tr,
                )

    # Graduation requirement questions should not fall back to generic course search.
    if graduation_intent and (not dept or not program_name):
        if not faculty:
            return _profiling_followup_message("faculty", profile, prefer_tr=prefer_tr, correction=_looks_like_profile_correction(question))
        return _profiling_followup_message("dept", profile, prefer_tr=prefer_tr, correction=_looks_like_profile_correction(question))

    if graduation_intent and dept:
        return _format_graduation_requirements(
            prefer_tr=prefer_tr,
            is_medicine=_is_medicine_department(dept, program_name=program_name, faculty=faculty),
        )

    # Deterministic regulations answers (works even without LLM).
    if regulations_intent and (not planning_intent) and (not graduation_intent) and (not web_info_intent):
        regs_answer = _format_regulations_answer(full_question, history, gpa_value)
        if regs_answer:
            return regs_answer

    # For course-planning questions, require missing details via follow-up
    # instead of falling into semantic search (which can hallucinate).
    # Also handle the case where user is directly answering department follow-up
    # with a short reply like "ce".
    if (not graduation_intent) and (planning_intent or (last_bot_q == "dept" and parsed["department"])):
        if not dept or not program_name:
            if not faculty:
                return _profiling_followup_message("faculty", profile, prefer_tr=prefer_tr, correction=_looks_like_profile_correction(question))
            return _profiling_followup_message("dept", profile, prefer_tr=prefer_tr, correction=_looks_like_profile_correction(question))
        if not start_year and has_multiple_curriculum_variants(dept, program_name=program_name):
            return _curriculum_entry_year_followup(prefer_tr=prefer_tr)
        if not year or not term:
            return _profiling_followup_message("year_term", profile, prefer_tr=prefer_tr, correction=_looks_like_profile_correction(question))

    # Deterministic combined answer when the user asks both semesters of a year.
    if (not graduation_intent) and both_semesters_intent and dept and year:
        if not start_year and has_multiple_curriculum_variants(dept, program_name=program_name):
            return _curriculum_entry_year_followup(prefer_tr=prefer_tr)

        fall_courses = _get_curriculum_courses_robust(
            dept,
            year,
            "Fall",
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
            faculty=faculty,
        )
        spring_courses = _get_curriculum_courses_robust(
            dept,
            year,
            "Spring",
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
            faculty=faculty,
        )

        parts = []
        if fall_courses:
            parts.append(_format_semester_courses(
                dept,
                year,
                "Fall",
                fall_courses,
                failed_courses=failed_courses,
                start_year=start_year,
                program_name=program_name,
                curriculum_url=curriculum_url,
                gpa_value=gpa_value,
                prefer_tr=prefer_tr,
            ))
        if spring_courses:
            parts.append(_format_semester_courses(
                dept,
                year,
                "Spring",
                spring_courses,
                failed_courses=failed_courses,
                start_year=start_year,
                program_name=program_name,
                curriculum_url=curriculum_url,
                gpa_value=gpa_value,
                prefer_tr=prefer_tr,
            ))

        if parts:
            answer = "\n\n".join(parts)
            if regulations_intent and _is_credit_load_question(full_question):
                regs_answer = _format_regulations_answer(full_question, history, gpa_value)
                if regs_answer:
                    answer = answer + "\n\n" + regs_answer.replace("FOLLOWUP_NEEDED: ", "")
            if entry_year_note:
                answer = answer + "\n\n" + entry_year_note
            return answer

    # Deterministic semester answer once dept + year + term are known.
    if (not graduation_intent) and (planning_intent or last_bot_q in ["year_term", "start_year"]) and dept and year and term:
        courses = _get_curriculum_courses_robust(
            dept,
            year,
            term,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
            faculty=faculty,
        )
        if not courses:
            if last_bot_q in ["year_term", "start_year"]:
                # Try an LLM-guided retry before giving up on curriculum matching.
                parsed_retry = parse_request_slots(full_question, history, last_bot_q)
                dept = parsed_retry["department"] or dept
                year = parsed_retry["year"] or year
                term = parsed_retry["term"] or term
                start_year = parsed_retry["start_year"] or start_year
                entry_year_note = _maybe_entry_year_note(
                    dept,
                    start_year,
                    prefer_tr,
                    program_name=program_name,
                    curriculum_url=curriculum_url,
                )
                courses = _get_curriculum_courses_robust(
                    dept,
                    year,
                    term,
                    start_year=start_year,
                    program_name=program_name,
                    curriculum_url=curriculum_url,
                    faculty=faculty,
                )
        if not courses:
            return get_fallback_response(session_language)
        answer = _format_semester_courses(
            dept,
            year,
            term,
            courses,
            failed_courses=failed_courses,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
            gpa_value=gpa_value,
            entry_year_note=entry_year_note,
            prefer_tr=prefer_tr,
        )
        if regulations_intent and _is_credit_load_question(full_question):
            regs_answer = _format_regulations_answer(full_question, history, gpa_value)
            if regs_answer:
                answer = answer + "\n\n" + regs_answer.replace("FOLLOWUP_NEEDED: ", "")
        return answer

    if dept and year and term:
        courses = get_curriculum_courses(
            dept,
            year,
            term,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
        )
        if courses:
            elec_count = sum(1 for c in courses
                           if c.get("code", "").startswith("ELEC"))
            regular = [
                f"- {c['code']}" if (c.get("code") or "").strip().startswith("ELEC")
                else f"- {c['code']}: {c['name']}"
                for c in courses
            ]
            curr_text = _t(
                f"{dept} Year {year} {term} Semester courses:\n" + "\n".join(regular),
                f"{dept} - {year}. sÄ±nÄ±f {term} dÃ¶nemi dersleri:\n" + "\n".join(regular),
                prefer_tr,
            )
            if elec_count:
                curr_text += _t(
                    f"\n\nPlus {elec_count} elective slot(s).",
                    f"\n\nAyrÄ±ca {elec_count} seÃ§meli ders hakkÄ±nÄ±z var.",
                    prefer_tr,
                )
            context_parts.append(curr_text)

    # 4. graduation requirements if dept mentioned
    if (not graduation_intent) and dept and any(k in full_question.lower() for k in
                   ["graduate", "graduation", "finish", "complete",
                    "mezun", "bitir"]):
        context_parts.append(
            _format_graduation_requirements(
                prefer_tr=prefer_tr,
                is_medicine=_is_medicine_department(dept, program_name=program_name, faculty=faculty),
            )
        )

    # 5. regulations if GPA/credits/grades mentioned (allow mixed planning+rules questions)
    if regulations_intent or any(k in full_question.lower() for k in
        ["gpa", "credit", "grade", "kredi", "not", "ortalama",
         "how many", "can i take", "withdraw", "probation", "letter grade", "harf"]):
        regs = _get_regulations_from_db(collection)
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
        certificate_question = any(
            kw in full_question.lower()
            for kw in [
                "student certificate", "degree certificate", "diploma", "mezuniyet belgesi",
                "diplomam", "öğrenci belgesi",
            ]
        )
        if certificate_question:
            context_parts = get_certificate_context(collection, query_texts, n_results=3, max_docs=6)
        else:
            context_parts = _query_relevant_context(collection, query_texts, n_results=3, max_docs=6)

        if certificate_question and not context_parts:
            return _t(
                "You can request your degree certificate or student certificate through the Student Information System or by visiting the Student Affairs office (Öğrenci İşleri). For more info visit: https://oim.ieu.edu.tr/en",
                "Diploma veya öğrenci belgenizi Öğrenci Bilgi Sistemi üzerinden ya da Öğrenci İşleri ofisine başvurarak talep edebilirsiniz. Detaylı bilgi için: https://oim.ieu.edu.tr/tr",
                prefer_tr,
            )

    if not context_parts:
        try:
            prefer_tr = _conversation_prefers_turkish(question, history)
            lang_instruction = "Always reply in Turkish." if prefer_tr else "Always reply in English."
            system_prompt = (
                "You are EkoBot, a friendly and witty academic assistant for Izmir University of Economics (IEU). "
                "You help students with courses, schedules, prerequisites, and academic planning. "
                "If it's a joke, respond with light humor. If it's sarcasm or frustration, respond with empathy and a gentle redirect. "
                "If it's a typo or unclear, ask them to clarify in a friendly way. "
                "If it's just casual chat, respond warmly but briefly and invite them to ask about their courses. "
                "If they seem frustrated or confused, be extra warm and offer to help step by step. "
                "Keep your response under 2 sentences. Never mention that you're an AI or that you can't help. "
                f"{lang_instruction}"
            )
            user_prompt = f"The student just sent you this message: '{question}'."
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
            )
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
        except Exception:
            pass
        prefer_tr = _conversation_prefers_turkish(question, history)
        return _soft_fallback_message(session_language)

    # â”€â”€ ask LLM to reason over ALL context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # truncate to avoid token limit (increased to 15000 to preserve evaluation system)
    if conversation_snapshot:
        context_parts.insert(0, conversation_snapshot)

    combined = "\n\n---\n\n".join(context_parts)
    if len(combined) > 15000:
        combined = combined[:15000]

    return call_llm(full_question, [combined], history, client_type=client_type, session_language=session_language)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _format_semester_courses(
    dept,
    year,
    term,
    courses,
    failed_courses=None,
    start_year=None,
    program_name=None,
    curriculum_url=None,
    gpa_value=None,
    entry_year_note=None,
    prefer_tr=False,
):
    elec_count = sum(1 for c in courses if c.get("code", "").startswith("ELEC"))
    term_tr = "Guz" if str(term).lower() == "fall" else "Bahar" if str(term).lower() == "spring" else term
    lines = [_t(
        f"{dept} - Year {year} {term} Semester:\n",
        f"{dept} - {year}. sinif {term_tr} donemi:\n",
        prefer_tr,
    )]
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
            program_name=program_name,
            curriculum_url=curriculum_url,
        )
        offered_term_by_code = get_scoped_offered_terms(
            dept,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
        )
        positions = get_curriculum_course_positions(
            dept,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
        )
        blocked_chain_preview = get_blocked_dependency_chain(
            failed_courses,
            dept,
            year,
            term,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
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
        if code.startswith("ELEC"):
            lines.append(f"- {code}")
        elif code in blocked_by_code:
            lines.append(_t(
                f"- {code}: {name} (blocked now by failed prerequisite chain via {blocked_by_code[code]})",
                f"- {code}: {name} (su anda {blocked_by_code[code]} basarisiz onkosul zinciri nedeniyle engelli)",
                prefer_tr,
            ))
        else:
            lines.append(f"- {code}: {name}")

    if elec_count > 0:
        lines.append(_t(
            f"\nYou also have {elec_count} elective slot(s) this semester.",
            f"\nBu donemde {elec_count} secmeli ders hakkiniz var.",
            prefer_tr,
        ))
        lines.append(_t(
            "If you want the elective options, ask me and I'll list them.",
            "Secmeli ders seceneklerini isterseniz, size listeleyebilirim.",
            prefer_tr,
        ))

    if failed_courses:
        current_codes = {(c.get("code") or "").strip().upper() for c in courses}
        lines.append(_t(
            "\nRetake priority (failed courses you mentioned):",
            "\nTekrar onceligi (soylediginiz basarisiz dersler):",
            prefer_tr,
        ))
        for code in failed_courses:
            offered_term = (
                (positions.get(code, {}).get("term") or offered_term_by_code.get(code) or "")
            ).strip()
            offered_note = f"; offered in {offered_term}" if offered_term else ""
            if code in current_codes:
                lines.append(_t(
                    f"- {code} (already in this semester curriculum; prioritize retaking this course{offered_note})",
                    f"- {code} (zaten bu donem mufredatinda; bu dersi tekrar almak oncelikli olsun{offered_note})",
                    prefer_tr,
                ))
            else:
                lines.append(_t(
                    f"- {code} (not listed in this semester curriculum; add it in its offered term{offered_note} if your credit load allows)",
                    f"- {code} (bu donem mufredatinda listelenmiyor; kredi yukunuz yetiyorsa sunuldugu donemde ekleyin{offered_note})",
                    prefer_tr,
                ))

        blocked_chain = get_blocked_dependency_chain(
            failed_courses,
            dept,
            year,
            term,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
            reverse_graph=scoped_reverse_graph,
        )
        if blocked_chain:
            lines.append(_t(
                "\nBlocked dependency chain (following semesters):",
                "\nEngellenen onkosul zinciri (sonraki donemler):",
                prefer_tr,
            ))
            for item in blocked_chain[:20]:
                title = f"{item['code']}"
                if item.get("name"):
                    title += f" - {item['name']}"
                lines.append(
                    _t(
                        f"- {title} (Year {item['year']} {item['term']}) blocked by {item['blocked_by']}",
                        f"- {title} ({item['year']}. sinif {item['term']}) {item['blocked_by']} nedeniyle engelleniyor",
                        prefer_tr,
                    )
                )
            if len(blocked_chain) > 20:
                lines.append(_t(
                    f"- ...and {len(blocked_chain) - 20} more downstream blocked course(s).",
                    f"- ...ve diger {len(blocked_chain) - 20} engelli ders daha.",
                    prefer_tr,
                ))
        else:
            lines.append(_t(
                "\nNo downstream prerequisite blockers were found in the available curriculum data.",
                "\nMevcut mufredat verisinde sonraki donemlerde onkosul engeli bulunamadi.",
                prefer_tr,
            ))

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
            lines.append(_t(
                "\nSafe alternatives for this term (from later years, excluding courses blocked by your failed-chain):",
                "\nBu donem icin guvenli alternatifler (son yillardan, basarisiz ders zincirinizle engellenmeyen dersler):",
                prefer_tr,
            ))
            for item in alternatives:
                title = f"{item['code']}"
                if item.get("name"):
                    title += f" - {item['name']}"
                lines.append(_t(
                    f"- {title} (Year {item['year']} {item['term']})",
                    f"- {title} ({item['year']}. sinif {item['term']})",
                    prefer_tr,
                ))
        else:
            lines.append(_t(
                "\nNo safe same-term alternatives were found from later semesters in the available curriculum data.",
                "\nMevcut mufredat verisinde sonraki donemlerden guvenli ayni-donem alternatif bulunamadi.",
                prefer_tr,
            ))

        recovery_order = build_suggested_recovery_order(
            failed_courses,
            blocked_chain,
            dept,
            start_year=start_year,
            program_name=program_name,
            curriculum_url=curriculum_url,
        )
        if recovery_order:
            lines.append(_t(
                "\nSuggested recovery order:",
                "\nOnerilen telafi sirasi:",
                prefer_tr,
            ))
            for i, item in enumerate(recovery_order[:20], start=1):
                title = item["code"]
                if item.get("name"):
                    title += f" - {item['name']}"
                semester_part = ""
                if item.get("year") and item.get("term"):
                    semester_part = _t(
                        f" (Year {item['year']} {item['term']})",
                        f" ({item['year']}. sinif {item['term']})",
                        prefer_tr,
                    )
                reason_label = item.get("reason") or _t("dependency order", "bagimlilik sirasi", prefer_tr)
                lines.append(_t(
                    f"{i}. {title}{semester_part} [{reason_label}]",
                    f"{i}. {title}{semester_part} [{reason_label}]",
                    prefer_tr,
                ))
            if len(recovery_order) > 20:
                lines.append(_t(
                    f"...and {len(recovery_order) - 20} more item(s).",
                    f"...ve {len(recovery_order) - 20} tane daha.",
                    prefer_tr,
                ))

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
                lines.append(_t(
                    f"\nECTS-aware recovery schedule ({gpa_note}, max {max_ects} ECTS/semester):",
                    f"\nECTS'ye duyarlı telafi plani ({gpa_note}, en fazla {max_ects} ECTS/donem):",
                    prefer_tr,
                ))
                lines.append(_t(
                    "Strict prerequisite sequencing is applied (dependent courses are scheduled after unmet failed prerequisites and only in their offered Fall/Spring term).",
                    "Sikı onkosul siralamasi uygulanir (bagimli dersler, giderilmemis basarisiz onkosullardan sonra ve sadece sunulduklari Güz/Bahar doneminde planlanir).",
                    prefer_tr,
                ))
                for sem in schedule[:8]:
                    header = _t(
                        f"- Year {sem['year']} {sem['term']}: {sem['total_ects']:.0f}/{max_ects} ECTS",
                        f"- {sem['year']}. sinif {sem['term']}: {sem['total_ects']:.0f}/{max_ects} ECTS",
                        prefer_tr,
                    )
                    lines.append(header)
                    for item in sem["items"]:
                        title = item["code"]
                        if item.get("name"):
                            title += f" - {item['name']}"
                        ects_txt = _t(
                            f"{item['ects']:.0f} ECTS" if item.get("ects", 0) > 0 else "ECTS unknown",
                            f"{item['ects']:.0f} ECTS" if item.get("ects", 0) > 0 else "ECTS bilinmiyor",
                            prefer_tr,
                        )
                        reason_label = item.get("reason") or _t("dependency order", "bagimlilik sirasi", prefer_tr)
                        lines.append(_t(
                            f"  - {title} [{ects_txt}; {reason_label}]",
                            f"  - {title} [{ects_txt}; {reason_label}]",
                            prefer_tr,
                        ))
                if len(schedule) > 8:
                    lines.append(_t(
                        f"- ...and {len(schedule) - 8} more semester(s) in the recovery schedule.",
                        f"- ...ve telafi planinda {len(schedule) - 8} donem daha var.",
                        prefer_tr,
                    ))
    else:
        lines.append(_t(
            "\nIf you have any failed courses to retake, share their course codes (e.g. CE 201, MATH 153) and I will prioritize them in your plan.",
            "\nTekrar almaniz gereken basarisiz dersler varsa, ders kodlarini paylasin (orn. CE 201, MATH 153); planda onceliklendireyim.",
            prefer_tr,
        ))

    if entry_year_note:
        lines.append("")
        lines.append(entry_year_note)

    return "\n".join(lines)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TERMINAL TEST
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    print("EkoBot â€” type your question (or 'quit' to exit)")
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
