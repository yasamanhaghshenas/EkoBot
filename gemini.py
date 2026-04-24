from groq import Groq
import chromadb
import re
import json
import os
from urllib.parse import urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from embeddings import SimpleHashEmbeddingFunction

GROQ_API_KEY = "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"

client_groq = Groq(api_key=GROQ_API_KEY)

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("ieu_courses", embedding_function=SimpleHashEmbeddingFunction())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DEFAULT_VALID_DEPTS = [
    "CE", "SE", "EEE", "IE", "BME", "MCE", "AE", "CIVE", "GBE", "FE", "ME", "NURSING"
]

DEPT_CANONICAL = {
    "NURSING": "HEM",
}

_COURSE_ECTS_CACHE = None
_VALID_COURSE_PREFIXES_CACHE = None


def load_valid_departments():
    """
    Build valid department code list from local data files.
    Keeps fallback defaults when files are missing.
    """
    codes = set(DEFAULT_VALID_DEPTS)

    for fname in ["curriculum.json", "dept_info.json"]:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                code = (item.get("department") or "").strip().upper()
                # Accept compact letter-only department codes, ignore long names.
                if re.fullmatch(r"[A-Z]{2,6}", code):
                    codes.add(code)
        except Exception as e:
            print(f"  {fname} department load error: {e}")

    return sorted(codes)


VALID_DEPTS = load_valid_departments()

DEPT_ALIASES = {
    "CE":   ["computer engineering", "bilgisayar muhendisligi", "bilgisayar mühendisliği"],
    "SE":   ["software engineering", "yazilim muhendisligi", "yazılım mühendisliği"],
    "EEE":  ["eee", "electrical", "electronics", "elektrik", "elektronik"],
    "IE":   ["ie", "industrial engineering", "endustri muhendisligi", "endüstri mühendisliği"],
    "BME":  ["bme", "biomedical engineering", "biyomedikal"],
    "ME":   ["mechanical engineering", "makine muhendisligi", "makine mühendisliği"],
    "CIVE": ["cie", "civil engineering", "insaat muhendisligi", "inşaat mühendisliği"],
    "AE":   ["ae", "aerospace engineering", "havacilik", "havacılık"],
    "MCE":  ["mce", "mechatronics", "mekatronik"],
    "GBE":  ["gbe", "genetics", "bioengineering"],
    "FE":   ["fe", "food engineering"],
    "NURSING": ["nursing", "nursing department"],
}


def normalize_department_code(dept_code):
    if not dept_code:
        return dept_code
    code = dept_code.strip().upper()
    return DEPT_CANONICAL.get(code, code)

# ─────────────────────────────────────────────
# DEPARTMENT DETECTION
# ─────────────────────────────────────────────

def detect_department(text):
    """
    Safely detect department code from text.
    Returns dept code string or None.
    Never matches 'me' from phrases like 'tell me', 'for me' etc.
    Never matches 'ae' from words like 'take', 'make' etc.
    """
    if not text:
        return None

    stripped = text.strip().upper()

    # Case 1: entire message is just a dept code e.g. user replied "ce"
    if stripped in VALID_DEPTS:
        return normalize_department_code(stripped)

    # Case 2: dept code appears as standalone word in message
    # Use strict word boundaries to avoid false matches
    for dept in VALID_DEPTS:
        # Skip short ambiguous codes for inline matching
        # These are only matched if the entire message is that code (Case 1)
        if dept in ["ME", "AE", "IE", "FE"]:
            # Allow short ambiguous codes only when profile cue exists.
            # Example: "I am in ME department" should be accepted.
            profile_cue = re.search(
                r'\b(department|dept|major|student|bolum|bölüm)\b',
                text,
                re.IGNORECASE
            )
            if profile_cue and re.search(rf'\b{re.escape(dept)}\b', text, re.IGNORECASE):
                return normalize_department_code(dept)
            continue
        pattern = rf'\b{re.escape(dept)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            return normalize_department_code(dept)

    # Case 3: full name aliases
    t_lower = text.lower()
    for dept, aliases in DEPT_ALIASES.items():
        for alias in aliases:
            # Short aliases like "ie" must be standalone words.
            if len(alias) <= 2:
                if re.search(rf'\b{re.escape(alias)}\b', t_lower):
                    return normalize_department_code(dept)
            else:
                if alias in t_lower:
                    return normalize_department_code(dept)

    return None


def detect_department_from_history(history):
    """
    Scan recent history (newest first) to find what department the user mentioned.
    """
    for msg in reversed(history[-10:]):
        content = msg.get("content", "")
        dept = detect_department(content)
        if dept:
            return dept
    return None


def _contains_course_code(text):
    t = text or ""
    for m in re.finditer(r'\b([A-Z]{2,6})\s*(\d{3,4})\b', t, re.IGNORECASE):
        if _is_valid_course_prefix(m.group(1)):
            return True
    return False


def _get_valid_course_prefixes():
    global _VALID_COURSE_PREFIXES_CACHE
    if _VALID_COURSE_PREFIXES_CACHE is not None:
        return _VALID_COURSE_PREFIXES_CACHE

    prefixes = set()
    if not os.path.exists("courses.json"):
        _VALID_COURSE_PREFIXES_CACHE = prefixes
        return prefixes

    try:
        with open("courses.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            code = item.get("code") or ""
            m = re.search(r'\b([A-Z]{2,6})\s*\d{3,4}\b', str(code).upper())
            if m:
                prefixes.add(m.group(1))
    except Exception:
        pass

    _VALID_COURSE_PREFIXES_CACHE = prefixes
    return prefixes


def _is_valid_course_prefix(prefix):
    p = (prefix or "").strip().upper()
    if not p:
        return False
    valid = _get_valid_course_prefixes()
    if not valid:
        return True
    return p in valid


def detect_department_strict(text, allow_short_code_reply=False):
    """
    Detect department only when the user explicitly indicates profile/major intent.
    Prevents treating course codes like 'CE 342' as the user's department.
    """
    if not text:
        return None

    stripped = text.strip().upper()
    if allow_short_code_reply and stripped in VALID_DEPTS:
        return stripped

    dept = detect_department(text)
    if not dept:
        return None

    t = text.lower()
    explicit_cues = [
        "department", "major", "student", "i am", "i'm", "im ",
        "my dept", "my department", "my major", "bolum", "bölüm"
    ]

    if any(cue in t for cue in explicit_cues):
        return dept

    # If the message contains a course code and no explicit profile cue,
    # do not infer department identity from it.
    if _contains_course_code(text):
        return None

    return dept


def detect_department_from_history_strict(history, allow_short_code_reply=False):
    """
    Scan recent user messages for explicit department identity only.
    """
    for msg in reversed(history[-10:]):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        dept = detect_department_strict(
            content,
            allow_short_code_reply=allow_short_code_reply
        )
        if dept:
            return dept
    return None


def detect_department_from_context(question, history, strict=False, allow_short_code_reply=False):
    """
    Try to find department from current question first, then history.
    """
    if strict:
        dept = detect_department_strict(
            question,
            allow_short_code_reply=allow_short_code_reply
        )
        if dept:
            return dept
        return detect_department_from_history_strict(
            history,
            allow_short_code_reply=allow_short_code_reply
        )

    dept = detect_department(question)
    if dept:
        return dept
    return detect_department_from_history(history)


# ─────────────────────────────────────────────
# LAST BOT QUESTION DETECTION
# ─────────────────────────────────────────────

def get_last_bot_question_type(history):
    """
    Returns what the bot last asked about so we know how to interpret
    the user's next short reply like 'ce' or '2nd year spring'.
    Returns: 'dept', 'year_term', 'start_year', 'failed_courses', or None
    """
    for msg in reversed(history[-6:]):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").lower()
        if "which department" in content or "what department" in content or "what is your department" in content:
            return "dept"
        if "starting year" in content or "entry year" in content or "which year did you start" in content:
            return "start_year"
        if "which year" in content or "what year" in content or "year and semester" in content:
            return "year_term"
        if "failed courses" in content and "course codes" in content:
            return "failed_courses"
    return None


def _is_likely_followup_reply(text):
    """
    Heuristic for short follow-up replies such as department code,
    year/term answer, or starting year.
    """
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    # Substantive questions should stay as anchors even if they mention a year/term.
    if is_semester_question(t) or is_graduation_question(t) or is_regulations_question(t):
        return False

    if detect_department(t):
        return True

    y, term = extract_year_and_term(t)
    if y and term:
        return True

    if extract_starting_year(t):
        return True

    return False


def get_latest_substantive_user_question(history):
    """
    Find the latest user message that represents the main intent/question,
    not a short follow-up reply.
    """
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if _is_likely_followup_reply(content):
            continue
        return content
    return None


def _normalize_term_value(value):
    if value is None:
        return None
    t = str(value).strip().lower()
    if not t:
        return None

    fall_patterns = ["fall", "autumn", "guz", "güz", "first semester", "1st semester", "first term", "1st term"]
    spring_patterns = ["spring", "bahar", "second semester", "2nd semester", "second term", "2nd term"]

    if any(p in t for p in fall_patterns):
        return "Fall"
    if any(p in t for p in spring_patterns):
        return "Spring"
    return None


def wants_both_semesters(text):
    t = (text or "").lower()
    patterns = [
        r"\b(first|1st)\s+(and|&)\s+(second|2nd)\s+(semester|term)\b",
        r"\b(second|2nd)\s+(and|&)\s+(first|1st)\s+(semester|term)\b",
        r"\b(both)\s+(semesters|terms)\b",
        r"\b(fall)\s+(and|&)\s+(spring)\b",
        r"\b(spring)\s+(and|&)\s+(fall)\b",
        r"\b(guz|güz)\s+(ve|and|&)\s+(bahar)\b",
        r"\b(bahar)\s+(ve|and|&)\s+(guz|güz)\b",
    ]
    return any(re.search(p, t) for p in patterns)


def _extract_json_object(text):
    if not text:
        return None
    match = re.search(r'\{.*\}', text, re.DOTALL)
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
        allow_short_code_reply=(last_bot_q == "dept")
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

    start_year = extract_starting_year(analysis_question)
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

    if likely_academic:
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
Conversation history (recent): {json.dumps(history[-6:], ensure_ascii=False)}
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
# QUESTION TYPE DETECTORS
# ─────────────────────────────────────────────

def is_graduation_question(text):
    keywords = [
        "graduation", "graduate", "how to graduate", "requirements to graduate",
        "mezuniyet", "mezun", "mezun olmak", "gerekli", "şart"
    ]
    return any(k in text.lower() for k in keywords)


def is_semester_question(text):
    t = text.lower()

    keywords = [
        "next semester",
        "which courses should i take",
        "what courses should i take",
        "what courses do i",
        "courses do i need",
        "courses should i pick",
        "what should i take",
        "which courses",
        "courses for spring",
        "courses for fall",
        "i should take",
        "bu dönem",
        "hangi ders",
        "hangi dersleri",
    ]
    if any(k in t for k in keywords):
        return True

    # Catch broader phrasings like:
    # "what classes should I take", "which subjects do I take now", etc.
    if re.search(r"\b(what|which)\b.*\b(course|courses|class|classes|subject|subjects|ders|dersler)\b.*\b(should|do)\b.*\b(take|pick|choose|al)\b", t):
        return True

    # Also catch shorter direct planning requests.
    if re.search(r"\b(should i take|hangi ders almal(i|ı)y(i|ı)m)\b", t):
        return True

    # Catch natural phrasing like:
    # "what are my courses for third year first and second semester"
    # "show my courses for 2nd year"
    if re.search(r"\b(what|show|list|tell)\b.*\b(my\s+)?(courses|dersler|dersleri)\b", t):
        return True

    if re.search(r"\b(for|in)\b.*\b([1-6](st|nd|rd|th)?|first|second|third|fourth|fifth|sixth)\s+year\b", t):
        return True

    return False


def is_elective_question(text):
    t = text.lower()
    # Must contain 'elective' or 'seçmeli' — NOT just 'elec' which is too short
    # Also matches "what are my options" after context
    return ("elective" in t or "seçmeli" in t or
            "what are the options" in t or "show me the options" in t or
            "electives for" in t or "available electives" in t)


# ─────────────────────────────────────────────
# YEAR / TERM EXTRACTION
# ─────────────────────────────────────────────

def extract_year_and_term(text):
    t = text.lower()
    year = None
    term = None

    # Prefer target-year phrasing (e.g., "for third year") over earlier
    # self-state phrases like "I am in second year".
    target_word_map = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
    }
    m_target_word = re.search(r'\bfor\s+(the\s+)?(first|second|third|fourth|fifth|sixth)\s+year\b', t)
    if m_target_word:
        year = target_word_map.get(m_target_word.group(2))

    if year is None:
        m_target_num = re.search(r'\bfor\s+(the\s+)?([1-6])(?:st|nd|rd|th)?\s*year\b', t)
        if m_target_num:
            year = int(m_target_num.group(2))

    # numeric: "4th year", "2nd year", "1 year"
    m = re.search(r'\b([1-6])(?:st|nd|rd|th)?\s*year\b', t)
    if m and year is None:
        year = int(m.group(1))

    # word: "second year", "fourth year"
    if year is None:
        word_map = {
            "first": 1, "second": 2, "third": 3,
            "fourth": 4, "fifth": 5, "sixth": 6,
            "birinci": 1, "ikinci": 2, "üçüncü": 3,
            "dördüncü": 4,
        }
        for word, num in word_map.items():
            if re.search(rf'\b{word}\s*year\b', t):
                year = num
                break

    # Turkish: "2. sınıf"
    if year is None:
        m_tr = re.search(r'\b([1-6])\.?\s*s[ıi]n[ıi]f\b', t)
        if m_tr:
            year = int(m_tr.group(1))

    if any(k in t for k in ["spring", "bahar"]):
        term = "Spring"
    elif any(k in t for k in ["fall", "autumn", "güz", "guz"]):
        term = "Fall"
    else:
        # Common English follow-up replies: "first semester", "second semester".
        if re.search(r'\b(first|1st|one)\s+(semester|term)\b', t):
            term = "Fall"
        elif re.search(r'\b(second|2nd|two)\s+(semester|term)\b', t):
            term = "Spring"

    return year, term


def extract_year_and_term_from_history(history):
    for msg in reversed(history[-10:]):
        y, t = extract_year_and_term(msg.get("content", ""))
        if y and t:
            return y, t
    return None, None


def extract_starting_year(text):
    if not text:
        return None
    t = text.lower()

    cue_keywords = [
        "started", "starting year", "entry year", "enrollment year",
        "admission year", "since", "giris", "giriş"
    ]

    if any(k in t for k in cue_keywords):
        m = re.search(r'\b(20\d{2})\b', t)
        if m:
            y = int(m.group(1))
            if 2000 <= y <= 2100:
                return y

    # Also accept short follow-up replies like "2023" after a bot prompt.
    m_plain = re.search(r'^\s*(20\d{2})\s*$', text)
    if m_plain:
        y = int(m_plain.group(1))
        if 2000 <= y <= 2100:
            return y

    return None


def extract_starting_year_from_history(history):
    for msg in reversed(history[-10:]):
        if msg.get("role") != "user":
            continue
        y = extract_starting_year(msg.get("content", ""))
        if y:
            return y
    return None


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def get_graduation_text(dept_code):
    if not os.path.exists("dept_info.json"):
        return None
    try:
        with open("dept_info.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("department") == dept_code:
                text = (item.get("text") or "").strip()
                if text and "Hata / Error" not in text:
                    return text
    except Exception as e:
        print(f"  dept_info.json read error: {e}")

    # Fallback: try vector DB docs that are tagged as department requirements.
    try:
        results = collection.get(
            where={
                "$and": [
                    {"department": dept_code},
                    {"doc_type": "dept_requirements"},
                ]
            },
            include=["documents"],
        )
        docs = results.get("documents") or []
        if docs:
            text = (docs[0] or "").strip()
            if text and "Hata / Error" not in text:
                return text
    except Exception as e:
        print(f"  dept requirements vector lookup error: {e}")

    # Final fallback: fetch Bologna graduation page from the department domain.
    try:
        bologna_url = None
        if os.path.exists("curriculum.json"):
            with open("curriculum.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if item.get("department") != dept_code:
                    continue
                cur_url = (item.get("url") or "").strip()
                if not cur_url:
                    continue
                parsed = urlparse(cur_url)
                if parsed.scheme and parsed.netloc:
                    bologna_url = f"{parsed.scheme}://{parsed.netloc}/en/bologna/aid/9"
                    break

        if bologna_url:
            response = requests.get(
                bologna_url,
                timeout=15,
                verify=False,
                headers={"User-Agent": "Mozilla/5.0 (compatible; EkoBot/1.0)"},
            )
            if response.ok:
                response.encoding = "utf-8"
                soup = BeautifulSoup(response.text, "html.parser")
                content = (
                    soup.find("div", class_="col-md-9")
                    or soup.find("div", class_="col-lg-9")
                    or soup.select_one("main")
                    or soup.select_one("div.container")
                )
                if content:
                    text = content.get_text(strip=True, separator=" ")
                    if text and "Hata / Error" not in text:
                        return text
    except Exception as e:
        print(f"  dept requirements web fallback error: {e}")

    return None


def get_available_entry_years(dept_code):
    if not os.path.exists("curriculum.json"):
        return []
    years = set()
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("department") != dept_code:
                continue
            ey = item.get("entry_year")
            if ey is None:
                continue
            try:
                years.add(int(ey))
            except Exception:
                pass
    except Exception as e:
        print(f"  curriculum.json entry-year read error: {e}")
    return sorted(years)


def has_multiple_curriculum_variants(dept_code):
    """
    Returns True when a department appears to have multiple curriculum variants.
    This is used to decide whether to ask for the student's entry year.
    """
    if not os.path.exists("curriculum.json"):
        return False

    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = [item for item in data if item.get("department") == dept_code]
        if len(rows) <= 1:
            return False

        years = set()
        urls = set()
        for item in rows:
            ey = item.get("entry_year")
            if ey is not None:
                try:
                    years.add(int(ey))
                except Exception:
                    pass
            url = (item.get("url") or "").strip()
            if url:
                urls.add(url)

        if len(years) > 1:
            return True
        if not years and len(urls) > 1:
            return True
        if len(rows) > 1:
            return True

    except Exception as e:
        print(f"  curriculum.json variant check error: {e}")

    return False


def get_curriculum_courses(dept_code, year, term, start_year=None):
    if not os.path.exists("curriculum.json"):
        return []
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        matches = []
        for item in data:
            if item.get("department") != dept_code:
                continue

            item_entry_year = item.get("entry_year")
            if start_year is not None and item_entry_year is not None:
                try:
                    if int(item_entry_year) != int(start_year):
                        continue
                except Exception:
                    continue

            for sem_name, courses in item.get("semesters", {}).items():
                name = sem_name.lower()
                if str(year) in name and term.lower() in name:
                    if isinstance(courses, list):
                        matches.append((item_entry_year, courses))

        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            if start_year is not None:
                filtered = []
                for item_entry_year, courses in matches:
                    try:
                        if item_entry_year is not None and int(item_entry_year) == int(start_year):
                            filtered.append(courses)
                    except Exception:
                        continue
                if filtered:
                    return filtered[0]

            # No explicit entry year given: prefer the latest available curriculum.
            numeric_matches = []
            for item_entry_year, courses in matches:
                try:
                    if item_entry_year is not None:
                        numeric_matches.append((int(item_entry_year), courses))
                except Exception:
                    continue

            if numeric_matches:
                numeric_matches.sort(key=lambda pair: pair[0], reverse=True)
                return numeric_matches[0][1]

            return matches[0][1]
    except Exception as e:
        print(f"  curriculum.json read error: {e}")
    return []


def _normalize_course_code(value):
    if not value:
        return None
    t = str(value).strip().upper()
    m = re.search(r'\b([A-Z]{2,6})\s*(\d{3,4})\b', t)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}"


def _semester_sort_index(year, term):
    t = (term or "").strip().lower()
    term_order = 0 if t.startswith("fall") else 1
    return int(year) * 2 + term_order


def _extract_year_term_from_semester_name(sem_name):
    s = (sem_name or "")
    y_match = re.search(r'([1-6])\s*\.\s*year|([1-6])\s*year', s, re.IGNORECASE)
    if not y_match:
        return None, None
    year_raw = y_match.group(1) or y_match.group(2)
    term = None
    s_lower = s.lower()
    if "fall" in s_lower:
        term = "Fall"
    elif "spring" in s_lower:
        term = "Spring"
    if not term:
        return None, None
    try:
        return int(year_raw), term
    except Exception:
        return None, None


def _select_curriculum_row(dept_code, start_year=None):
    if not os.path.exists("curriculum.json"):
        return None
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  curriculum.json read error: {e}")
        return None

    rows = [item for item in data if item.get("department") == dept_code]
    if not rows:
        return None

    if start_year is not None:
        for item in rows:
            try:
                if item.get("entry_year") is not None and int(item.get("entry_year")) == int(start_year):
                    return item
            except Exception:
                continue

    if len(rows) == 1:
        return rows[0]

    with_year = []
    for item in rows:
        try:
            if item.get("entry_year") is not None:
                with_year.append((int(item.get("entry_year")), item))
        except Exception:
            continue
    if with_year:
        with_year.sort(key=lambda x: x[0], reverse=True)
        return with_year[0][1]

    return rows[0]


def get_curriculum_course_positions(dept_code, start_year=None):
    row = _select_curriculum_row(dept_code, start_year=start_year)
    if not row:
        return {}

    positions = {}
    semesters = row.get("semesters", {})
    for sem_name, courses in semesters.items():
        year, term = _extract_year_term_from_semester_name(sem_name)
        if not year or not term or not isinstance(courses, list):
            continue
        idx = _semester_sort_index(year, term)
        for c in courses:
            code = _normalize_course_code(c.get("code"))
            if not code:
                continue
            positions[code] = {
                "year": year,
                "term": term,
                "index": idx,
                "name": (c.get("name") or "").strip(),
            }
    return positions


def _extract_field_from_course_text(text, label):
    m = re.search(rf'^{re.escape(label)}:\s*(.+)$', text or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def _build_scoped_course_index(dept_code, start_year=None):
    positions = get_curriculum_course_positions(dept_code, start_year=start_year)
    candidate_codes = set(positions.keys())
    if not candidate_codes:
        return {}, positions

    if not os.path.exists("courses.json"):
        return {}, positions

    try:
        with open("courses.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  courses.json read error: {e}")
        return {}, positions

    per_code_candidates = {}
    for item in data:
        dep_code = _normalize_course_code(item.get("code"))
        if not dep_code or dep_code not in candidate_codes:
            continue

        text = item.get("text") or ""
        prereq_lines = re.findall(r'^Prerequisites:\s*(.+)$', text, re.IGNORECASE | re.MULTILINE)
        prereq_text = " ".join(prereq_lines)
        prereq_codes = [_normalize_course_code(x) for x in re.findall(r'\b[A-Z]{2,6}\s*\d{3,4}\b', prereq_text)]
        prereq_codes = sorted(set([p for p in prereq_codes if p]))

        item_dept = normalize_department_code((item.get("department") or "").strip().upper())
        semester_raw = (_extract_field_from_course_text(text, "Semester") or "").strip().lower()
        offered_term = None
        if semester_raw.startswith("fall"):
            offered_term = "Fall"
        elif semester_raw.startswith("spring"):
            offered_term = "Spring"
        course_type = (_extract_field_from_course_text(text, "Course Type") or "").lower()
        target_term = (positions.get(dep_code, {}).get("term") or "").lower()

        score = 0
        if item_dept == dept_code:
            score += 100
        elif item_dept:
            score -= 100
        if target_term and target_term in semester_raw:
            score += 10
        if "service" in course_type:
            score -= 1

        per_code_candidates.setdefault(dep_code, []).append({
            "score": score,
            "prereqs": prereq_codes,
            "offered_term": offered_term,
        })

    scoped_index = {}
    for dep_code, candidates in per_code_candidates.items():
        if not candidates:
            continue
        candidates.sort(key=lambda x: (-x["score"], len(x["prereqs"])))
        best = candidates[0]
        scoped_index[dep_code] = {
            "prereqs": set(best["prereqs"]),
            "offered_term": best.get("offered_term") or positions.get(dep_code, {}).get("term"),
        }

    # Ensure every curriculum code has at least term metadata even if
    # the detailed catalog record is unavailable.
    for code, pos in positions.items():
        scoped_index.setdefault(code, {
            "prereqs": set(),
            "offered_term": pos.get("term"),
        })

    return scoped_index, positions


def get_scoped_prerequisite_maps(dept_code, start_year=None):
    scoped_index, _ = _build_scoped_course_index(dept_code, start_year=start_year)
    prereq_map = {}
    for dep_code, meta in scoped_index.items():
        prereq_map[dep_code] = set(meta.get("prereqs") or set())

    reverse_graph = {}
    for dep_code, prereqs in prereq_map.items():
        for prereq in prereqs:
            reverse_graph.setdefault(prereq, set()).add(dep_code)

    return prereq_map, reverse_graph


def get_scoped_offered_terms(dept_code, start_year=None):
    scoped_index, _ = _build_scoped_course_index(dept_code, start_year=start_year)
    return {
        code: meta.get("offered_term")
        for code, meta in scoped_index.items()
        if meta.get("offered_term")
    }


def extract_gpa_from_context(question, history):
    def _to_float(raw):
        if raw is None:
            return None
        t = str(raw).strip().replace(",", ".")
        try:
            v = float(t)
            if 0.0 <= v <= 4.0:
                return v
        except Exception:
            return None
        return None

    texts = [(question or "")]
    for msg in reversed(history[-8:]):
        if msg.get("role") == "user":
            texts.append(msg.get("content") or "")

    cue_pattern = re.compile(
        r'(?:\bgpa\b|\bcgpa\b|\bgano\b|\bortalama\b)\D{0,12}([0-4](?:[\.,]\d{1,2})?)',
        re.IGNORECASE,
    )
    for text in texts:
        m = cue_pattern.search(text)
        if m:
            v = _to_float(m.group(1))
            if v is not None:
                return v

    return None


def get_credit_limit_by_gpa(gpa_value):
    if gpa_value is None:
        return 32
    if gpa_value >= 2.5:
        return 44
    if gpa_value >= 2.0:
        return 38
    return 32


def get_course_ects_map():
    global _COURSE_ECTS_CACHE
    if _COURSE_ECTS_CACHE is not None:
        return _COURSE_ECTS_CACHE

    ects_map = {}
    if not os.path.exists("courses.json"):
        _COURSE_ECTS_CACHE = ects_map
        return ects_map

    try:
        with open("courses.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  courses.json read error: {e}")
        _COURSE_ECTS_CACHE = ects_map
        return ects_map

    for item in data:
        code = _normalize_course_code(item.get("code"))
        if not code:
            continue
        text = item.get("text") or ""
        m = re.search(r'^ECTS:\s*(\d+(?:\.\d+)?)\s*$', text, re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        try:
            ects_map[code] = float(m.group(1))
        except Exception:
            continue

    _COURSE_ECTS_CACHE = ects_map
    return ects_map


def _index_to_year_term(index):
    idx = int(index)
    year = idx // 2
    term = "Fall" if idx % 2 == 0 else "Spring"
    return year, term


def _align_index_to_offering(index, offered_term):
    term = (offered_term or "").strip().lower()
    idx = int(index)
    if term.startswith("fall") and idx % 2 == 1:
        return idx + 1
    if term.startswith("spring") and idx % 2 == 0:
        return idx + 1
    return idx


def build_ects_aware_recovery_schedule(recovery_order, gpa_value, current_index, prereq_map=None, offered_term_by_code=None):
    if not recovery_order:
        return [], get_credit_limit_by_gpa(gpa_value)

    ects_map = get_course_ects_map()
    max_ects = get_credit_limit_by_gpa(gpa_value)
    prereq_map = prereq_map or {}
    offered_term_by_code = offered_term_by_code or {}
    recovery_codes = {item.get("code") for item in recovery_order if item.get("code")}
    planned_index_by_code = {
        item.get("code"): int(item.get("index", current_index))
        for item in recovery_order
        if item.get("code")
    }

    slots = {}
    scheduled_index = {}
    for item in recovery_order:
        code = item.get("code")
        if not code:
            continue
        ects = ects_map.get(code, 0.0)
        target_index = item.get("index", current_index)
        if target_index < current_index:
            target_index = current_index
        offered_term = item.get("term") or offered_term_by_code.get(code)
        target_index = _align_index_to_offering(target_index, offered_term)

        # Strict prerequisite sequencing: if prerequisite is in the recovery set,
        # this course cannot be in the same semester as that prerequisite.
        for prereq in prereq_map.get(code, set()):
            if prereq not in recovery_codes:
                continue
            if prereq in scheduled_index:
                target_index = max(target_index, scheduled_index[prereq] + 1)
            else:
                target_index = max(target_index, planned_index_by_code.get(prereq, current_index) + 1)
        target_index = _align_index_to_offering(target_index, offered_term)

        while True:
            used = sum(x.get("ects", 0.0) for x in slots.get(target_index, []))
            if ects <= 0:
                slots.setdefault(target_index, []).append({
                    "code": code,
                    "name": item.get("name", ""),
                    "ects": ects,
                    "reason": item.get("reason", "dependency order"),
                })
                break
            if used + ects <= max_ects:
                slots.setdefault(target_index, []).append({
                    "code": code,
                    "name": item.get("name", ""),
                    "ects": ects,
                    "reason": item.get("reason", "dependency order"),
                })
                break
            if (offered_term or "").strip().lower() in ["fall", "spring"]:
                target_index += 2
            else:
                target_index += 1

        scheduled_index[code] = target_index

    scheduled = []
    for idx in sorted(slots.keys()):
        year, term = _index_to_year_term(idx)
        items = slots[idx]
        total = sum(x.get("ects", 0.0) for x in items)
        scheduled.append({
            "index": idx,
            "year": year,
            "term": term,
            "total_ects": total,
            "items": items,
        })

    return scheduled, max_ects


def get_blocked_dependency_chain(failed_courses, dept_code, year, term, start_year=None, reverse_graph=None):
    failed = _dedupe_keep_order([_normalize_course_code(c) for c in (failed_courses or []) if _normalize_course_code(c)])
    if not failed:
        return []

    positions = get_curriculum_course_positions(dept_code, start_year=start_year)
    if not positions:
        return []

    reverse_graph = reverse_graph or get_scoped_prerequisite_maps(dept_code, start_year=start_year)[1]
    if not reverse_graph:
        return []

    try:
        current_idx = _semester_sort_index(int(year), term)
    except Exception:
        return []

    seen = set(failed)
    depth = {c: 0 for c in failed}
    parent = {}
    frontier = list(failed)

    while frontier:
        next_frontier = []
        for prereq in frontier:
            for dep in sorted(reverse_graph.get(prereq, [])):
                if dep in seen:
                    continue
                pos = positions.get(dep)
                if not pos:
                    continue
                if pos["index"] < current_idx:
                    continue
                seen.add(dep)
                depth[dep] = depth.get(prereq, 0) + 1
                parent[dep] = prereq
                next_frontier.append(dep)
        frontier = next_frontier

    blocked = []
    for code, blocker in parent.items():
        pos = positions.get(code)
        if not pos:
            continue
        blocked.append({
            "code": code,
            "name": pos.get("name", ""),
            "year": pos.get("year"),
            "term": pos.get("term"),
            "blocked_by": blocker,
            "depth": depth.get(code, 1),
            "index": pos.get("index", 9999),
        })

    blocked.sort(key=lambda x: (x["index"], x["depth"], x["code"]))
    return blocked


def build_suggested_recovery_order(failed_courses, blocked_chain, dept_code, start_year=None):
    failed = _dedupe_keep_order([_normalize_course_code(c) for c in (failed_courses or []) if _normalize_course_code(c)])
    positions = get_curriculum_course_positions(dept_code, start_year=start_year)

    order_items = []
    for code in failed:
        pos = positions.get(code, {})
        order_items.append({
            "code": code,
            "name": pos.get("name", ""),
            "year": pos.get("year"),
            "term": pos.get("term"),
            "depth": 0,
            "index": pos.get("index", 0),
            "reason": "failed course",
        })

    for item in blocked_chain or []:
        order_items.append({
            "code": item.get("code"),
            "name": item.get("name", ""),
            "year": item.get("year"),
            "term": item.get("term"),
            "depth": item.get("depth", 1),
            "index": item.get("index", 9999),
            "reason": f"depends on {item.get('blocked_by')}",
        })

    dedup = {}
    for item in order_items:
        code = item.get("code")
        if not code:
            continue
        if code not in dedup:
            dedup[code] = item
            continue
        prev = dedup[code]
        if (item.get("depth", 99), item.get("index", 9999)) < (prev.get("depth", 99), prev.get("index", 9999)):
            dedup[code] = item

    ordered = sorted(
        dedup.values(),
        key=lambda x: (x.get("depth", 99), x.get("index", 9999), x.get("code", "")),
    )
    return ordered


def _safe_alternative_courses_for_term(
    dept,
    year,
    term,
    current_codes,
    blocked_codes,
    positions,
    prereq_map,
    limit=8,
):
    try:
        current_idx = _semester_sort_index(int(year), term)
    except Exception:
        return []

    want_term = (term or "").strip().lower()
    alternatives = []

    for code, pos in positions.items():
        if code.startswith("ELEC"):
            continue
        if code in current_codes or code in blocked_codes:
            continue
        idx = pos.get("index", -1)
        if idx <= current_idx:
            continue
        if (pos.get("term") or "").strip().lower() != want_term:
            continue

        prereqs = prereq_map.get(code, set())
        if any(p in blocked_codes for p in prereqs):
            continue

        alternatives.append({
            "code": code,
            "name": pos.get("name", ""),
            "year": pos.get("year"),
            "term": pos.get("term"),
            "index": idx,
        })

    alternatives.sort(key=lambda x: (x["index"], x["code"]))
    return alternatives[:limit]


def get_electives(dept_code):
    if not os.path.exists("curriculum.json"):
        return []
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("department") == dept_code:
                return item.get("electives", [])
    except Exception as e:
        print(f"  curriculum.json electives read error: {e}")
    return []


def get_credit_load_rules():
    return """
IEU CREDIT LOAD RULES (from official regulations):
- Students with CGPA below 2.00: normal course load (maximum 32 ECTS per semester)
- Students with CGPA 2.00 to 2.49: normal load + 6 extra ECTS (up to 38 ECTS)
- Students with CGPA 2.50 and above: normal load + 12 extra ECTS (up to 44 ECTS)
Source: https://www.ieu.edu.tr/en/bylaws/type/read/id/72
"""


def get_regulations():
    if not os.path.exists("ieu_regulations.json"):
        return None
    try:
        with open("ieu_regulations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def is_regulations_question(text):
    t = text.lower()
    keywords = [
        "gpa", "grade", "grading", "coefficient", "credit load",
        "how many credits", "credits can i take", "academic standing",
        "probation", "honor", "withdraw", "withdrawal", "letter grade",
        "aa", "bb", "cc", "dd", "ff", "gano", "not ortalaması",
        "akademik", "başarı koşulu", "credit limit",
        "kaç kredi", "ne kadar kredi"
    ]
    return any(k in t for k in keywords)


# ─────────────────────────────────────────────
# COURSE CODE LOOKUP (ChromaDB)
# ─────────────────────────────────────────────

def extract_course_codes(text):
    pattern = r'\b([A-Z]{2,6})\s*(\d{3,4})\b'
    matches = re.findall(pattern, text or "", re.IGNORECASE)
    out = []
    for m in matches:
        prefix = (m[0] or "").upper()
        if not _is_valid_course_prefix(prefix):
            continue
        out.append(f"{prefix} {m[1]}")
    return list(set(out))


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        key = (item or "").strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _has_failed_course_cue(text):
    t = (text or "").lower()
    cues = [
        "failed", "fail", "retake", "retaking", "repeat", "repeating",
        "kald", "kaldim", "kaldım", "tekrar", "ret"
    ]
    return any(c in t for c in cues)


def extract_failed_courses_from_context(question, history, last_bot_q=None):
    codes_now = _dedupe_keep_order(extract_course_codes(question or ""))
    if codes_now and (_has_failed_course_cue(question) or last_bot_q == "failed_courses"):
        return codes_now

    merged = []
    for msg in reversed(history[-8:]):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        codes = _dedupe_keep_order(extract_course_codes(content))
        if not codes:
            continue
        if _has_failed_course_cue(content):
            merged.extend(codes)

    return _dedupe_keep_order(merged)


def get_latest_planning_tuple_from_history(history):
    pattern = r'\b([A-Z]{2,6})\s*-\s*Year\s*([1-6])\s*(Fall|Spring)\s*Semester\b'
    for msg in reversed(history[-10:]):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        m = re.search(pattern, content, re.IGNORECASE)
        if not m:
            continue
        dept = normalize_department_code(m.group(1).upper())
        year = int(m.group(2))
        term = "Fall" if m.group(3).lower().startswith("fall") else "Spring"
        return dept, year, term
    return None


def get_context_for_codes(codes):
    contexts = []
    for code in codes:
        found = False
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

        if not found:
            print(f"  '{code}' not found exactly, trying fuzzy search...")
            results = collection.query(query_texts=[code], n_results=1)
            if results["documents"][0]:
                doc = results["documents"][0][0]
                match = re.search(r'Course Code: ([A-Z]{2,6} \d{3,4})', doc)
                if match:
                    print(f"  Fuzzy matched '{code}' -> '{match.group(1)}'")
                    contexts.append(doc)
    return contexts


def format_course_direct_response(doc):
    def pick(label):
        m = re.search(rf"^{re.escape(label)}:\s*(.+)$", doc, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    code = pick("Course Code") or "Unknown"
    name = pick("Course Name") or "Unknown"
    dept = pick("Department") or "Unknown"
    semester = pick("Semester")
    credits = pick("Credits")
    ects = pick("ECTS")
    prereq = pick("Prerequisites")
    desc = pick("Course Description") or pick("Course Aim")

    lines = [f"{code} - {name}", f"Department: {dept}"]
    if semester:
        lines.append(f"Semester: {semester}")
    if credits:
        lines.append(f"Credits: {credits}")
    if ects:
        lines.append(f"ECTS: {ects}")
    if prereq:
        lines.append(f"Prerequisites: {prereq}")
    if desc:
        lines.append(f"Description: {desc}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(question, context_docs, history):
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
Always be specific and helpful — do not just say you don't have the information."""
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
        )
    })

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
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
            if y and not profile["year"]:
                profile["year"] = y
            if t and not profile["term"]:
                profile["term"] = t
        
        # Start year
        if not profile["start_year"]:
            profile["start_year"] = extract_starting_year(content)
        
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
            "department"
        )
    
    # Required: year and semester
    if not profile["year"] or not profile["term"]:
        return (
            "Got it! 👍\n\n"
            "What year and semester are you currently in?\n"
            "(e.g., '2nd year spring', 'third year fall', etc.)",
            "year_term"
        )
    
    # If curriculum has variants, ask for start year
    dept = profile["department"]
    if (len(get_available_entry_years(dept)) > 1 or has_multiple_curriculum_variants(dept)) and not profile["start_year"]:
        return (
            "Which year did you start university? (e.g., 2022, 2023)\n"
            "This helps me show you the right curriculum version.",
            "start_year"
        )
    
    # All required profiling is done
    return (None, None)


def is_greeting_or_small_talk(text):
    """
    Check if the message is just a greeting or small talk, not a substantive question.
    """
    t = text.lower().strip()
    greeting_patterns = [
        "hello", "hi", "hey", "greetings",
        "how are you", "what's up", "sup",
        "good morning", "good afternoon", "good evening",
        "merhaba", "selam", "nasılsın", "nasilsin"
    ]
    return any(pattern in t for pattern in greeting_patterns) and len(t) < 50


# ─────────────────────────────────────────────
# MAIN ASK FUNCTION
# ─────────────────────────────────────────────

def ask(question, history=[]):
    print(f"\n[EkoBot] Question: {question}")

    question_lower = question.lower().strip()
    
    # ── PROFILING FLOW ───────────────────────────────────────────────
    # Check if we need to complete user profiling first
    profile = extract_profile_from_history(history)
    next_question, question_type = get_next_profiling_question(profile, question_lower)
    
    # If the user is just saying hello/greeting and we haven't profiled yet,
    # respond with a greeting + first profiling question
    if is_greeting_or_small_talk(question) and next_question:
        return next_question
    
    # If profiling is not complete, ask for the next required info
    # (unless they're answering a follow-up profiling question)
    if next_question:
        # Check if they might be answering the profiling question
        # (e.g., replying "SE" to "what department are you in?")
        last_bot_msg = None
        if history:
            for msg in reversed(history[-3:]):
                if msg.get("role") == "assistant":
                    last_bot_msg = msg.get("content", "")
                    break
        
        # If the last bot message was asking for dept/year/term/start_year,
        # be more lenient and try to parse their answer
        is_answering_profiling = False
        if last_bot_msg:
            if "department" in last_bot_msg.lower() and detect_department(question):
                is_answering_profiling = True
            elif "year" in last_bot_msg.lower() and ("year" in question_lower or any(c.isdigit() for c in question)):
                is_answering_profiling = True
        
        # If they didn't seem to be answering the profiling question, ask again
        if not is_answering_profiling and not any(keyword in question_lower for keyword in 
                                                   ["course", "courses", "semester", "graduation", "requirements", 
                                                    "ders", "dersleri", "mezun", "graduation"]):
            return next_question
    
    # ── ACADEMIC QUESTION HANDLING ───────────────────────────────────────────────
    # From here on, proceed with the existing logic

    last_bot_q = get_last_bot_question_type(history)
    parsed = parse_request_slots(question, history, last_bot_q)
    full_question = parsed["analysis_question"]

    # ── gather ALL potentially relevant context ───────────────────────
    context_parts = []

    # 1. specific course codes
    all_text = full_question + " " + " ".join(
        m["content"] for m in history[-6:] if m.get("role") == "user")
    codes = extract_course_codes(all_text)
    if codes:
        course_contexts = get_context_for_codes(codes)
        if course_contexts:
            ql = full_question.lower()
            if len(codes) == 1 and any(p in ql for p in ["tell me about", "more details", "details about", "information about", "what is "]):
                return format_course_direct_response(course_contexts[0])
            context_parts.extend(course_contexts)

    # 2. department detection
    dept = parsed["department"]
    year = parsed["year"]
    term = parsed["term"]
    start_year = parsed["start_year"]
    failed_courses = extract_failed_courses_from_context(question, history, last_bot_q)
    gpa_value = extract_gpa_from_context(full_question, history)

    q_lower = full_question.lower()
    planning_keywords = any(k in q_lower for k in [
        "which course", "which courses", "what course", "what courses",
        "should i take", "i should take", "hangi ders"
    ])
    graduation_intent = parsed["intent"] == "graduation" or is_graduation_question(full_question)
    regulations_intent = (
        parsed["intent"] == "regulations" or is_regulations_question(full_question)
    ) and not planning_keywords
    planning_intent = (
        parsed["intent"] == "planning"
        or is_semester_question(full_question)
        or ((year and term) and planning_keywords)
        or (planning_keywords and not regulations_intent)
    )
    both_semesters_intent = planning_intent and wants_both_semesters(full_question)

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
        grad_text = get_graduation_text(dept)
        if not grad_text:
            return "I don't have graduation requirements for that department in my database yet."
        context_parts.append(grad_text[:2000])

    # For course-planning questions, require missing details via follow-up
    # instead of falling into semantic search (which can hallucinate).
    # Also handle the case where user is directly answering department follow-up
    # with a short reply like "ce".
    if (not graduation_intent) and (planning_intent or (last_bot_q == "dept" and parsed["department"])):
        if not dept:
            return "FOLLOWUP_NEEDED: Which department are you in? (major) (e.g. CE, SE, EEE, IE, BME, ME, Nursing)"
        if not year or not term:
            needs_start_year = len(get_available_entry_years(dept)) > 1 or has_multiple_curriculum_variants(dept)
            if needs_start_year and not start_year:
                return "FOLLOWUP_NEEDED: Which year and semester are you in, and which year did you start university? (e.g. 2nd year spring, started 2022)"
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
            return "\n\n".join(parts)

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
        return _format_semester_courses(dept, year, term, courses, failed_courses=failed_courses, start_year=start_year, gpa_value=gpa_value)

    if dept and year and term:
        courses = get_curriculum_courses(dept, year, term, start_year=start_year)
        if courses:
            elec_count = sum(1 for c in courses 
                           if c.get("code","").startswith("ELEC"))
            regular = [f"- {c['code']}: {c['name']}" 
                      for c in courses 
                      if not c.get("code","").startswith("ELEC")]
            curr_text = (f"{dept} Year {year} {term} Semester courses:\n" 
                        + "\n".join(regular))
            if elec_count:
                curr_text += (f"\n\nPlus {elec_count} elective slot(s). "
                             f"Available electives for {dept}:")
                electives = get_electives(dept)
                for e in electives[:20]:  # top 20 to avoid token limit
                    curr_text += f"\n- {e['code']}: {e['name']}"
            context_parts.append(curr_text)

    # 4. graduation requirements if dept mentioned
    if (not graduation_intent) and dept and any(k in full_question.lower() for k in 
                   ["graduate", "graduation", "finish", "complete",
                    "mezun", "bitir"]):
        grad_text = get_graduation_text(dept)
        if grad_text:
            context_parts.append(grad_text[:1000])

    # 5. regulations if GPA/credits/grades mentioned, but do not let them
    # override planning questions that also mention GPA.
    if (regulations_intent and not planning_intent) or (
        not planning_intent and any(k in full_question.lower() for k in
           ["gpa", "credit", "grade", "kredi", "not", "ortalama",
            "how many", "can i take"])
    ):
        context_parts.append(get_credit_load_rules())
        regs = get_regulations()
        if regs:
            context_parts.append(json.dumps(
                regs["grading"], ensure_ascii=False))

    # 6. semantic search fallback
    if not context_parts:
        results = collection.query(
            query_texts=[full_question], n_results=3)
        context_parts = results["documents"][0]

    if not context_parts:
        return ("I don't have enough information to answer that. "
                "Could you give me more details?")

    # ── ask LLM to reason over ALL context ───────────────────────────
    # truncate to avoid token limit
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
                lines.append(f"{i}. {title}{semester_part} [{item.get('reason', 'dependency order')}]")
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
                        lines.append(f"  - {title} [{ects_txt}; {item.get('reason', 'dependency order')}]")
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