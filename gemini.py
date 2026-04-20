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
    return bool(re.search(r'\b[A-Z]{2,5}\s*\d{3}\b', (text or "").upper()))


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
    Returns: 'dept', 'year_term', 'start_year', or None
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
    keywords = [
        "next semester", "which courses should i take",
        "what courses do i", "courses do i need", "courses should i pick",
        "what should i take", "which courses", "courses for spring",
        "courses for fall", "i should take", "bu dönem", "hangi ders", "hangi dersleri"
    ]
    return any(k in text.lower() for k in keywords)


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

    # numeric: "4th year", "2nd year", "1 year"
    m = re.search(r'\b([1-6])(?:st|nd|rd|th)?\s*year\b', t)
    if m:
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
                        matches.append(courses)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Ambiguous variant (likely multiple entry years) without enough context.
            return []
    except Exception as e:
        print(f"  curriculum.json read error: {e}")
    return []


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
    pattern = r'\b([A-Z]{2,5})\s*(\d{3})\b'
    matches = re.findall(pattern, text.upper())
    return list(set([f"{m[0]} {m[1]}" for m in matches]))


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
                match = re.search(r'Course Code: ([A-Z]{2,5} \d{3})', doc)
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
# MAIN ASK FUNCTION
# ─────────────────────────────────────────────

def ask(question, history=[]):
    print(f"\n[EkoBot] Question: {question}")

    last_bot_q = get_last_bot_question_type(history)
    full_question = question

    # if user is answering a follow-up, reconstruct the full question
    if last_bot_q in ["dept", "year_term", "start_year"]:
        anchor_question = get_latest_substantive_user_question(history)
        if anchor_question:
            full_question = anchor_question + " " + question

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
    dept = detect_department_from_context(
        full_question,
        history,
        strict=True,
        allow_short_code_reply=(last_bot_q == "dept")
    )

    # 3. curriculum data if dept + year/term mentioned
    year, term = extract_year_and_term(all_text)
    if not year or not term:
        year, term = extract_year_and_term_from_history(history)

    # If user is directly answering year/term follow-up, trust current reply first.
    if last_bot_q == "year_term":
        y_now, t_now = extract_year_and_term(question)
        if y_now:
            year = y_now
        if t_now:
            term = t_now

    # 4. starting year extraction (for curricula that vary by entry year)
    start_year = extract_starting_year(all_text)
    if not start_year:
        start_year = extract_starting_year_from_history(history)
    if last_bot_q == "start_year":
        y_start_now = extract_starting_year(question)
        if y_start_now:
            start_year = y_start_now

    q_lower = full_question.lower()
    graduation_intent = is_graduation_question(full_question)
    planning_intent = is_semester_question(full_question) or (
        (year and term) and any(k in q_lower for k in [
            "which course", "which courses", "what course", "what courses",
            "should i take", "i should take", "hangi ders"
        ])
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
    if (not graduation_intent) and (planning_intent or (last_bot_q == "dept" and detect_department(question))):
        if not dept:
            return "FOLLOWUP_NEEDED: Which department are you in? (major) (e.g. CE, SE, EEE, IE, BME, ME, Nursing)"
        if not year or not term:
            return "FOLLOWUP_NEEDED: Which year and semester are you in? (e.g. 2nd year spring)"
        entry_years = get_available_entry_years(dept)
        needs_start_year = len(entry_years) > 1 or has_multiple_curriculum_variants(dept)
        if needs_start_year and not start_year:
            return "FOLLOWUP_NEEDED: Which year did you start university? (e.g. 2023)"

    # Deterministic semester answer once dept + year + term are known.
    if (not graduation_intent) and (planning_intent or last_bot_q in ["year_term", "start_year"]) and dept and year and term:
        courses = get_curriculum_courses(dept, year, term, start_year=start_year)
        if not courses:
            return "I don't have that semester's curriculum data in my database."
        return _format_semester_courses(dept, year, term, courses)

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

    # 5. regulations if GPA/credits/grades mentioned
    if any(k in full_question.lower() for k in 
           ["gpa", "credit", "grade", "kredi", "not", "ortalama",
            "how many", "can i take"]):
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

def _format_semester_courses(dept, year, term, courses):
    elec_count = sum(1 for c in courses if c.get("code", "").startswith("ELEC"))
    lines = [f"{dept} - Year {year} {term} Semester:\n"]
    for c in courses:
        code = c.get("code", "").strip()
        name = c.get("name", "").strip()
        if not code.startswith("ELEC"):
            lines.append(f"- {code}: {name}")

    if elec_count > 0:
        lines.append(f"\nYou also have {elec_count} elective slot(s) this semester.")
        lines.append("Would you like to see what courses you can choose for those slots?")

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