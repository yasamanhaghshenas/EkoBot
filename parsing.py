import json
import os
import re
from functools import lru_cache

from catalog import (
    load_ieu_department_directory,
    load_ieu_profile_hierarchy,
    load_program_catalog,
    resolve_ieu_department,
    resolve_ieu_faculty,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DEFAULT_VALID_DEPTS = [
    "CE", "SE", "EEE", "IE", "BME", "MCE", "AE", "CIVE", "GBE", "FE", "ME", "NURSING",
    "LAW", "VCD",
]

DEPT_CANONICAL = {
    "NURSING": "HEM",
    "LAW": "HUK",
}

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
    "LAW":  ["law", "law school", "faculty of law"],
    "VCD":  ["visual communication design"],
}

_VALID_COURSE_PREFIXES_CACHE = None


def load_valid_departments():
    """
    Build valid department code list from local data files.
    Keeps fallback defaults when files are missing.
    """
    codes = set(DEFAULT_VALID_DEPTS)

    for fname in ["curriculum.json", "dept_info.json", "program_catalog.json"]:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                code = (item.get("department") or item.get("department_code") or "").strip().upper()
                # Accept compact letter-only department codes, ignore long names.
                if re.fullmatch(r"[A-Z]{2,6}", code):
                    codes.add(code)
        except Exception as e:
            print(f"  {fname} department load error: {e}")

    return sorted(codes)


VALID_DEPTS = load_valid_departments()


def normalize_department_code(dept_code):
    if not dept_code:
        return dept_code
    code = dept_code.strip().upper()
    return DEPT_CANONICAL.get(code, code)


def _normalize_key(value):
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _build_department_indexes():
    valid = set(DEFAULT_VALID_DEPTS)
    canonical_lookup = {}

    # Keep the legacy manual aliases for backwards compatibility.
    manual_aliases = {
        "CE": ["computer engineering", "bilgisayar muhendisligi"],
        "SE": ["software engineering", "yazilim muhendisligi"],
        "EEE": ["eee", "electrical", "electronics", "elektrik", "elektronik"],
        "IE": ["industrial engineering", "endustri muhendisligi"],
        "BME": ["biomedical engineering", "biyomedikal"],
        "ME": ["mechanical engineering", "makine muhendisligi"],
        "CIVE": ["civil engineering", "insaat muhendisligi"],
        "AE": ["aerospace engineering", "havacilik"],
        "MCE": ["mechatronics", "mekatronik"],
        "GBE": ["genetics", "bioengineering"],
        "FE": ["food engineering"],
        "NURSING": ["nursing", "nursing department"],
        "LAW": ["law", "law school", "faculty of law"],
        "VCD": ["visual communication design"],
    }

    for canonical, aliases in manual_aliases.items():
        canonical_lookup[_normalize_key(canonical)] = canonical
        for alias in aliases:
            valid.add(alias)
            canonical_lookup[_normalize_key(alias)] = canonical

    for item in load_ieu_department_directory():
        canonical = (item.get("canonical_department") or "").strip()
        if not canonical:
            continue
        valid.add(canonical)
        canonical_lookup[_normalize_key(canonical)] = canonical

        source_label = (item.get("source_department") or "").strip()
        if source_label:
            valid.add(source_label)
            canonical_lookup[_normalize_key(source_label)] = canonical

        for alias in item.get("aliases") or []:
            alias = (alias or "").strip()
            if not alias:
                continue
            valid.add(alias)
            canonical_lookup[_normalize_key(alias)] = canonical

        for token in item.get("code_tokens") or []:
            token = (token or "").strip().upper()
            if not token:
                continue
            valid.add(token)
            canonical_lookup[_normalize_key(token)] = canonical

    return sorted(valid), canonical_lookup


VALID_DEPTS, _DEPT_CANONICAL_LOOKUP = _build_department_indexes()
VALID_DEPT_KEYS = {_normalize_key(v) for v in VALID_DEPTS}


def normalize_department_code(dept_code):
    if not dept_code:
        return dept_code
    key = _normalize_key(dept_code)
    return _DEPT_CANONICAL_LOOKUP.get(key, dept_code.strip())


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

    stripped = text.strip()
    stripped_key = _normalize_key(stripped)

    # Case 1: entire message is just a dept code e.g. user replied "ce"
    if stripped_key in VALID_DEPT_KEYS:
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
                r"\b(department|dept|major|student|bolum|bölüm)\b",
                text,
                re.IGNORECASE,
            )
            if profile_cue and re.search(rf"\b{re.escape(dept)}\b", text, re.IGNORECASE):
                return normalize_department_code(dept)
            continue
        pattern = rf"\b{re.escape(dept)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            return normalize_department_code(dept)

    # Case 3: full name aliases
    t_lower = text.lower()
    for dept, aliases in DEPT_ALIASES.items():
        for alias in aliases:
            # Short aliases like "ie" must be standalone words.
            if len(alias) <= 2:
                if re.search(rf"\b{re.escape(alias)}\b", t_lower):
                    return normalize_department_code(dept)
            else:
                if alias in t_lower:
                    return normalize_department_code(dept)

    resolved = resolve_ieu_department(text)
    if resolved:
        code = (resolved.get("department_code") or "").strip().upper()
        if code:
            return normalize_department_code(code)

    return None


def detect_department_from_history(history):
    """
    Scan recent history (newest first) to find what department the user mentioned.
    """
    for msg in reversed(history[-12:]):
        content = msg.get("content", "")
        dept = detect_department(content)
        if dept:
            return dept
    return None


def _contains_course_code(text):
    t = text or ""
    for m in re.finditer(r"\b([A-Z]{2,6})\s*(\d{3,4})\b", t, re.IGNORECASE):
        if _is_valid_course_prefix(m.group(1)):
            return True
    return False


def _get_valid_course_prefixes():
    global _VALID_COURSE_PREFIXES_CACHE
    if _VALID_COURSE_PREFIXES_CACHE is not None:
        return _VALID_COURSE_PREFIXES_CACHE

    prefixes = set()
    source_files = ["courses.json", "course_codes_full.json", "course_codes.json"]

    for fname in source_files:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for item in data:
                if isinstance(item, dict):
                    code = item.get("code") or ""
                else:
                    code = str(item)
                m = re.search(r"\b([A-Z]{2,6})\s*\d{3,4}\b", str(code).upper())
                if m:
                    prefixes.add(m.group(1))
        except Exception:
            continue

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

    stripped = text.strip()
    stripped_key = _normalize_key(stripped)
    if allow_short_code_reply and stripped_key in VALID_DEPT_KEYS:
        return normalize_department_code(stripped)

    dept = detect_department(text)
    if not dept:
        return None

    t = text.lower()
    explicit_cues = [
        "department", "major", "student", "i am", "i'm", "im ",
        "my dept", "my department", "my major", "bolum", "bölüm",
    ]

    if any(cue in t for cue in explicit_cues):
        return dept

    # If the message contains a course code and no explicit profile cue,
    # do not infer department identity from it.
    if _contains_course_code(text):
        return None

    return dept


def detect_department_strict(text, allow_short_code_reply=False):
    """
    Safer strict department detector that ignores course codes when the user
    is clearly answering a profiling question.
    """
    if not text:
        return None

    stripped = text.strip()
    stripped_key = _normalize_key(stripped)
    if allow_short_code_reply and stripped_key in VALID_DEPT_KEYS:
        return normalize_department_code(stripped)

    t = text.lower()
    explicit_cues = [
        "department", "major", "student", "i am", "i'm", "im ",
        "my dept", "my department", "my major", "bolum", "bأ¶lأ¼m",
    ]

    if any(cue in t for cue in explicit_cues):
        sanitized = re.sub(r"\b[A-Z]{2,6}\s*\d{3,4}\b", " ", text, flags=re.IGNORECASE)
        dept = detect_department(sanitized)
        if dept:
            return dept

    dept = detect_department(text)
    if not dept:
        return None

    if any(cue in t for cue in explicit_cues) and not _contains_course_code(text):
        return dept

    if _contains_course_code(text):
        return None

    return dept


def detect_department_from_history_strict(history, allow_short_code_reply=False):
    """
    Scan recent user messages for explicit department identity only.
    """
    for msg in reversed(history[-12:]):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        dept = detect_department_strict(
            content,
            allow_short_code_reply=allow_short_code_reply,
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
            allow_short_code_reply=allow_short_code_reply,
        )
        if dept:
            return dept
        return detect_department_from_history_strict(
            history,
            allow_short_code_reply=allow_short_code_reply,
        )

    dept = detect_department(question)
    if dept:
        return dept
    return detect_department_from_history(history)


# ─────────────────────────────────────────────
# LAST BOT QUESTION DETECTION
# ─────────────────────────────────────────────

def detect_faculty_choice(text):
    """
    Detect a faculty/school choice from a button click or typed reply.
    Returns the normalized faculty label or None.
    """
    if not text:
        return None

    faculty = resolve_ieu_faculty(text)
    if faculty:
        return faculty.get("label")
    return None


def get_last_bot_question_type(history):
    """
    Returns what the bot last asked about so we know how to interpret
    the user's next short reply like 'ce' or '2nd year spring'.
    Returns: 'faculty', 'dept', 'year_term', 'start_year', 'failed_courses', or None
    """
    last_assistant = None
    for msg in reversed(history[-12:]):
        if msg.get("role") != "assistant":
            continue
        last_assistant = msg.get("content", "")
        break

    if not last_assistant:
        return None

    content = last_assistant.lower()
    content_norm = _normalize_chatty_text(last_assistant)
    if any(phrase in content_norm for phrase in [
        "hangi fakulte",
        "hangi fakülte",
        "hangi okul",
        "fakulte",
        "fakülte",
    ]):
        return "faculty"
    if any(phrase in content_norm for phrase in [
        "hangi bolum",
        "hangi bölüm",
        "bolumunuz",
        "bölümünüz",
        "bolumde",
        "bölümde",
    ]):
        return "dept"
    if any(phrase in content_norm for phrase in [
        "universiteye hangi yil",
        "universiteye hangi yıl",
        "hangi yil girdiniz",
        "hangi yıl girdiniz",
        "hangi yilda kayit",
        "hangi yılda kayıt",
        "kayit yiliniz",
        "kayıt yılınız",
        "kayit yili",
        "kayıt yılı",
    ]):
        return "start_year"
    if any(phrase in content_norm for phrase in [
        "hangi sinif",
        "hangi sınıf",
        "kacinci sinif",
        "kaçıncı sınıf",
        "yil ve donem",
        "yıl ve dönem",
        "hangi donem",
        "hangi dönem",
    ]):
        return "year_term"
    if any(phrase in content_norm for phrase in [
        "basarisiz ders",
        "başarısız ders",
        "kaldigin ders",
        "kaldığın ders",
        "ders kod",
    ]):
        return "failed_courses"
    if True:
        if (
            "which faculty" in content
            or "what faculty" in content
            or "select your faculty" in content
            or "choose your faculty" in content
            or "which school" in content
            or "what school" in content
            or "faculty or school" in content
        ):
            return "faculty"
        if (
            "which department" in content
            or "what department" in content
            or "what is your department" in content
            or "hangi bölüm" in content
            or "hangi bolum" in content
            or "bölümde" in content
            or "bolumde" in content
        ):
            return "dept"
        if (
            "starting year" in content
            or "entry year" in content
            or "enrollment year" in content
            or "curriculum version" in content
            or "curriculum choice" in content
            or "which year did you start" in content
            or "which year did you enroll" in content
            or "what year did you enroll" in content
            or "üniversiteye hangi yıl" in content
            or "universiteye hangi yil" in content
            or "kayıt yılı" in content
            or "kayit yili" in content
            or "kaydoldu" in content
            or "kaydoldugun" in content
        ):
            return "start_year"
        if (
            "which year" in content
            or "what year" in content
            or "year and semester" in content
            or "kaçıncı sınıf" in content
            or "kacinci sinif" in content
            or "yıl ve dönem" in content
            or "yil ve donem" in content
        ):
            return "year_term"
        if (
            "which semester" in content
            or "what semester" in content
            or "hangi dönem" in content
            or "hangi donem" in content
        ):
            return "year_term"
        if (
            ("failed courses" in content and "course codes" in content)
            or "başarısız" in content
            or "basarisiz" in content
            or "kaldığın" in content
            or "kaldigin" in content
            or "ders kod" in content
        ):
            return "failed_courses"
    return None


def _normalize_chatty_text(text):
    t = (text or "").strip().casefold()
    if not t:
        return ""
    # Collapse stretched-out chat text like "hiiii" or "helloooo".
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _looks_like_greeting_or_small_talk(text):
    t = _normalize_chatty_text(text)
    if not t:
        return False

    words = re.findall(r"[a-z0-9']+", t)
    if not words:
        return False

    raw_lower = (text or "").casefold()
    has_repeated_letters = bool(re.search(r"([a-z])\1{2,}", raw_lower))

    greeting_words = {
        "hi", "hello", "hey", "greetings", "merhaba", "selam", "sup", "yo",
        "hiya", "hai", "hewwo", "helo", "slm", "naber", "tamam", "tmm",
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
        "boş ver",
        "bos ver",
        "neyse",
        "iyiyim",
    }
    filler_words = {
        "lol", "lmao", "haha", "ok", "okay", "k", "kk", "bruh", "bro",
        "ugh", "hmm", "umm", "idk", "tmm",
    }
    frustrated_phrases = {
        "you are useless",
        "you're useless",
        "this is stupid",
        "whatever",
        "nevermind",
        "nvm",
        "forget it",
        "saçma",
        "sacma",
        "işe yaramaz",
        "ise yaramaz",
        "salak",
    }
    keyboard_mash_patterns = [r"asdf", r"asdfjkl", r"qwerty", r"zxcvbn"]
    phrase = " ".join(words)
    if phrase in greeting_phrases:
        return True
    if has_repeated_letters and len(words) <= 3 and any(word in greeting_words or word in filler_words for word in words):
        return True
    if len(words) <= 3 and any(word in filler_words for word in words):
        return True
    if len(words) <= 4 and any(re.fullmatch(pattern, phrase) for pattern in keyboard_mash_patterns):
        return True
    if any(cue in raw_lower for cue in frustrated_phrases):
        return True
    if len(words) <= 3 and any(word in greeting_words for word in words):
        return True
    return False


def _is_likely_followup_reply(text):
    """
    Heuristic for short follow-up replies such as department code,
    year/term answer, or starting year.
    """
    if not text:
        return False

    t = _normalize_chatty_text(text)
    if not t:
        return False

    if _looks_like_greeting_or_small_talk(t):
        return True

    # Substantive questions should stay as anchors even if they mention a year/term.
    if is_semester_question(t) or is_graduation_question(t) or is_regulations_question(t):
        return False

    if detect_faculty_choice(t):
        return True

    if detect_department(t):
        return True

    y, term = extract_year_and_term(t)
    if y and term:
        return True

    if extract_year_only(t) is not None:
        return True

    if extract_term_only(t) is not None:
        return True

    if extract_starting_year(t):
        return True

    followup_cues = [
        "what about",
        "how about",
        "what if",
        "and what",
        "then what",
        "so what",
        "what now",
        "same question",
        "same one",
        "boş ver",
        "bos ver",
        "neyse",
    ]
    if any(cue in t for cue in followup_cues):
        return True

    correction_cues = [
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
        "boş ver",
        "bos ver",
        "neyse",
    ]
    if any(cue in t for cue in correction_cues):
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


# ─────────────────────────────────────────────
# QUESTION TYPE DETECTORS
# ─────────────────────────────────────────────

def is_graduation_question(text):
    keywords = [
        "graduation", "graduate", "how to graduate", "requirements to graduate",
        "mezuniyet", "mezun", "mezun olmak", "gerekli", "şart",
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
    return (
        "elective" in t or "seçmeli" in t or
        "what are the options" in t or "show me the options" in t or
        "electives for" in t or "available electives" in t
    )


def _has_course_code(text):
    return bool(re.search(r"\b[A-Z]{2,6}\s*\d{3,4}\b", text or "", re.IGNORECASE))


def is_prerequisite_lookup_question(text):
    """
    Detect questions that ask about prerequisites or instructors for a specific course.
    These should be treated as course lookup questions even if they mention "courses".
    """
    t = (text or "").lower()
    if not _has_course_code(text):
        return False

    cues = [
        "prereq", "prerequisite", "prerequisites",
        "before taking", "before i take", "before taking this",
        "need to pass", "need to complete", "must pass",
        "pass before", "take before", "required before",
        "to take", "for taking",
        "who teaches", "who is teaching", "instructor", "lecturer", "coordinator",
    ]
    return any(cue in t for cue in cues)


def is_course_lookup_question(text):
    t = (text or "").lower()
    if is_prerequisite_lookup_question(text):
        return True
    if is_semester_question(text) or _has_failed_course_cue(text):
        return False

    # Generic "what is ..." questions should not be treated as course lookups
    # unless a real course code is present.
    if not _has_course_code(text):
        return False

    if re.fullmatch(r"\s*[A-Z]{2,6}\s*\d{3,4}\s*", (text or "").strip(), re.IGNORECASE):
        return True

    # Follow-up lookup phrasing that still intends a single-course explanation.
    # Example: "what about CE 342?"
    if re.search(r"\b(what about|how about|and|also)\b", t) and re.search(r"\b[A-Z]{2,6}\s*\d{3,4}\b", text or "", re.IGNORECASE):
        return True

    lookup_cues = [
        "prereq", "prerequisite", "prerequisites",
        "credit", "credits", "ects",
        "syllabus", "course description", "description", "course aim",
        "tell me about", "information about", "details about", "info about",
        "course name", "instructor", "coordinator", "teacher", "teaches",
    ]
    return any(cue in t for cue in lookup_cues)


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
    m_target_word = re.search(r"\bfor\s+(the\s+)?(first|second|third|fourth|fifth|sixth)\s+year\b", t)
    if m_target_word:
        year = target_word_map.get(m_target_word.group(2))

    if year is None:
        m_target_num = re.search(r"\bfor\s+(the\s+)?([1-6])(?:st|nd|rd|th)?\s*year\b", t)
        if m_target_num:
            year = int(m_target_num.group(2))

    # numeric: "4th year", "2nd year", "1 year"
    m = re.search(r"\b([1-6])(?:st|nd|rd|th)?\s*year\b", t)
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
            if re.search(rf"\b{word}\s*year\b", t):
                year = num
                break

    # Turkish: "2. sınıf"
    if year is None:
        m_tr = re.search(r"\b([1-6])\.?\s*s[ıi]n[ıi]f\b", t)
        if m_tr:
            year = int(m_tr.group(1))

    if any(k in t for k in ["spring", "bahar"]):
        term = "Spring"
    elif any(k in t for k in ["fall", "autumn", "güz", "guz"]):
        term = "Fall"
    else:
        # Common English follow-up replies: "first semester", "second semester".
        if re.search(r"\b(first|1st|one)\s+(semester|term)\b", t):
            term = "Fall"
        elif re.search(r"\b(second|2nd|two)\s+(semester|term)\b", t):
            term = "Spring"

    return year, term


def extract_year_only(text):
    """
    Extract a year-of-study value even when the user doesn't say "year".
    Examples: "3rd", "third", "3", "2nd"
    """
    if not text:
        return None
    t = (text or "").strip().lower()

    # Pure numeric or ordinal replies.
    m = re.fullmatch(r"([1-6])(?:st|nd|rd|th)?", t)
    if m:
        return int(m.group(1))

    word_map = {
        "first": 1, "second": 2, "third": 3,
        "fourth": 4, "fifth": 5, "sixth": 6,
    }
    if t in word_map:
        return word_map[t]

    # Fallback: find an ordinal token somewhere (e.g., "im 3rd")
    m2 = re.search(r"\b([1-6])(?:st|nd|rd|th)\b", t)
    if m2:
        return int(m2.group(1))

    return None


def extract_term_only(text):
    if not text:
        return None
    t = (text or "").lower()
    if any(k in t for k in ["spring", "bahar"]):
        return "Spring"
    if any(k in t for k in ["fall", "autumn", "guz", "güz"]):
        return "Fall"

    # Common follow-up replies: "first semester", "second semester".
    if re.search(r"\b(first|1st|one)\s+(semester|term)\b", t):
        return "Fall"
    if re.search(r"\b(second|2nd|two)\s+(semester|term)\b", t):
        return "Spring"

    return None


def extract_year_and_term_from_history(history):
    year = None
    term = None

    for msg in reversed(history[-12:]):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "") or ""

        y, t = extract_year_and_term(content)
        y = y or extract_year_only(content)
        t = t or extract_term_only(content)

        if year is None and y is not None:
            year = y
        if term is None and t is not None:
            term = t
        if year is not None and term is not None:
            return year, term

    return year, term


def extract_starting_year(text):
    if not text:
        return None
    t = text.lower()
    compact = re.sub(r"\s+", " ", t).strip()

    cue_keywords = [
        "started", "starting year", "entry year", "enrollment year",
        "enroll", "enrolled", "admission year", "matriculation",
        "since", "giris", "giriş",
    ]

    if any(k in t for k in cue_keywords):
        m = re.search(r"\b(20\d{2})\b", t)
        if m:
            y = int(m.group(1))
            if 2000 <= y <= 2100:
                return y

    # Also accept short follow-up replies like "2023" after a bot prompt.
    m_plain = re.search(r"^\s*(20\d{2})\s*$", text)
    if m_plain:
        y = int(m_plain.group(1))
        if 2000 <= y <= 2100:
            return y

    # Accept curriculum-choice replies such as "2013-2019" or "2025+".
    m_range = re.fullmatch(r"(20\d{2})\s*(?:-|to)\s*(20\d{2})\+?", compact)
    if not m_range:
        m_range = re.search(r"\b(20\d{2})\s*(?:-|to)\s*(20\d{2})\b", compact)
    if m_range:
        y = int(m_range.group(1))
        if 2000 <= y <= 2100:
            return y

    m_plus = re.fullmatch(r"(20\d{2})\s*\+", compact)
    if not m_plus:
        m_plus = re.search(r"\b(20\d{2})\s*\+", compact)
    if m_plus:
        y = int(m_plus.group(1))
        if 2000 <= y <= 2100:
            return y

    return None


def extract_starting_year_from_history(history):
    for msg in reversed(history[-12:]):
        if msg.get("role") != "user":
            continue
        y = extract_starting_year_loose(msg.get("content", ""))
        if y:
            return y
    return None


def extract_starting_year_loose(text):
    """
    Extended start-year detection for phrases such as:
    - "enrolled in 2022"
    - "entered university in 2022"
    - "joined university in 2022"

    The base extractor stays conservative; this helper broadens coverage for
    user-written profile statements.
    """
    if not text:
        return None

    y = extract_starting_year(text)
    if y:
        return y

    t = text.lower()
    extra_cues = [
        "enrolled in", "entered university", "entered in",
        "joined university", "joined in", "admitted to", "matriculated",
    ]
    if any(k in t for k in extra_cues):
        m = re.search(r"\b(20\d{2})\b", t)
        if m:
            year = int(m.group(1))
            if 2000 <= year <= 2100:
                return year

    return None


# ─────────────────────────────────────────────
# COURSE CODE LOOKUP
# ─────────────────────────────────────────────

def extract_course_codes(text):
    pattern = r"\b([A-Z]{2,6})\s*(\d{3,4})\b"
    matches = re.findall(pattern, text or "", re.IGNORECASE)
    out = []
    for m in matches:
        prefix = (m[0] or "").upper()
        if not _is_valid_course_prefix(prefix):
            continue
        out.append(f"{prefix} {m[1]}")
    return _dedupe_keep_order(out)


def _extract_course_codes_loose(text):
    pattern = r"\b([A-Z]{2,6})\s*(\d{3,4})\b"
    matches = re.findall(pattern, text or "", re.IGNORECASE)
    return _dedupe_keep_order([f"{m[0].upper()} {m[1]}" for m in matches])


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
        "kald", "kaldim", "kaldım", "tekrar", "ret",
    ]
    return any(c in t for c in cues)


def extract_failed_courses_from_context(question, history, last_bot_q=None):
    has_failure_cue = _has_failed_course_cue(question) or last_bot_q == "failed_courses"
    codes_now = _dedupe_keep_order(extract_course_codes(question or ""))
    if codes_now and has_failure_cue:
        return codes_now
    if has_failure_cue:
        loose_now = _dedupe_keep_order(_extract_course_codes_loose(question or ""))
        if loose_now:
            return loose_now

    merged = []
    for msg in reversed(history[-12:]):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        codes = _dedupe_keep_order(extract_course_codes(content))
        if _has_failed_course_cue(content):
            if codes:
                merged.extend(codes)
            else:
                merged.extend(_dedupe_keep_order(_extract_course_codes_loose(content)))

    return _dedupe_keep_order(merged)


def get_latest_planning_tuple_from_history(history):
    pattern = r"\b([A-Z]{2,6})\s*-\s*Year\s*([1-6])\s*(Fall|Spring)\s*Semester\b"
    for msg in reversed(history[-12:]):
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


# ─────────────────────────────────────────────
# GPA EXTRACTION
# ─────────────────────────────────────────────

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
    for msg in reversed(history[-12:]):
        if msg.get("role") == "user":
            texts.append(msg.get("content") or "")

    cue_pattern = re.compile(
        r"(?:\bgpa\b|\bcgpa\b|\bgano\b|\bortalama\b)\D{0,12}([0-4](?:[\.,]\d{1,2})?)",
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


def is_regulations_question(text):
    t = text.lower()
    keywords = [
        "rule", "rules", "regulation", "regulations", "bylaw", "bylaws",
        "gpa", "grade", "grading", "coefficient", "credit load",
        "how many credits", "credits can i take", "academic standing",
        "how many ects", "ects can i take", "maximum ects", "max ects",
        "probation", "honor", "withdraw", "withdrawal", "letter grade",
        "aa", "bb", "cc", "dd", "ff", "gano", "not ortalaması",
        "akademik", "başarı koşulu", "credit limit",
        "kaç kredi", "ne kadar kredi",
        "kural", "kurallar", "yonetmelik", "yonerge",
    ]
    return any(k in t for k in keywords)


_WEB_INFO_STOPWORDS = {
    "a", "an", "and", "about", "at", "by", "for", "from", "in", "of", "on",
    "the", "to", "with", "via",
    "student", "students", "academic", "official", "office", "program",
    "programs", "service", "services", "guide", "info", "information",
    "system", "systems", "university", "department", "faculty", "center",
    "centre", "portal", "application", "applications", "support",
}

_WEB_INFO_HIGH_SIGNAL_TOKENS = {
    "blackboard", "calendar", "conduct", "deadline", "desk", "dmp", "dormitory", "emergency", "erasmus",
    "exchange", "europass", "guidance", "help", "internship", "library", "mobility",
    "minor", "obs", "registration", "security", "shuttle", "scholarship", "staff", "tim",
    "tlc", "transcript", "transfer", "visa",
}


def _normalize_web_info_phrase(text):
    words = []
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        token = raw
        if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is", "ics")):
            token = token[:-1]
        words.append(token)
    return " ".join(words)


@lru_cache(maxsize=1)
def _load_web_info_source_signals(path="ecobot_rules_sources.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  web info source read error: {e}")
        return []

    signals = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        combined = f"{title} {description}".strip()
        if not combined:
            continue

        title_norm = _normalize_web_info_phrase(title)
        tokens = []
        for token in _normalize_web_info_phrase(combined).split():
            if token in _WEB_INFO_STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)

        if tokens or title_norm:
            signals.append(
                {
                    "title_norm": title_norm,
                    "tokens": tokens,
                }
            )

    return signals


def is_web_info_question(text):
    t = (text or "").lower()

    # Pure greetings should never be treated as web-info questions.
    # This avoids false positives from short strings like "hi" matching
    # substrings inside words such as "traineeship".
    greeting_words = {
        "hi", "hello", "hey", "greetings", "merhaba", "selam", "sup", "yo",
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
    }
    q_words = re.findall(r"[a-z0-9']+", t)
    q_phrase = " ".join(q_words)
    if q_phrase in greeting_phrases:
        return False
    if len(q_words) <= 3 and any(word in greeting_words for word in q_words):
        return False

    # Regulations / GPA / credit-load questions should be handled by the
    # regulations path, not the generic web-info retriever.
    if is_regulations_question(t):
        return False

    keywords = [
        "erasmus", "exchange program", "exchange programs", "international exchange",
        "exchange coordinator", "coordinator", "coordinators", "student mobility",
        "student charter", "agreements and quotas", "forms", "incoming student mobility",
        "outgoing student mobility", "incoming international exchange", "outgoing international exchange",
        "free mover", "international partners", "office of international affairs",
        "visa", "residence permit", "insurance", "europass", "survival guide",
        "student affairs", "document requests", "administrative forms", "student id card",
        "semester leave", "duplicate diploma", "military procedures", "medical procedures",
        "career guidance", "internship", "guaranteed internship", "campus shuttle",
        "transportation", "dormitory", "cafes and restaurants", "student clubs",
        "events calendar", "library", "blackboard", "obs", "academic support", "student life",
        "academic calendar", "double major", "minor", "transfer regulation",
        "summer school", "english preparatory", "second foreign language", "credit transfer",
        "academic staff", "staff directory", "help desk", "security", "scholarship",
        "student code of conduct", "student information system", "ecobot rules",
        "eco bot rules", "ekobot rules",
    ]
    if any(k in t for k in keywords):
        return True

    q_norm = _normalize_web_info_phrase(t)
    if "ekobot rules" in q_norm or "ecobot rules" in q_norm or "eco bot rules" in q_norm:
        return True

    q_tokens = set(q_norm.split())
    if not q_tokens:
        return False

    for item in _load_web_info_source_signals():
        title_norm = item.get("title_norm") or ""
        if title_norm and (title_norm in q_norm or q_norm in title_norm):
            return True

        tokens = item.get("tokens") or []
        if not tokens:
            continue

        overlap = [token for token in tokens if token in q_tokens]
        if not overlap:
            continue

        if len(overlap) >= 2 or any(token in _WEB_INFO_HIGH_SIGNAL_TOKENS for token in overlap):
            return True

    return False


def is_out_of_scope_question(text):
    t = (text or "").lower()
    cues = [
        "weather", "forecast",
        "election", "politics",
        "poem", "poetry",
        "bitcoin", "crypto", "price prediction",
        "cook", "cooking", "recipe", "lasagna",
        "translate", "translation",
        "football", "soccer",
        "best phone", "phone to buy",
    ]
    return any(cue in t for cue in cues)
