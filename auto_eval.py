import argparse
import importlib
import json
import os
import re
import sys
import types
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_questions.json must contain a JSON array")
    return data


def _safe_text(value):
    return ("" if value is None else str(value)).encode("ascii", "backslashreplace").decode("ascii")


def load_departments_from_curriculum(path="curriculum.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        codes = []
        for item in data:
            d = (item.get("department") or "").strip().upper()
            if re.fullmatch(r"[A-Z]{2,6}", d):
                codes.append(d)
        return sorted(set(codes))
    except Exception:
        return []


def load_entry_years_by_department(path="curriculum.json"):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dept_years = {}
        for item in data:
            dept = (item.get("department") or "").strip().upper()
            if not re.fullmatch(r"[A-Z]{2,6}", dept):
                continue
            ey = item.get("entry_year")
            if ey is None:
                continue
            try:
                dept_years.setdefault(dept, set()).add(int(ey))
            except Exception:
                continue
        return {dept: sorted(years) for dept, years in dept_years.items()}
    except Exception:
        return {}


def load_curriculum_row_counts(path="curriculum.json"):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        counts = {}
        for item in data:
            dept = (item.get("department") or "").strip().upper()
            if not re.fullmatch(r"[A-Z]{2,6}", dept):
                continue
            counts[dept] = counts.get(dept, 0) + 1
        return counts
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_course_docs_from_courses_json(path="courses.json"):
    """
    Build a course-code -> course text map from the real scraped course data.
    This keeps mocked eval aligned with the actual dataset instead of relying
    on hand-written course fixtures.
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    def score_item(code, item_text, item_dept):
        score = 0
        code_prefix = (code.split()[0] if code else "").strip().upper()
        dept = (item_dept or "").strip().upper()
        text = item_text or ""
        if dept == code_prefix:
            score += 4
        if f"Course Code: {code}" in text:
            score += 3
        if "Course Lecturer(s)" in text or "Teacher/Lecturer(s)" in text:
            score += 2
        if "Course Coordinator" in text:
            score += 1
        if "Prerequisites" in text:
            score += 1
        return score, len(text)

    grouped = {}
    for item in data:
        code = (item.get("code") or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{2,6}\s*\d{3,4}", code):
            continue

        text = (item.get("text") or "").strip()
        if not text:
            continue

        dept = (item.get("department") or "").strip().upper()
        current = grouped.get(code)
        candidate = {
            "text": text,
            "score": score_item(code, text, dept),
            "department": dept,
        }
        if current is None or candidate["score"] > current["score"]:
            grouped[code] = candidate

    return {code: meta["text"] for code, meta in grouped.items()}


def _parse_semester_name(sem_name):
    s = sem_name or ""
    m_year = re.search(r"([1-6])\s*\.\s*Year|([1-6])\s*Year", s, re.IGNORECASE)
    if not m_year:
        return None, None
    year_raw = m_year.group(1) or m_year.group(2)
    term = None
    s_lower = s.lower()
    if "fall" in s_lower:
        term = "Fall"
    elif "spring" in s_lower:
        term = "Spring"
    if not term:
        return None, None
    return int(year_raw), term


def _semester_sort_index(year, term):
    term_order = 0 if (term or "").lower().startswith("fall") else 1
    return int(year) * 2 + term_order


def _year_ordinal(year):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(year, f"{year}th")


def load_latest_curriculum_rows(path="curriculum.json"):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    by_dept = {}
    for row in data:
        dept = (row.get("department") or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{2,6}", dept):
            continue
        by_dept.setdefault(dept, []).append(row)

    selected = {}
    for dept, rows in by_dept.items():
        rows_with_year = []
        for row in rows:
            ey = row.get("entry_year")
            try:
                rows_with_year.append((int(ey), row))
            except Exception:
                continue
        if rows_with_year:
            rows_with_year.sort(key=lambda x: x[0], reverse=True)
            selected[dept] = rows_with_year[0][1]
        else:
            selected[dept] = rows[0]

    return selected


def auto_generate_failed_course_cases(limit_per_department=1):
    rows = load_latest_curriculum_rows()
    row_counts = load_curriculum_row_counts()
    cases = []

    for dept, row in sorted(rows.items()):
        semesters = row.get("semesters") or {}
        sem_entries = []
        for sem_name, courses in semesters.items():
            year, term = _parse_semester_name(sem_name)
            if not year or not term or not isinstance(courses, list):
                continue
            sem_entries.append({
                "year": year,
                "term": term,
                "index": _semester_sort_index(year, term),
                "courses": courses,
            })

        sem_entries.sort(key=lambda x: x["index"])
        if len(sem_entries) < 2:
            continue

        generated = 0
        for i in range(1, len(sem_entries)):
            target = sem_entries[i]
            prior = sem_entries[i - 1]

            failed_code = None
            for c in prior["courses"]:
                code = (c.get("code") or "").strip().upper()
                if not code or code.startswith("ELEC"):
                    continue
                failed_code = code
                break

            if not failed_code:
                continue

            start_year = row.get("entry_year")
            started_phrase = ""
            if start_year is not None:
                started_phrase = f", started in {start_year}"
            elif row_counts.get(dept, 0) > 1:
                # Skip ambiguous variants when start year is required but unavailable.
                continue

            cases.append({
                "id": f"AUTO_FAILED_{dept}_{target['year']}_{target['term'].upper()}_{generated + 1}",
                "category": "planning",
                "question": (
                    f"I am in {dept} department{started_phrase}, {_year_ordinal(target['year'])} year "
                    f"{target['term'].lower()}. I failed {failed_code}. Which courses should I take?"
                ),
                "history": [],
                "checks": [
                    {"type": "contains", "value": "Retake priority"},
                    {"type": "contains", "value": "Suggested recovery order"},
                    {"type": "contains", "value": "ECTS-aware recovery schedule"},
                    {"type": "contains", "value": "Strict prerequisite sequencing is applied"},
                    {"type": "contains_any", "value": ["Blocked dependency chain", "No downstream prerequisite blockers"]},
                    {"type": "contains_any", "value": ["offered in Fall", "offered in Spring"]}
                ]
            })
            generated += 1
            if generated >= limit_per_department:
                break

    return cases


def auto_generate_cases(target_count=60):
    """
    Generate deterministic planning/refusal coverage based on available departments.
    This keeps evaluation broad even when the static case file is still small.
    """
    depts = load_departments_from_curriculum()
    entry_years = load_entry_years_by_department()
    row_counts = load_curriculum_row_counts()
    cases = []

    for dept in depts:
        latest_year = entry_years.get(dept, [])[-1] if entry_years.get(dept) else None
        fall_question = f"I am in {dept} department, 1st year fall. Which courses should I take?"
        spring_question = f"I am in {dept} department, 2nd year spring. Which courses should I take?"
        if latest_year:
            fall_question = f"I am in {dept} department, started in {latest_year}, 1st year fall. Which courses should I take?"
            spring_question = f"I am in {dept} department, started in {latest_year}, 2nd year spring. Which courses should I take?"

        if latest_year:
            cases.append({
                "id": f"AUTO_PLAN_{dept}_1_FALL",
                "category": "planning",
                "question": fall_question,
                "history": [],
                "checks": [
                    {"type": "contains", "value": dept},
                    {"type": "contains", "value": "Year 1 Fall Semester"}
                ]
            })
            cases.append({
                "id": f"AUTO_PLAN_{dept}_2_SPRING",
                "category": "planning",
                "question": spring_question,
                "history": [],
                "checks": [
                    {"type": "contains", "value": dept},
                    {"type": "contains", "value": "Year 2 Spring Semester"}
                ]
            })
            cases.append({
                "id": f"AUTO_PLAN_{dept}_PARAPHRASE_1",
                "category": "planning",
                "question": f"I am in {dept} department and started in {latest_year}, what courses should I take in 2nd year spring?",
                "history": [],
                "checks": [
                    {"type": "contains", "value": dept},
                    {"type": "contains", "value": "Year 2 Spring Semester"}
                ]
            })
            cases.append({
                "id": f"AUTO_PLAN_{dept}_PARAPHRASE_2",
                "category": "planning",
                "question": f"Could you show me the courses I need in 1st year fall for {dept}? I started in {latest_year}.",
                "history": [],
                "checks": [
                    {"type": "contains", "value": dept},
                    {"type": "contains", "value": "Year 1 Fall Semester"}
                ]
            })
        elif row_counts.get(dept, 0) == 1:
            cases.append({
                "id": f"AUTO_PLAN_{dept}_1_FALL",
                "category": "planning",
                "question": fall_question,
                "history": [],
                "checks": [
                    {"type": "contains", "value": dept},
                    {"type": "contains", "value": "Year 1 Fall Semester"}
                ]
            })
            cases.append({
                "id": f"AUTO_PLAN_{dept}_2_SPRING",
                "category": "planning",
                "question": spring_question,
                "history": [],
                "checks": [
                    {"type": "contains", "value": dept},
                    {"type": "contains", "value": "Year 2 Spring Semester"}
                ]
            })
        else:
            cases.append({
                "id": f"AUTO_PLAN_{dept}_1_FALL",
                "category": "planning",
                "question": fall_question,
                "history": [],
                "checks": [
                    {"type": "starts_with", "value": "FOLLOWUP_NEEDED:"},
                    {"type": "contains", "value": "Which year did you enroll in university"}
                ]
            })
            cases.append({
                "id": f"AUTO_PLAN_{dept}_2_SPRING",
                "category": "planning",
                "question": spring_question,
                "history": [],
                "checks": [
                    {"type": "starts_with", "value": "FOLLOWUP_NEEDED:"},
                    {"type": "contains", "value": "Which year did you enroll in university"}
                ]
            })

    # Fill up to target with out-of-scope refusal checks.
    refusal_prompts = [
        "What is the weather in Izmir tomorrow?",
        "Who will win the next election?",
        "Write me a poem about cats.",
        "What is the best phone to buy in 2026?",
        "Tell me bitcoin price prediction.",
        "How do I cook lasagna?",
        "Translate this random sentence to Japanese.",
        "Who is the best football team this year?"
    ]

    idx = 1
    while len(cases) < target_count:
        prompt = refusal_prompts[(idx - 1) % len(refusal_prompts)]
        cases.append({
            "id": f"AUTO_REFUSAL_{idx:03d}",
            "category": "refusal",
            "question": prompt,
            "history": [],
            "checks": [
                {"type": "contains_any", "value": ["I don't have", "IEU database"]}
            ]
        })
        idx += 1

    return cases


def call_api(question, history, url):
    payload = json.dumps({"question": question, "history": history}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            answer = data.get("answer", "")
            is_followup = bool(data.get("is_followup", False))
            return answer, is_followup, None
    except HTTPError as e:
        return "", False, f"HTTPError {e.code}: {e.reason}"
    except URLError as e:
        return "", False, f"URLError: {e.reason}"
    except Exception as e:
        return "", False, f"API call failed: {e}"


def _install_mock_modules():
    mock_groq = types.ModuleType("groq")

    class MockResponse:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]

    class MockCompletions:
        def create(self, model, messages, **kwargs):
            system_prompt = messages[0]["content"] if messages else ""
            user_prompt = messages[-1]["content"]
            m = re.search(
                r"Current user message:\s*(.*?)(?:\nLatest substantive user question:|\nLast bot question type:|\nConversation history:|\Z)",
                system_prompt,
                re.IGNORECASE | re.DOTALL,
            )
            if not m:
                m = re.search(
                    r"Latest substantive user question:\s*(.*?)(?:\nLast bot question type:|\nConversation history:|\Z)",
                    system_prompt,
                    re.IGNORECASE | re.DOTALL,
                )
            if not m:
                m = re.search(r"student question:\s*(.+)$", user_prompt, re.IGNORECASE | re.DOTALL)
            q_text = (m.group(1) if m else user_prompt).strip().lower()

            def has_pattern(patterns):
                return any(re.search(p, q_text) for p in patterns)

            if "student-request parser" in system_prompt.lower() or "return only valid json" in system_prompt.lower():
                dept = None
                dept_codes = load_departments_from_curriculum()
                for code, aliases in {
                    "CE": ["computer engineering", "ce"],
                    "SE": ["software engineering", "se"],
                    "EEE": ["eee", "electrical", "electronics"],
                    "IE": ["ie", "industrial engineering"],
                    "BME": ["bme", "biomedical"],
                    "ME": ["me", "mechanical engineering"],
                    "MCE": ["mce", "mechatronics"],
                    "AE": ["ae", "aerospace engineering"],
                    "CIVE": ["cive", "civil engineering"],
                    "GBE": ["gbe", "genetics"],
                    "FE": ["fe", "food engineering"],
                    "HEM": ["nursing", "hem"],
                }.items():
                    if code in {"ME", "AE", "IE", "FE"} and not re.search(r"\b(department|dept|major|student|bolum|bölüm)\b", q_text):
                        continue
                    patterns = []
                    for alias in aliases:
                        if len(alias) <= 2:
                            patterns.append(rf"\b{re.escape(alias)}\b")
                        else:
                            patterns.append(rf"\b{re.escape(alias)}\b")
                    if has_pattern(patterns):
                        dept = code
                        break

                if dept is None:
                    for code in dept_codes:
                        if code in {"ME", "AE", "IE", "FE"} and not re.search(r"\b(department|dept|major|student|bolum|bölüm)\b", q_text):
                            continue
                        if re.search(rf"\b{re.escape(code.lower())}\b", q_text):
                            dept = code
                            break

                year = None
                m_year_for = re.search(r"for\s+(the\s+)?(first|second|third|fourth|fifth|sixth|[1-6](?:st|nd|rd|th)?)\s+year", q_text)
                if m_year_for:
                    word = m_year_for.group(2)
                    year_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
                    year = year_map.get(word, None)
                    if year is None:
                        year = int(re.sub(r"\D", "", word))
                else:
                    m_year = re.search(r"\b([1-6])(?:st|nd|rd|th)?\s+year\b", q_text)
                    if m_year:
                        year = int(m_year.group(1))

                term = None
                if any(k in q_text for k in ["spring", "bahar", "second semester", "2nd semester", "second term"]):
                    term = "Spring"
                elif any(k in q_text for k in ["fall", "autumn", "güz", "guz", "first semester", "1st semester", "first term"]):
                    term = "Fall"

                start_year = None
                m_start = re.search(r"(?:started|starting year|entry year|since)\D*(20\d{2})", q_text)
                if m_start:
                    start_year = int(m_start.group(1))

                intent = "planning"
                if any(k in q_text for k in ["graduation", "graduate", "mezun"]):
                    intent = "graduation"
                elif any(k in q_text for k in ["gpa", "credit", "grade", "not ortalaması", "credit load", "how many ects", "ects can i take", "maximum ects", "max ects"]):
                    intent = "regulations"
                elif any(k in q_text for k in ["course", "courses", "what should i take", "which courses", "hangi ders"]):
                    intent = "planning"

                return MockResponse(json.dumps({
                    "intent": intent,
                    "department": dept,
                    "year": year,
                    "term": term,
                    "start_year": start_year,
                    "confidence": 0.88,
                }))

            if "weather" in q_text:
                return MockResponse("I don't have that information in the IEU database.")
            if "=== ieu database context" in user_prompt.lower() and any(
                k in q_text for k in ["erasmus", "exchange", "coordinator", "international"]
            ):
                if "coordinator" in q_text:
                    return MockResponse(
                        "The IEU website context points to the Exchange Program Coordinators and Erasmus / international exchange pages."
                    )
                return MockResponse(
                    "The IEU website context includes Erasmus+ student mobility rules, agreements, forms, and exchange program information."
                )
            if "graduation" in q_text or "graduate" in q_text:
                return MockResponse("To graduate, you need 240 ECTS and at least 2.00 CGPA.")
            if "below 2.00" in q_text or "below 2" in q_text:
                return MockResponse("If GPA is below 2.00, normal load applies with maximum 32 ECTS.")
            if "2.2" in q_text or "2.20" in q_text:
                return MockResponse("With GPA between 2.00 and 2.49, you may take up to 38 ECTS (+6 extra).")
            if "how many credits" in q_text or "gpa" in q_text:
                return MockResponse("With GPA 2.50 and above, you may take up to 44 ECTS (+12 extra).")
            return MockResponse("I don't have that information in the IEU database.")

    class MockGroq:
        def __init__(self, api_key=None):
            self.chat = types.SimpleNamespace(completions=MockCompletions())

    mock_groq.Groq = MockGroq
    sys.modules["groq"] = mock_groq

    mock_chroma = types.ModuleType("chromadb")

    fixture_docs = load_course_docs_from_courses_json()

    class MockCollection:
        def query(self, query_texts=None, n_results=1, where=None):
            code = None
            if where and isinstance(where, dict):
                code = where.get("code")
            if not code and query_texts:
                q = str(query_texts[0]).upper()
                m = re.search(r"\b([A-Z]{2,6}\s*\d{3,4})\b", q)
                if m:
                    code = re.sub(r"\s+", " ", m.group(1)).strip()

            if code and code in fixture_docs:
                return {"documents": [[fixture_docs[code]]]}
            return {"documents": [[]]}

        def get(self, where=None, include=None):
            code = None
            if where and isinstance(where, dict):
                code = where.get("code")
            if code and code in fixture_docs:
                return {"documents": [[fixture_docs[code]]]}
            return {"documents": [[]]}

    class MockClient:
        def __init__(self, path="./db"):
            self.path = path

        def get_or_create_collection(self, name, **kwargs):
            return MockCollection()

    def persistent_client(path="./db"):
        return MockClient(path=path)

    mock_chroma.PersistentClient = persistent_client
    sys.modules["chromadb"] = mock_chroma


def call_direct(question, history, mocked=False):
    try:
        if mocked:
            _install_mock_modules()
        gemini = importlib.import_module("gemini")
        answer = gemini.ask(question, history)
        is_followup = answer.startswith("FOLLOWUP_NEEDED:")
        return answer, is_followup, None
    except Exception as e:
        return "", False, f"Direct mode failed: {e}"


def normalize(text):
    return (text or "").strip().lower()


def run_check(answer, check):
    ctype = check.get("type")
    value = check.get("value")
    ans = answer or ""
    ans_n = normalize(ans)

    if ctype == "contains":
        return normalize(str(value)) in ans_n, f"contains '{value}'"
    if ctype == "not_contains":
        return normalize(str(value)) not in ans_n, f"not_contains '{value}'"
    if ctype == "starts_with":
        return ans.startswith(str(value)), f"starts_with '{value}'"
    if ctype == "contains_any":
        vals = value if isinstance(value, list) else [value]
        ok = any(normalize(str(v)) in ans_n for v in vals)
        return ok, f"contains_any {vals}"

    return False, f"unknown check type '{ctype}'"


def evaluate_case(case, mode, api_url):
    q = case.get("question", "")
    history = case.get("history", [])

    if mode == "api":
        answer, is_followup, err = call_api(q, history, api_url)
    elif mode == "direct":
        answer, is_followup, err = call_direct(q, history, mocked=False)
    elif mode == "direct-mocked":
        answer, is_followup, err = call_direct(q, history, mocked=True)
    else:
        return {
            "id": case.get("id", "UNKNOWN"),
            "passed": False,
            "error": f"Unsupported mode: {mode}",
            "checks": [],
            "answer": "",
            "is_followup": False,
            "category": case.get("category", "uncategorized")
        }

    check_results = []
    passed = err is None
    for check in case.get("checks", []):
        ok, desc = run_check(answer, check)
        check_results.append({"ok": ok, "description": desc})
        passed = passed and ok

    return {
        "id": case.get("id", "UNKNOWN"),
        "category": case.get("category", "uncategorized"),
        "passed": passed,
        "error": err,
        "checks": check_results,
        "answer": answer,
        "is_followup": is_followup
    }


def summarize(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    by_category = {}
    for r in results:
        cat = r.get("category", "uncategorized")
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "by_category": by_category
    }


def suggest_fix(result):
    answer = (result.get("answer") or "").lower()
    err = (result.get("error") or "").lower()
    rid = result.get("id", "")

    if "http" in err or "urlerror" in err:
        return "API mode failed. Ensure FastAPI server is running at --api-url and endpoint /chat is reachable."
    if "direct mode failed" in err and "groq" in err:
        return "Install missing dependencies in your venv (for example groq/chromadb) or run in direct-mocked mode."
    if "which department are you in" in str(result.get("checks", "")).lower() and "followup_needed" not in answer:
        return "Review strict department inference in planning flow; bot should ask department instead of assuming one."
    if "which year and semester" in str(result.get("checks", "")).lower() and "followup_needed" not in answer:
        return "Review year/term extraction and follow-up branching for planning intent."
    if result.get("category") == "refusal" and "ieu database" not in answer and "i don't have" not in answer:
        return "Strengthen out-of-scope refusal guardrail to keep answers grounded in IEU data only."
    if result.get("category") == "planning":
        return "Check curriculum lookup for this department/year/term in curriculum.json and extract_year_and_term parsing."
    if result.get("category") == "graduation":
        return "Validate dept_info.json indexing and graduation-context retrieval path."
    if result.get("category") == "regulations":
        return "Validate regulations context injection and GPA/credit keyword triggers."
    return "Inspect this case output and adjust check expectation or response logic accordingly."


def print_failed_only(results):
    failed = [r for r in results if not r.get("passed")]
    print("\n" + "=" * 60)
    print("Failed Cases With Suggestions")
    print("=" * 60)
    if not failed:
        print("No failed cases.")
        return

    for r in failed:
        print(f"- {_safe_text(r.get('id'))} [{_safe_text(r.get('category','uncategorized'))}]")
        if r.get("error"):
            print(f"  Error: {_safe_text(r['error'])}")
        print(f"  Answer: {_safe_text((r.get('answer') or '').replace(chr(10), ' | '))}")
        for c in r.get("checks", []):
            if not c.get("ok"):
                print(f"  Failed check: {_safe_text(c.get('description'))}")
        print(f"  Suggestion: {_safe_text(suggest_fix(r))}")


def write_reports(results, summary, mode, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = Path(out_dir) / f"eval_report_{mode}_{ts}.json"
    md_path = Path(out_dir) / f"eval_report_{mode}_{ts}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "summary": summary,
        "results": results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"# EkoBot Evaluation Report ({mode})",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Total tests: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']}%",
        "",
        "## Category Breakdown"
    ]

    for cat, stat in sorted(summary["by_category"].items()):
        rate = round((stat["passed"] / stat["total"]) * 100, 2) if stat["total"] else 0.0
        lines.append(f"- {cat}: {stat['passed']}/{stat['total']} ({rate}%)")

    lines.append("")
    lines.append("## Failed Cases")

    failed_rows = [r for r in results if not r["passed"]]
    if not failed_rows:
        lines.append("- None")
    else:
        for r in failed_rows:
            lines.append(f"- {r['id']} ({r.get('category','uncategorized')})")
            if r.get("error"):
                lines.append(f"  - Error: {r['error']}")
            lines.append(f"  - Answer: {r.get('answer','').replace(chr(10), ' | ')}")
            for c in r.get("checks", []):
                status = "PASS" if c.get("ok") else "FAIL"
                lines.append(f"  - {status}: {c.get('description')}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(json_path), str(md_path)


def main():
    parser = argparse.ArgumentParser(description="Automated evaluator for EkoBot.")
    parser.add_argument("--cases", default="eval_questions.json", help="Path to test cases JSON")
    parser.add_argument("--mode", choices=["api", "direct", "direct-mocked"], default="direct-mocked")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/chat", help="Chat endpoint (api mode)")
    parser.add_argument("--out", default="reports", help="Report output directory")
    parser.add_argument("--min-pass-rate", type=float, default=95.0, help="Quality gate threshold (percent)")
    parser.add_argument("--target-cases", type=int, default=60, help="Auto-augment test set to at least this many cases")
    parser.add_argument("--failed-only", action="store_true", help="Print only failed cases with suggestions")
    args = parser.parse_args()

    cases = load_cases(args.cases)

    # Always add broad failed-course robustness checks derived from curriculum
    # so coverage scales across departments without manually authored scenarios.
    existing_ids = {c.get("id") for c in cases}
    for c in auto_generate_failed_course_cases(limit_per_department=1):
        if c.get("id") not in existing_ids:
            cases.append(c)
            existing_ids.add(c.get("id"))

    if len(cases) < args.target_cases:
        needed = args.target_cases - len(cases)
        for c in auto_generate_cases(target_count=max(args.target_cases * 2, needed + 10)):
            if c.get("id") not in existing_ids:
                cases.append(c)
                existing_ids.add(c.get("id"))
            if len(cases) >= args.target_cases:
                break

    results = [evaluate_case(case, args.mode, args.api_url) for case in cases]
    summary = summarize(results)
    json_path, md_path = write_reports(results, summary, args.mode, args.out)

    print("=" * 60)
    print("EkoBot Auto Evaluation")
    print("=" * 60)
    print(f"Mode      : {args.mode}")
    print(f"Cases     : {summary['total']}")
    print(f"Passed    : {summary['passed']}")
    print(f"Failed    : {summary['failed']}")
    print(f"Pass Rate : {summary['pass_rate']}%")
    print(f"Gate      : {'PASS' if summary['pass_rate'] >= args.min_pass_rate else 'FAIL'} (threshold {args.min_pass_rate}%)")
    print(f"JSON      : {json_path}")
    print(f"Markdown  : {md_path}")

    if args.failed_only or summary["failed"] > 0:
        print_failed_only(results)

    # CI-friendly quality gate
    if summary["pass_rate"] < args.min_pass_rate:
        sys.exit(1)


if __name__ == "__main__":
    main()
