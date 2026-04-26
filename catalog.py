import json
import os
import re
from urllib.parse import urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_COURSE_ECTS_CACHE = None


def get_graduation_text(dept_code, collection=None):
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
    if collection is not None:
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
    m = re.search(r"\b([A-Z]{2,6})\s*(\d{3,4})\b", t)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}"


def _semester_sort_index(year, term):
    t = (term or "").strip().lower()
    term_order = 0 if t.startswith("fall") else 1
    return int(year) * 2 + term_order


def _extract_year_term_from_semester_name(sem_name):
    s = (sem_name or "")
    y_match = re.search(r"([1-6])\s*\.\s*year|([1-6])\s*year", s, re.IGNORECASE)
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
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", text or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


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
        m = re.search(r"^ECTS:\s*(\d+(?:\.\d+)?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        try:
            ects_map[code] = float(m.group(1))
        except Exception:
            continue

    _COURSE_ECTS_CACHE = ects_map
    return ects_map


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
    except Exception:
        return None
