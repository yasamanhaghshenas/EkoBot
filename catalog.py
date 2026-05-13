import json
import os
import re
from functools import lru_cache
from urllib.parse import urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_COURSE_ECTS_CACHE = None
_KNOWN_DEPARTMENT_KEYS_CACHE = None
_PROGRAM_CATALOG_CACHE = None


def _normalize_lookup_key(value):
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _first_url(value):
    if not value:
        return None
    match = re.search(r"https?://[^\s\"']+", str(value))
    if not match:
        return None
    return match.group(0).rstrip(").,;\"'")


def _extract_code_tokens(text):
    tokens = []
    if not text:
        return tokens
    for raw in re.findall(r"[A-Z]{2,6}", str(text).upper()):
        if raw in {"NA", "N", "A", "ETC"}:
            continue
        if raw not in tokens:
            tokens.append(raw)
    return tokens


def _strip_parenthetical(label):
    return re.sub(r"\s*\([^)]*\)", "", (label or "")).strip()


def _unique_keep_order(values):
    seen = set()
    out = []
    for value in values or []:
        text = (value or "").strip()
        if not text:
            continue
        key = _normalize_lookup_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _alias_matches_lookup_key(alias_key, text_key):
    if not alias_key or not text_key:
        return False
    if alias_key == text_key:
        return True
    pattern = rf"\b{re.escape(alias_key)}\b"
    return re.search(pattern, text_key) is not None


@lru_cache(maxsize=1)
def load_ieu_url_catalog(path="ieu_url_catalog.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [item for item in data if isinstance(item, dict)]
    except Exception as e:
        print(f"  ieu_url_catalog.json read error: {e}")
        return []


@lru_cache(maxsize=1)
def load_program_catalog(path="program_catalog.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [item for item in data if isinstance(item, dict)]
    except Exception as e:
        print(f"  program_catalog.json read error: {e}")
        return []


@lru_cache(maxsize=1)
def _load_known_department_keys():
    global _KNOWN_DEPARTMENT_KEYS_CACHE
    if _KNOWN_DEPARTMENT_KEYS_CACHE is not None:
        return _KNOWN_DEPARTMENT_KEYS_CACHE

    keys = set()
    for fname in ["curriculum.json", "dept_info.json", "courses.json"]:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                if not isinstance(item, dict):
                    continue
                dept = item.get("department")
                if dept:
                    keys.add(_normalize_lookup_key(dept))
        except Exception as e:
            print(f"  {fname} department-key load error: {e}")

    _KNOWN_DEPARTMENT_KEYS_CACHE = keys
    return keys


def _pick_first_field_url(row, field_names):
    for field in field_names:
        value = row.get(field)
        url = _first_url(value)
        if url:
            return url
        if isinstance(value, str) and value.strip().startswith("http"):
            return value.strip()
    return None


def _normalize_profile_sheet_label(sheet):
    raw = (sheet or "").strip()
    key = _normalize_lookup_key(raw)
    if key == "facylty of business":
        return "Faculty of Business"
    if key == "schools vocational schools":
        return "Schools & Vocational Schools"
    return raw


def _should_include_profile_sheet(sheet, include_common=False, include_graduate=False):
    key = _normalize_lookup_key(sheet)
    if not include_common and key == "common courses (gec)":
        return False
    if not include_graduate and key == "graduate school":
        return False
    return True


def _faculty_aliases(sheet_label):
    label = _normalize_profile_sheet_label(sheet_label)
    aliases = [label]

    raw = (sheet_label or "").strip()
    if raw and raw not in aliases:
        aliases.append(raw)

    lower = label.lower()
    if lower.startswith("faculty of "):
        short = label[len("Faculty of "):].strip()
        if short and short not in aliases:
            aliases.append(short)
        if short.lower() not in {"arts and sciences", "fine arts and design", "health sciences"}:
            short_lower = short.lower()
            if short_lower and short_lower not in aliases:
                aliases.append(short_lower)

    if lower == "schools & vocational schools":
        for candidate in [
            "Schools and Vocational Schools",
            "Vocational Schools",
            "Vocational School",
            "Schools",
            "School",
        ]:
            if candidate not in aliases:
                aliases.append(candidate)

    return _unique_keep_order(aliases)


def _department_aliases(name, department_code=None, prefixes=None):
    aliases = []
    for candidate in [
        name,
        _strip_parenthetical(name),
        department_code,
    ]:
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    for prefix in prefixes or []:
        prefix = (prefix or "").strip().upper()
        if prefix and prefix not in aliases:
            aliases.append(prefix)

    return _unique_keep_order(aliases)


def _extract_year_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _derive_curriculum_url_label(url, fallback=None):
    raw = (url or "").strip()
    if not raw:
        return fallback or "Curriculum"

    parsed = urlparse(raw)
    parts = [part for part in parsed.path.split("/") if part]
    ignored = {
        "en", "tr", "curr", "curriculum", "syllabus", "type", "read", "id", "new",
    }

    slug = None
    for part in reversed(parts):
        part_lower = part.strip().lower()
        if not part_lower or part_lower in ignored:
            continue
        slug = part
        break

    if not slug:
        slug = parts[-1] if parts else raw

    slug = re.sub(r"[_-]+", " ", slug).strip()
    slug = re.sub(r"\s+", " ", slug)
    if not slug:
        return fallback or raw

    label = slug.title()
    label = label.replace("Ieu ", "IEU ")
    label = label.replace("Iue ", "IEU ")
    return label


def _format_curriculum_variant_label(start_year=None, end_year=None, open_ended=False, fallback=None):
    if start_year is None and end_year is None:
        return fallback or "Curriculum"
    if start_year is None:
        return str(end_year)
    if open_ended:
        return f"{start_year}+"
    if end_year is None or int(end_year) == int(start_year):
        return str(start_year)
    return f"{start_year}-{end_year}"


_FACULTY_LABEL_TR = {
    "Common Courses (GEC)": "Ortak Dersler (GEC)",
    "Faculty of Arts and Sciences": "Fen-Edebiyat Fakültesi",
    "Faculty of Communication": "İletişim Fakültesi",
    "Faculty of Engineering": "Mühendislik Fakültesi",
    "Faculty of Fine Arts and Design": "Güzel Sanatlar ve Tasarım Fakültesi",
    "Faculty of Health Sciences": "Sağlık Bilimleri Fakültesi",
    "Faculty of Law": "Hukuk Fakültesi",
    "Faculty of Medicine": "Tıp Fakültesi",
    "Faculty of Business": "İşletme Fakültesi",
    "Graduate School": "Lisansüstü Okulu",
    "Schools & Vocational Schools": "Okullar ve Meslek Yüksekokulları",
}

_DEPARTMENT_LABEL_TR = {
    "Accounting and Auditing Program": "Muhasebe ve Denetim Programı",
    "Aerospace Engineering": "Havacılık ve Uzay Mühendisliği",
    "Applied English Translation": "Uygulamalı İngilizce Çevirmenlik",
    "Architectural Restoration": "Mimari Restorasyon",
    "Architecture": "Mimarlık",
    "Atatürk / History": "Atatürk / Tarih",
    "Banking and Insurance": "Bankacılık ve Sigortacılık",
    "Biomedical Engineering": "Biyomedikal Mühendisliği",
    "Business Administration": "İşletme",
    "Child Development": "Çocuk Gelişimi",
    "Cinema and Digital Media": "Sinema ve Dijital Medya",
    "Civil Aviation Cabin Services": "Sivil Havacılık Kabin Hizmetleri",
    "Civil Aviation Transportation Management": "Sivil Havacılık Ulaştırma İşletmeciliği",
    "Civil Engineering": "İnşaat Mühendisliği",
    "Computer Engineering": "Bilgisayar Mühendisliği",
    "Computer Programming": "Bilgisayar Programcılığı",
    "Construction Technology": "İnşaat Teknolojisi",
    "Economics": "Ekonomi",
    "Elderly Care": "Yaşlı Bakımı",
    "Electrical and Electronics Engineering": "Elektrik-Elektronik Mühendisliği",
    "English Prep + Language Programs": "İngilizce Hazırlık + Dil Programları",
    "English Translation and Interpreting": "İngilizce Mütercim ve Tercümanlık",
    "Food Engineering": "Gıda Mühendisliği",
    "Foreign Trade": "Dış Ticaret",
    "Gastronomy and Culinary Arts": "Gastronomi ve Mutfak Sanatları",
    "GEC Pool 1": "GEC Havuzu 1",
    "GEC Pool 2": "GEC Havuzu 2",
    "GEC Pool 3": "GEC Havuzu 3",
    "GEC Pool 4": "GEC Havuzu 4",
    "GEC Pool 5": "GEC Havuzu 5",
    "GEC Pool 6": "GEC Havuzu 6",
    "GEC Pool 7": "GEC Havuzu 7",
    "Genetics and Bioengineering": "Genetik ve Biyomühendislik",
    "Graphic Design": "Grafik Tasarımı",
    "Industrial Design": "Endüstriyel Tasarım",
    "Industrial Engineering": "Endüstri Mühendisliği",
    "Interior Architecture and Environmental Design": "İç Mimarlık ve Çevre Tasarımı",
    "Interior Design": "İç Mekân Tasarımı",
    "International Trade and Finance": "Uluslararası Ticaret ve Finansman",
    "Justice": "Adalet",
    "Law": "Hukuk",
    "Legal Office Management and Secretariat": "Adalet Büro Yönetimi ve Sekreterliği",
    "Logistics Management": "Lojistik Yönetimi",
    "Mathematics": "Matematik",
    "Mechanical Engineering": "Makine Mühendisliği",
    "Mechatronics Engineering": "Mekatronik Mühendisliği",
    "Media and Communication": "Medya ve İletişim",
    "Medical Documentation and Secretariat": "Tıbbi Dokümantasyon ve Sekreterlik",
    "Medical Imaging Techniques": "Tıbbi Görüntüleme Teknikleri",
    "Medical Laboratory Techniques": "Tıbbi Laboratuvar Teknikleri",
    "Medicine": "Tıp",
    "New Media and Communication": "Yeni Medya ve İletişim",
    "Nursing": "Hemşirelik",
    "Nutrition and Dietetics": "Beslenme ve Diyetetik",
    "Occupational Health and Safety": "İş Sağlığı ve Güvenliği",
    "Opticianry": "Optisyenlik",
    "Orientation": "Oryantasyon",
    "Paramedic": "Paramedik",
    "Physics": "Fizik",
    "Physiotherapy": "Fizyoterapi",
    "Physiotherapy and Rehabilitation": "Fizyoterapi ve Rehabilitasyon",
    "Political Science and International Relations": "Siyaset Bilimi ve Uluslararası İlişkiler",
    "Psychology": "Psikoloji",
    "Public Relations and Advertising": "Halkla İlişkiler ve Reklamcılık",
    "Radio and TV Programming": "Radyo ve Televizyon Programcılığı",
    "Sociology": "Sosyoloji",
    "Software Engineering": "Yazılım Mühendisliği",
    "Textile and Fashion Design": "Tekstil ve Moda Tasarımı",
    "Tourism and Hotel Management": "Turizm ve Otel İşletmeciliği",
    "Turkish Language": "Türk Dili",
    "Visual Communication Design": "Görsel İletişim Tasarımı",
}


def _localize_profile_option_label(label, prefer_tr=False):
    raw = (label or "").strip()
    if not raw or not prefer_tr:
        return raw

    base = re.sub(r"\s*\(Turkish\)$", "", raw).strip()
    return _FACULTY_LABEL_TR.get(base) or _DEPARTMENT_LABEL_TR.get(base) or base


def _build_curriculum_variants(rows):
    grouped = {}
    order = []

    for idx, row in enumerate(rows or []):
        url = (row.get("curriculum_url") or row.get("url") or "").strip()
        entry_year = _extract_year_int(row.get("entry_year"))
        if url:
            group_key = _normalize_lookup_key(url)
        else:
            group_key = f"entry_year:{entry_year if entry_year is not None else idx}"

        group = grouped.get(group_key)
        if group is None:
            group = {
                "curriculum_url": url or None,
                "entry_years": [],
                "rows": [],
                "order": idx,
                "label": None,
                "value": None,
                "aliases": [],
                "start_year": None,
                "end_year": None,
            }
            grouped[group_key] = group
            order.append(group_key)

        group["rows"].append(row)
        if url and not group["curriculum_url"]:
            group["curriculum_url"] = url
        if entry_year is not None:
            group["entry_years"].append(entry_year)

    groups = []
    for key in order:
        group = grouped[key]
        years = sorted(set(group["entry_years"]))
        group["entry_years"] = years
        if years:
            group["start_year"] = years[0]
            group["end_year"] = years[-1]
        groups.append(group)

    year_groups = [group for group in groups if group.get("start_year") is not None]
    year_groups.sort(key=lambda g: (g["start_year"], g["end_year"] or g["start_year"], g["order"]))
    other_groups = [group for group in groups if group.get("start_year") is None]
    other_groups.sort(key=lambda g: g["order"])

    ordered_groups = year_groups + other_groups

    for idx, group in enumerate(ordered_groups):
        years = group["entry_years"]
        start_year = group.get("start_year")
        end_year = group.get("end_year")
        open_ended = bool(year_groups) and idx == len(year_groups) - 1 and start_year is not None

        if years:
            label = _format_curriculum_variant_label(start_year, end_year, open_ended=open_ended)
        else:
            label = _derive_curriculum_url_label(group.get("curriculum_url"), fallback=fallback if (fallback := group.get("label")) else None)
        if not label:
            label = group.get("curriculum_url") or "Curriculum"

        group["label"] = label
        group["value"] = label
        aliases = [label]
        if group.get("curriculum_url"):
            aliases.append(group["curriculum_url"])
        if start_year is not None:
            aliases.append(str(start_year))
            aliases.append(f"{start_year}+")
        if start_year is not None and end_year is not None:
            aliases.append(f"{start_year}-{end_year}")
            aliases.append(f"{start_year} to {end_year}")
        if end_year is not None:
            aliases.append(str(end_year))
        group["aliases"] = _unique_keep_order(aliases)

    return ordered_groups


@lru_cache(maxsize=1)
def load_ieu_profile_hierarchy(
    path="program_catalog.json",
    include_common=False,
    include_graduate=False,
):
    rows = load_program_catalog(path)
    if not rows:
        return {
            "faculties": [],
            "faculty_lookup": {},
            "department_lookup": {},
            "department_code_lookup": {},
        }

    faculty_order = []
    faculty_map = {}
    department_lookup = {}
    department_code_lookup = {}

    for row in rows:
        sheet = (row.get("sheet") or "").strip()
        if not sheet or not _should_include_profile_sheet(sheet, include_common=include_common, include_graduate=include_graduate):
            continue

        department_name = (row.get("department_name") or row.get("program_name") or row.get("department") or "").strip()
        if not department_name or department_name in {"?", "NA"}:
            continue

        faculty_label = _normalize_profile_sheet_label(sheet)
        faculty_key = _normalize_lookup_key(faculty_label)
        faculty = faculty_map.get(faculty_key)
        if faculty is None:
            faculty = {
                "sheet": sheet,
                "label": faculty_label,
                "aliases": _faculty_aliases(sheet),
                "departments": [],
                "_department_map": {},
            }
            faculty_map[faculty_key] = faculty
            faculty_order.append(faculty_key)

        dept_key = _normalize_lookup_key(department_name)
        dept = faculty["_department_map"].get(dept_key)
        if dept is None:
            dept = {
                "label": department_name,
                "value": department_name,
                "program_name": department_name,
                "department_code": None,
                "curriculum_url": None,
                "curriculum_urls": [],
                "syllabus_pattern": None,
                "entry_years": [],
                "prefixes": [],
                "aliases": [],
                "rows": [],
                "faculty_sheet": sheet,
                "faculty_label": faculty_label,
            }
            faculty["_department_map"][dept_key] = dept
            faculty["departments"].append(dept)

        dept["rows"].append(row)

        department_code = (row.get("department_code") or "").strip().upper()
        if not department_code:
            prefixes = [p.strip().upper() for p in (row.get("prefixes") or []) if (p or "").strip()]
            if prefixes:
                department_code = prefixes[0]
        if department_code and not dept["department_code"]:
            dept["department_code"] = department_code

        curriculum_url = (row.get("curriculum_url") or "").strip()
        if curriculum_url:
            dept["curriculum_urls"].append(curriculum_url)
            if not dept["curriculum_url"]:
                dept["curriculum_url"] = curriculum_url

        syllabus_pattern = (row.get("syllabus_pattern") or "").strip()
        if syllabus_pattern and not dept["syllabus_pattern"]:
            dept["syllabus_pattern"] = syllabus_pattern

        entry_year = row.get("entry_year")
        if entry_year is not None:
            try:
                year_int = int(entry_year)
                dept["entry_years"].append(year_int)
            except Exception:
                pass

        for prefix in row.get("prefixes") or []:
            prefix = (prefix or "").strip().upper()
            if prefix and prefix not in dept["prefixes"]:
                dept["prefixes"].append(prefix)

        dept["aliases"] = _unique_keep_order(
            dept["aliases"]
            + _department_aliases(
                department_name,
                dept["department_code"],
                dept["prefixes"],
            )
        )

    faculties = []
    faculty_lookup = {}
    for faculty_key in faculty_order:
        faculty = faculty_map[faculty_key]
        departments = []
        for dept in faculty["departments"]:
            dept["entry_years"] = sorted(set(dept["entry_years"]))
            dept["curriculum_urls"] = _unique_keep_order(dept["curriculum_urls"])
            dept["prefixes"] = _unique_keep_order(dept["prefixes"])
            dept["aliases"] = _unique_keep_order(dept["aliases"])
            dept.pop("_department_map", None)

            departments.append(dept)

            for alias in dept["aliases"] + [dept["label"], dept["value"], dept["program_name"], dept["department_code"]]:
                if not alias:
                    continue
                department_lookup[_normalize_lookup_key(alias)] = dept

            if dept["department_code"]:
                department_code_lookup.setdefault(dept["department_code"], []).append(dept)

        faculty["departments"] = departments
        faculties.append(faculty)

        for alias in faculty["aliases"] + [faculty["label"], faculty["sheet"]]:
            if not alias:
                continue
            faculty_lookup[_normalize_lookup_key(alias)] = faculty

    return {
        "faculties": faculties,
        "faculty_lookup": faculty_lookup,
        "department_lookup": department_lookup,
        "department_code_lookup": department_code_lookup,
    }


def get_ieu_faculty_options(include_common=False, include_graduate=False, prefer_tr=False):
    hierarchy = load_ieu_profile_hierarchy(include_common=include_common, include_graduate=include_graduate)
    return [
        {
            "label": _localize_profile_option_label(faculty["label"], prefer_tr=prefer_tr),
            "value": faculty["label"],
        }
        for faculty in hierarchy.get("faculties", [])
    ]


def get_ieu_department_options(faculty_name, include_common=False, include_graduate=False, prefer_tr=False):
    hierarchy = load_ieu_profile_hierarchy(include_common=include_common, include_graduate=include_graduate)
    faculty = resolve_ieu_faculty(faculty_name, include_common=include_common, include_graduate=include_graduate)
    if not faculty:
        return []
    return [
        {
            "label": _localize_profile_option_label(dept["label"], prefer_tr=prefer_tr),
            "value": dept["label"],
        }
        for dept in faculty.get("departments", [])
    ]


def resolve_ieu_faculty(text, include_common=False, include_graduate=False):
    hierarchy = load_ieu_profile_hierarchy(include_common=include_common, include_graduate=include_graduate)
    if not text:
        return None

    key = _normalize_lookup_key(text)
    faculty = hierarchy.get("faculty_lookup", {}).get(key)
    if faculty:
        return faculty

    for candidate in hierarchy.get("faculties", []):
        for alias in candidate.get("aliases") or []:
            alias_key = _normalize_lookup_key(alias)
            if _alias_matches_lookup_key(alias_key, key):
                return candidate
    return None


def resolve_ieu_department(text, faculty_name=None, include_common=False, include_graduate=False):
    hierarchy = load_ieu_profile_hierarchy(include_common=include_common, include_graduate=include_graduate)
    if not text:
        return None

    key = _normalize_lookup_key(text)
    faculty = resolve_ieu_faculty(faculty_name, include_common=include_common, include_graduate=include_graduate) if faculty_name else None

    search_space = []
    if faculty:
        search_space.extend(faculty.get("departments", []))
    else:
        for fac in hierarchy.get("faculties", []):
            search_space.extend(fac.get("departments", []))

    code_key = (text or "").strip().upper()
    if code_key and code_key in hierarchy.get("department_code_lookup", {}):
        matches = hierarchy["department_code_lookup"].get(code_key) or []
        if faculty:
            matches = [
                dept for dept in matches
                if _normalize_lookup_key(dept.get("faculty_label")) == _normalize_lookup_key(faculty.get("label"))
            ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None

    if key in hierarchy.get("department_lookup", {}):
        dept = hierarchy["department_lookup"][key]
        if faculty and _normalize_lookup_key(dept.get("faculty_label")) != _normalize_lookup_key(faculty.get("label")):
            return None
        return dept

    for dept in search_space:
        for alias in dept.get("aliases") or []:
            alias_key = _normalize_lookup_key(alias)
            if _alias_matches_lookup_key(alias_key, key):
                return dept
    return None


def _derive_syllabus_pattern(row):
    url = _pick_first_field_url(
        row,
        [
            "syllabus_url_dept_subdomain_2025",
            "syllabus_url_ects_old_curricula",
            "syllabus_url",
        ],
    )
    if not url:
        return None
    pattern = re.sub(r"/id/[A-Z]{2,6}\+\d{3,4}", "/id/{CODE}+{NUM}", url)
    pattern = re.sub(r"course_code=[A-Z]{2,6}\+\d{3,4}", "course_code={CODE}+{NUM}", pattern)
    return pattern


def _extract_row_code_tokens(row):
    tokens = []
    prefix = row.get("course_code_prefix") or ""
    for token in _extract_code_tokens(prefix):
        if token not in tokens:
            tokens.append(token)

    for key, value in row.items():
        if not isinstance(value, str):
            continue
        if "syllabus_url" in key or "curriculum_url" in key:
            for token in re.findall(r"/id/([A-Z]{2,6})\+\d{3,4}", value.upper()):
                if token not in tokens:
                    tokens.append(token)
            for token in re.findall(r"\b([A-Z]{2,6})\+\d{3,4}\b", value.upper()):
                if token not in tokens:
                    tokens.append(token)

    return tokens


def _infer_department_canonical(row, known_keys):
    source_label = _strip_parenthetical(row.get("department") or row.get("program_name") or row.get("name"))
    program_name = _strip_parenthetical(row.get("program_name") or "")
    code_tokens = _extract_row_code_tokens(row)

    candidates = [source_label, program_name] + code_tokens
    for candidate in candidates:
        if not candidate:
            continue
        if _normalize_lookup_key(candidate) in known_keys:
            return candidate.strip()

    if code_tokens:
        return code_tokens[0]
    if source_label:
        return source_label
    return program_name or None


@lru_cache(maxsize=1)
def load_ieu_department_directory(path="ieu_url_catalog.json"):
    rows = load_ieu_url_catalog(path)
    if not rows:
        return []

    known_keys = _load_known_department_keys()
    directory = []
    seen = set()

    for row in rows:
        source_label = _strip_parenthetical(row.get("department") or row.get("program_name") or row.get("name"))
        if not source_label:
            continue

        canonical = _infer_department_canonical(row, known_keys) or source_label
        curriculum_url = _pick_first_field_url(
            row,
            [
                "curriculum_url_2025_present_ects_sid_curr",
                "curriculum_url_2020_2024",
                "curriculum_url_never_changed",
                "curriculum_url_2013_2019_ects_sid_curr_before2020",
                "curriculum_url_2013_2014_new_system",
                "curriculum_url_2012_2013_old_system",
                "curriculum_url",
            ],
        )
        syllabus_pattern = _derive_syllabus_pattern(row)
        code_tokens = _extract_row_code_tokens(row)

        aliases = []
        for candidate in [source_label, _strip_parenthetical(source_label), row.get("program_name")]:
            if candidate:
                candidate = candidate.strip()
                if candidate and candidate not in aliases:
                    aliases.append(candidate)
        for token in code_tokens:
            if token not in aliases:
                aliases.append(token)

        key = (
            _normalize_lookup_key(canonical),
            _normalize_lookup_key(curriculum_url or ""),
            _normalize_lookup_key(syllabus_pattern or ""),
        )
        if key in seen:
            continue
        seen.add(key)

        directory.append(
            {
                "canonical_department": canonical,
                "source_department": source_label,
                "aliases": aliases,
                "code_tokens": code_tokens,
                "curriculum_url": curriculum_url,
                "syllabus_pattern": syllabus_pattern,
                "source_row": row,
            }
        )

    return directory


@lru_cache(maxsize=1)
def load_web_page_cache(path="ecobot_rules_pages.json"):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        cache = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            text = (item.get("text") or item.get("content") or item.get("page_text") or "").strip()
            if url:
                cache[url] = text
        return cache
    except Exception as e:
        print(f"  web page cache read error: {e}")
        return {}


def get_graduation_text(dept_code, collection=None):
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

    # Local fallback for offline use (kept for tooling/debug, not runtime).
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

    return None


def _matches_curriculum_selector(item, dept_code=None, program_name=None, curriculum_url=None):
    if dept_code is not None:
        item_dept = (item.get("department") or "").strip().upper()
        if item_dept != (dept_code or "").strip().upper():
            return False

    if program_name is not None:
        item_program = _normalize_lookup_key(item.get("program_name"))
        if item_program != _normalize_lookup_key(program_name):
            return False

    if curriculum_url is not None:
        item_url = _normalize_lookup_key(item.get("url"))
        if item_url != _normalize_lookup_key(curriculum_url):
            return False

    return True


def get_available_entry_years(dept_code, program_name=None, curriculum_url=None):
    if not os.path.exists("curriculum.json"):
        return []
    years = set()
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if not _matches_curriculum_selector(
                item,
                dept_code=dept_code,
                program_name=program_name,
                curriculum_url=curriculum_url,
            ):
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


@lru_cache(maxsize=1)
def _load_curriculum_rows(path="curriculum.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  curriculum.json read error: {e}")
        return []


def _get_curriculum_variant_rows(dept_code, program_name=None):
    rows = []
    for item in _load_curriculum_rows():
        if _matches_curriculum_selector(
            item,
            dept_code=dept_code,
            program_name=program_name,
            curriculum_url=None,
        ):
            rows.append(item)
    return rows


def get_curriculum_variant_options(dept_code, program_name=None):
    """
    Return the enrollment-year choices that should be shown to the user.

    We only ask for an enrollment year when there is more than one
    year-based curriculum group for the selected department/program.
    The returned option values are the canonical start years for each group.
    """
    rows = _get_curriculum_variant_rows(dept_code, program_name=program_name)
    if not rows:
        return []

    variants = _build_curriculum_variants(rows)
    year_variants = [variant for variant in variants if variant.get("entry_years")]
    if len(year_variants) <= 1:
        return []

    options = []
    for variant in year_variants:
        start_year = variant.get("start_year")
        if start_year is None:
            continue
        options.append({
            "label": variant.get("label") or str(start_year),
            "value": str(start_year),
        })

    return options


def has_multiple_curriculum_variants(dept_code, program_name=None, curriculum_url=None):
    """
    Returns True when a department/program has more than one year-based
    curriculum variant. This is used to decide whether to ask for the
    student's enrollment year.
    """
    return len(get_curriculum_variant_options(dept_code, program_name=program_name)) > 0


def get_curriculum_courses(dept_code, year, term, start_year=None, program_name=None, curriculum_url=None):
    if not os.path.exists("curriculum.json"):
        return []
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        matches = []
        for item in data:
            selector_url = None if start_year is not None else curriculum_url
            if not _matches_curriculum_selector(
                item,
                dept_code=dept_code,
                program_name=program_name,
                curriculum_url=selector_url,
            ):
                continue

            item_entry_year = item.get("entry_year")
            for sem_name, courses in item.get("semesters", {}).items():
                name = sem_name.lower()
                if str(year) in name and term.lower() in name:
                    if isinstance(courses, list):
                        matches.append((item_entry_year, courses))

        if not matches:
            return []

        if start_year is not None:
            exact = []
            for item_entry_year, courses in matches:
                try:
                    if item_entry_year is not None and int(item_entry_year) == int(start_year):
                        exact.append(courses)
                except Exception:
                    continue
            if exact:
                return exact[0]

            numeric_matches = []
            for item_entry_year, courses in matches:
                try:
                    if item_entry_year is not None:
                        numeric_matches.append((int(item_entry_year), courses))
                except Exception:
                    continue
            if numeric_matches:
                numeric_matches.sort(key=lambda pair: abs(pair[0] - int(start_year)))
                return numeric_matches[0][1]

        if len(matches) == 1:
            return matches[0][1]

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


def _select_curriculum_row(dept_code, start_year=None, program_name=None, curriculum_url=None):
    if not os.path.exists("curriculum.json"):
        return None
    try:
        with open("curriculum.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  curriculum.json read error: {e}")
        return None

    rows = [
        item for item in data
        if _matches_curriculum_selector(
            item,
            dept_code=dept_code,
            program_name=program_name,
            curriculum_url=None if start_year is not None else curriculum_url,
        )
    ]
    if not rows:
        return None

    if start_year is not None:
        for item in rows:
            try:
                if item.get("entry_year") is not None and int(item.get("entry_year")) == int(start_year):
                    return item
            except Exception:
                continue

        numeric_rows = []
        for item in rows:
            try:
                if item.get("entry_year") is not None:
                    numeric_rows.append((int(item.get("entry_year")), item))
            except Exception:
                continue
        if numeric_rows:
            numeric_rows.sort(key=lambda pair: abs(pair[0] - int(start_year)))
            return numeric_rows[0][1]

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


def get_curriculum_course_positions(dept_code, start_year=None, program_name=None, curriculum_url=None):
    row = _select_curriculum_row(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
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


def get_electives(dept_code, start_year=None, program_name=None, curriculum_url=None):
    row = _select_curriculum_row(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
    if row:
        electives = row.get("electives", [])
        return electives if isinstance(electives, list) else []
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


@lru_cache(maxsize=1)
def load_web_sources(path="ecobot_rules_sources.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if title and url:
                out.append(item)
        return out
    except Exception as e:
        print(f"  web source catalog read error: {e}")
        return []


@lru_cache(maxsize=128)
def fetch_web_page_text(url):
    if not url:
        return None
    cache = load_web_page_cache()
    url_key = url.strip()
    if url_key in cache:
        cached = cache.get(url_key)
        return cached or None
    try:
        response = requests.get(
            url_key,
            timeout=15,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0 (compatible; EkoBot/1.0)"},
        )
        if not response.ok:
            return None
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        content = (
            soup.find("div", class_="col-md-9")
            or soup.find("div", class_="col-lg-9")
            or soup.select_one("main")
            or soup.select_one("div.container")
        )
        if not content:
            return None
        text = content.get_text(strip=True, separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text or None
    except Exception as e:
        print(f"  web page fetch error for {url}: {e}")
        return None


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def get_web_context(question, max_sources=3):
    """
    Pull relevant content from the workbook-driven web source catalog.
    Returns combined page text for the best matching pages, or None.
    """
    sources = load_web_sources()
    if not sources:
        return None

    page_cache = load_web_page_cache()
    q = (question or "").lower()
    q_tokens = [tok for tok in _tokenize(q) if len(tok) > 2]
    if not q_tokens:
        q_tokens = _tokenize(q)

    scored = []
    for item in sources:
        title = (item.get("title") or "").strip()
        desc = (item.get("description") or "").strip()
        url = (item.get("url") or "").strip()
        cached_text = page_cache.get(url, "") if url else ""
        hay = f"{title} {desc} {cached_text[:3000]}".lower()
        score = 0
        if title and title.lower() in q:
            score += 8
        if desc and desc.lower() in q:
            score += 3
        for tok in q_tokens:
            if tok in hay:
                score += 1
        if any(k in q for k in ["erasmus", "exchange", "mobility", "international"]):
            if any(k in hay for k in ["erasmus", "exchange", "mobility", "international"]):
                score += 2
        if "coordinator" in q and "coordinator" in hay:
            score += 6
        if "coordinators" in q and "coordinators" in hay:
            score += 6
        if "exchange program coordinators" in hay:
            score += 20
        if "who are" in q and "coordinator" in hay:
            score += 6
        if ("who are" in q or "who is" in q) and "coordinator" in hay:
            score += 3
        if any(k in q for k in ["student", "campus", "library", "blackboard", "obs", "dormitory", "shuttle"]):
            if any(k in hay for k in ["student", "campus", "library", "blackboard", "obs", "dormitory", "shuttle"]):
                score += 2
        if any(k in q for k in ["calendar", "leave", "document", "form", "transcript", "id card", "internship", "career"]):
            if any(k in hay for k in ["calendar", "leave", "document", "form", "transcript", "id card", "internship", "career"]):
                score += 2
        scored.append((score, item))

    scored.sort(key=lambda pair: (pair[0], pair[1].get("title") or ""), reverse=True)
    picked = [item for score, item in scored[:max_sources] if score > 0]
    if not picked:
        return None

    parts = []
    for item in picked:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        desc = (item.get("description") or "").strip()
        if url and url in page_cache:
            text = page_cache.get(url) or None
        else:
            text = None
        if text:
            parts.append(f"{title}\n{desc}\n{text}")
        else:
            parts.append(f"{title}\n{desc}\nURL: {url}")

    if not parts:
        return None

    combined = "\n\n---\n\n".join(parts)
    return combined[:8000].strip()
