import json
import os
import re

from catalog import _normalize_course_code
from grading_helpers import (
    extract_evaluation_lines,
    extract_grading_only_response,
    is_grading_question,
)


# Chroma returns a distance score where:
# - lower distance = more similar (better match)
# - higher distance = less similar (worse match)
# Only reject results with very poor distance (> MIN_CONFIDENCE).
MIN_CONFIDENCE = 1.5
MAX_DISTANCE = MIN_CONFIDENCE

_COURSE_DOCS_BY_CODE_CACHE = None


def _looks_like_course_code(value):
    return bool(re.match(r"^[A-Z]{2,6}\s\d{3,4}$", (value or "").strip()))


def _load_course_docs_by_code():
    global _COURSE_DOCS_BY_CODE_CACHE
    if _COURSE_DOCS_BY_CODE_CACHE is not None:
        return _COURSE_DOCS_BY_CODE_CACHE

    docs_by_code = {}
    path = os.path.join(os.path.dirname(__file__), "courses.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            courses = json.load(f)
    except Exception:
        _COURSE_DOCS_BY_CODE_CACHE = docs_by_code
        return docs_by_code

    if not isinstance(courses, list):
        _COURSE_DOCS_BY_CODE_CACHE = docs_by_code
        return docs_by_code

    for item in courses:
        if not isinstance(item, dict):
            continue
        code = _normalize_course_code(item.get("code"))
        text = item.get("text") or ""
        if code and text:
            docs_by_code.setdefault(code, []).append(text)

    _COURSE_DOCS_BY_CODE_CACHE = docs_by_code
    return docs_by_code


def _get_exact_course_docs(code):
    normalized = _normalize_course_code(code) or code
    return list(_load_course_docs_by_code().get(normalized, []))


def get_context_for_codes(collection, codes):
    contexts = []
    missing = []
    corrections = {}
    low_confidence_score = None
    for code in codes:
        normalized = _normalize_course_code(code) or code
        found = False

        exact_docs = _get_exact_course_docs(normalized)
        if exact_docs:
            contexts.append(exact_docs[0])
            found = True

        if found:
            continue

        if collection is None:
            missing.append(normalized)
            continue

        try:
            results = collection.query(
                query_texts=[normalized],
                n_results=1,
                where={"code": normalized},
                include=["documents", "metadatas", "distances"],
            )
            docs = (results.get("documents") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            top_distance = distances[0] if distances else None
            if top_distance is not None:
                print(f"[RETRIEVAL] Top distance: {top_distance:.2f} — {'ACCEPTED' if top_distance < MIN_CONFIDENCE else 'REJECTED'}")
            if docs and (top_distance is None or top_distance <= MAX_DISTANCE):
                contexts.append(docs[0])
                found = True
            elif docs and top_distance is not None and top_distance > MAX_DISTANCE:
                print(f"[RETRIEVAL] Low confidence ({top_distance:.2f}), falling back")
                low_confidence_score = top_distance
        except Exception:
            pass

        if not found:
            try:
                results = collection.query(
                    query_texts=[normalized],
                    n_results=1,
                    include=["documents", "metadatas", "distances"],
                )
                docs = (results.get("documents") or [[]])[0]
                metas = (results.get("metadatas") or [[]])[0]
                distances = (results.get("distances") or [[]])[0]
                top_distance = distances[0] if distances else None
                if top_distance is not None:
                    print(f"[RETRIEVAL] Top distance: {top_distance:.2f} — {'ACCEPTED' if top_distance < MIN_CONFIDENCE else 'REJECTED'}")
                if docs and metas and (top_distance is None or top_distance <= MAX_DISTANCE):
                    candidate_code = (metas[0] or {}).get("code") or ""
                    candidate_code = candidate_code.strip().upper()
                    if _looks_like_course_code(candidate_code) and candidate_code != normalized:
                        contexts.append(docs[0])
                        corrections[normalized] = candidate_code
                        found = True
                elif docs and top_distance is not None and top_distance > MAX_DISTANCE:
                    print(f"[RETRIEVAL] Low confidence ({top_distance:.2f}), falling back")
                    low_confidence_score = top_distance
            except Exception:
                pass

        if not found:
            missing.append(normalized)
    return contexts, missing, corrections, low_confidence_score


def _pick_field(doc, labels):
    for label in labels:
        m = re.search(rf"^{re.escape(label)}:\s*(.+)$", doc, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def _course_header(doc):
    code = _pick_field(doc, ["Course Code"]) or "Unknown"
    name = _pick_field(doc, ["Course Name"]) or "Unknown"
    return f"{code} - {name}"


def _course_focus_from_question(question):
    q = (question or "").lower()
    if is_grading_question(q):
        return "grading"

    if any(cue in q for cue in [
        "who teaches", "who is teaching", "who will teach", "who taught",
        "instructor", "lecturer", "teacher",
    ]):
        return "teacher"

    if "coordinator" in q:
        return "coordinator"

    if any(cue in q for cue in ["prereq", "prerequisite", "prerequist", "prerequisit", "pre requisite"]):
        return "prerequisites"

    if any(cue in q for cue in ["credit", "credits", "ects"]):
        return "credits"

    if any(cue in q for cue in ["description", "course description", "course aim", "aim"]):
        return "description"

    return None


def _extract_focused_course_response(doc, focus):
    if not doc or not focus:
        return ""

    header = _course_header(doc)

    if focus == "teacher":
        teacher = _pick_field(doc, ["Course Lecturer(s)", "Course Lecturer", "Course Instructor(s)"])
        coordinator = _pick_field(doc, ["Course Coordinator"])
        lines = [header]
        if teacher:
            lines.append(f"Teacher/Lecturer(s): {teacher}")
        elif coordinator:
            lines.append(f"Course Coordinator: {coordinator}")
        else:
            return ""
        return "\n".join(lines)

    if focus == "coordinator":
        coordinator = _pick_field(doc, ["Course Coordinator"])
        teacher = _pick_field(doc, ["Course Lecturer(s)", "Course Lecturer", "Course Instructor(s)"])
        lines = [header]
        if coordinator:
            lines.append(f"Course Coordinator: {coordinator}")
        elif teacher:
            lines.append(f"Teacher/Lecturer(s): {teacher}")
        else:
            return ""
        return "\n".join(lines)

    if focus == "prerequisites":
        prereq = _pick_field(doc, ["Prerequisites"])
        if prereq is None:
            return ""
        return "\n".join([header, f"Prerequisites: {prereq}"])

    if focus == "credits":
        credits = _pick_field(doc, ["Credits"])
        ects = _pick_field(doc, ["ECTS"])
        lines = [header]
        if credits is not None:
            lines.append(f"Credits: {credits}")
        if ects is not None:
            lines.append(f"ECTS: {ects}")
        return "\n".join(lines) if len(lines) > 1 else ""

    if focus == "description":
        desc = _pick_field(doc, ["Course Description", "Course Aim"])
        if desc is None:
            return ""
        return "\n".join([header, f"Description: {desc}"])

    return ""


def extract_course_specific_response(course_contexts, question):
    """
    Return only the requested course section when the user asks for a
    specific detail such as teacher, prerequisites, credits, or grading.
    """
    if not course_contexts:
        return ""

    focus = _course_focus_from_question(question)
    if not focus:
        return ""

    if focus == "grading":
        return extract_grading_only_response(course_contexts)

    results = []
    for doc in course_contexts:
        focused = _extract_focused_course_response(doc, focus)
        if focused:
            results.append(focused)

    return "\n\n".join(results)


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
    lecturer = pick("Course Lecturer(s)") or pick("Course Lecturer") or pick("Course Instructor(s)")
    coordinator = pick("Course Coordinator")
    prereq = pick("Prerequisites")
    desc = pick("Course Description") or pick("Course Aim")

    lines = [f"{code} - {name}", f"Department: {dept}"]
    
    # Extract and format Evaluation System FIRST (prioritize to avoid truncation)
    # Fixed regex: use (?:(?=\nPrerequisites:)|\Z) to capture all lines until Prerequisites or end
    eval_match = re.search(r"^Evaluation System:\s*\n(.*?)(?:(?=\nPrerequisites:)|\Z)", doc, re.MULTILINE | re.DOTALL)
    eval_section = []
    if eval_match:
        eval_text = eval_match.group(1).strip()
        eval_lines = extract_evaluation_lines(eval_text)
        if eval_lines:
            eval_section.append("Evaluation System:")
            for line in eval_lines:
                eval_section.append(f"  - {line}")
    
    # Add basic course info
    if semester:
        lines.append(f"Semester: {semester}")
    if credits:
        lines.append(f"Credits: {credits}")
    if ects:
        lines.append(f"ECTS: {ects}")
    if lecturer:
        lines.append(f"Teacher/Lecturer(s): {lecturer}")
    if coordinator and coordinator != lecturer:
        lines.append(f"Course Coordinator: {coordinator}")
    if prereq:
        lines.append(f"Prerequisites: {prereq}")
    
    # Add evaluation system early (before long description)
    if eval_section:
        lines.append("")
        lines.extend(eval_section)
        lines.append("")
    
    if desc:
        lines.append(f"Description: {desc}")

    return "\n".join(lines)


def get_certificate_context(collection_obj, query_texts, n_results=3, max_docs=6):
    def _query_and_collect(texts):
        if not texts:
            return []
        try:
            results = collection_obj.query(query_texts=texts, n_results=n_results, include=["documents"])
        except Exception:
            return []

        docs = []
        for batch in results.get("documents") or []:
            for doc in batch or []:
                if not doc or doc in docs:
                    continue
                docs.append(doc)
                if len(docs) >= max_docs:
                    return docs
        return docs

    alt_queries = [
        "student certificate IEU",
        "diploma belgesi",
        "öğrenci belgesi nasıl alınır",
    ]

    merged = []
    for batch in (_query_and_collect(query_texts), _query_and_collect(alt_queries)):
        for doc in batch:
            if doc not in merged:
                merged.append(doc)
                if len(merged) >= max_docs:
                    return merged
    return merged
