import re

from catalog import _normalize_course_code


def get_context_for_codes(collection, codes):
    contexts = []
    missing = []
    for code in codes:
        normalized = _normalize_course_code(code) or code
        found = False
        try:
            results = collection.query(
                query_texts=[normalized],
                n_results=1,
                where={"code": normalized},
            )
            if results["documents"][0]:
                contexts.append(results["documents"][0][0])
                found = True
        except Exception:
            pass
        if not found:
            missing.append(normalized)
    return contexts, missing


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
    if desc:
        lines.append(f"Description: {desc}")

    return "\n".join(lines)
