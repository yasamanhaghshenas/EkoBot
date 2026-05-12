import re

from catalog import _normalize_course_code


def _looks_like_course_code(value):
    return bool(re.match(r"^[A-Z]{2,6}\s\d{3,4}$", (value or "").strip()))


def get_context_for_codes(collection, codes):
    contexts = []
    missing = []
    corrections = {}
    for code in codes:
        normalized = _normalize_course_code(code) or code
        found = False
        try:
            results = collection.query(
                query_texts=[normalized],
                n_results=1,
                where={"code": normalized},
                include=["documents", "metadatas"],
            )
            docs = (results.get("documents") or [[]])[0]
            if docs:
                contexts.append(docs[0])
                found = True
        except Exception:
            pass

        if not found:
            try:
                results = collection.query(
                    query_texts=[normalized],
                    n_results=1,
                    include=["documents", "metadatas"],
                )
                docs = (results.get("documents") or [[]])[0]
                metas = (results.get("metadatas") or [[]])[0]
                if docs and metas:
                    candidate_code = (metas[0] or {}).get("code") or ""
                    candidate_code = candidate_code.strip().upper()
                    if _looks_like_course_code(candidate_code) and candidate_code != normalized:
                        contexts.append(docs[0])
                        corrections[normalized] = candidate_code
                        found = True
            except Exception:
                pass

        if not found:
            missing.append(normalized)
    return contexts, missing, corrections


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
        eval_lines = [line.strip() for line in eval_text.split("\n") if line.strip()]
        eval_lines = [line for line in eval_lines if not line.startswith("time(s)") and "%" in line]
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
