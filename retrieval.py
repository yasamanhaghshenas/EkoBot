import re

from catalog import _normalize_course_code


def _looks_like_course_code(value):
    return bool(re.match(r"^[A-Z]{2,6}\s\d{3,4}$", (value or "").strip()))


def get_context_for_codes(collection_or_codes, codes=None):
    contexts = []
    missing = []
    corrections = {}

    if codes is None:
        codes = collection_or_codes
        try:
            from gemini import collection as chroma_collection
        except Exception:
            chroma_collection = None
    else:
        chroma_collection = collection_or_codes

    if chroma_collection is None:
        return contexts, [(_normalize_course_code(code) or code) for code in codes], corrections

    for code in codes:
        normalized = _normalize_course_code(code) or code
        found = False

        try:
            results = chroma_collection.get(
                where={"code": normalized},
                include=["documents", "metadatas"],
            )
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            if docs:
                contexts.append(docs[0])
                found = True
                if metas:
                    candidate_code = (metas[0] or {}).get("code") or ""
                    candidate_code = candidate_code.strip().upper()
                    if _looks_like_course_code(candidate_code) and candidate_code != normalized:
                        corrections[normalized] = candidate_code
        except Exception:
            pass

        try:
            results = chroma_collection.query(
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
                results = chroma_collection.query(
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

    def extract_evaluation_lines():
        match = re.search(r"^Evaluation System:\s*(.+?)(?:\n(?:Prerequisites|Course Code|Department|$):|\Z)", doc, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if not match:
            return []

        section = match.group(1)
        lines = []
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("evaluation system:"):
                continue
            if line.startswith("Prerequisites:"):
                continue
            if "time(s)" not in line and "% weight" not in line:
                continue
            lines.append(line)
        return lines

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

    evaluation_lines = extract_evaluation_lines()
    if evaluation_lines:
        lines.append("Evaluation System:")
        lines.extend(f"- {line}" for line in evaluation_lines)

    return "\n".join(lines)
