"""Deterministic handling of IEU GPA-based semester credit limits."""

import re


def is_gpa_credit_limit_question(question: str) -> bool:
    """Return True when a question combines GPA context with a semester-load query."""
    q = (question or "").casefold()
    has_gpa = bool(
        re.search(r"\b(?:cgpa|gpa|grade\s+point\s+average)\b|not\s+ortalamas", q)
    )
    has_load = bool(
        re.search(
            r"\b(?:credit(?:s)?|ects|course\s+load|semester\s+load|max(?:imum)?\s+load)\b|"
            r"kredi|ders\s+y[uü]k",
            q,
        )
    )
    return has_gpa and has_load


def extract_gpa_from_question(question: str):
    """Extract a GPA value from common '2.2 GPA' / 'GPA: 2.2' phrasings."""
    q = (question or "").casefold().replace(",", ".")
    below_match = re.search(
        r"(?:below|under|less\s+than|underneath|alt[ıi]nda)\s*(?:a\s+)?(2(?:\.0+)?)",
        q,
    )
    if below_match:
        return 1.99

    patterns = [
        r"\b(?:cgpa|gpa|grade\s+point\s+average)\s*(?:of|is|:|=)?\s*(\d(?:\.\d+)?)\b",
        r"\b(\d(?:\.\d+)?)\s*(?:cgpa|gpa)\b",
        r"not\s+ortalamas[ıi](?:\s*[:=]?\s*)(\d(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            try:
                value = float(match.group(1))
                return value if 0 <= value <= 4.0 else None
            except ValueError:
                return None
    return None


def get_credit_limit_by_gpa(gpa):
    """Return the maximum ECTS load for a GPA, or None for an invalid value."""
    try:
        value = float(gpa)
    except (TypeError, ValueError):
        return None
    if not 0 <= value <= 4.0:
        return None
    if value < 2.0:
        return 32
    if value < 2.5:
        return 38
    return 44
