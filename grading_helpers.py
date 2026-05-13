"""Helper functions for grading-specific question handling."""
import re


def extract_evaluation_lines(section_text):
    """Return only evaluation lines that contain a real numeric percentage."""
    if not section_text:
        return []

    lines = []
    for raw_line in section_text.split("\n"):
        line = (raw_line or "").strip()
        if not line:
            continue

        # Keep only rows that include a numeric percentage, and drop placeholder
        # rows such as "Participation: time(s), % weight".
        if not re.search(r"\b\d+(?:[.,]\d+)?\s*%", line):
            continue

        lines.append(line)
    return lines


def is_grading_question(question):
    """Detect if question is specifically about grading/evaluation/exam weights."""
    q = (question or "").lower().strip()
    grading_patterns = [
        r"\bgrading\b",
        r"\bgrading\s+system\b",
        r"\bgrade\b",
        r"\bgraded\b",
        r"\bevaluation\b",
        r"\bevaluation\s+system\b",
        r"\bassessment\b",
        r"\bassessment\s+(?:method|system)\b",
        r"\bgrade\s+breakdown\b",
        r"\bhow\s+(?:am|are|will)\s+i\s+be\s+graded\b",
        r"\bexam\s+weights?\b",
        r"\bmidterm\b",
        r"\bfinal\b",
        r"\bquiz(?:zes)?\b",
        r"\bweight(?:s)?\b",
        r"\bpercent(?:age)?\b",
        r"\bscore\b",
        r"\bmark\b",
        r"\bpoint(?:s)?\b",
        r"\brubric\b",
        r"\bcriteria\b",
    ]
    for pattern in grading_patterns:
        if re.search(pattern, q):
            return True
    return False


def extract_grading_only_response(course_contexts):
    """Extract ONLY the evaluation system from course context(s), no other info."""
    results = []

    for context in course_contexts:
        if not context:
            continue

        # Extract only the evaluation system block and stop before the next section.
        eval_match = re.search(
            r"^Evaluation System:\s*\n(.*?)(?=\n(?:Description|Prerequisites|Course Code|Course Name|Department|Semester|Credits|ECTS|Teacher/Lecturer\(s\)|Course Coordinator):|\Z)",
            context,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if eval_match:
            eval_section = eval_match.group(1).strip()
            if eval_section:
                lines = extract_evaluation_lines(eval_section)
                if lines:
                    response = ["Evaluation System:"]
                    response.extend(f"- {line}" for line in lines)
                    results.append("\n".join(response))

    return "\n\n".join(results) if results else ""
