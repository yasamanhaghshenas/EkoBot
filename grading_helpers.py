"""Helper functions for grading-specific question handling."""
import re


def is_grading_question(question):
    """Detect if question is specifically about grading/evaluation/exam weights."""
    q = (question or "").lower().strip()
    grading_keywords = [
        "grad", "evaluat", "assessment", "exam", "midterm", "final", "quiz",
        "weight", "percent", "score", "mark", "point", "rubric", "criteria",
    ]
    for keyword in grading_keywords:
        if re.search(keyword, q):
            return True
    return False


def extract_grading_only_response(course_contexts):
    """Extract ONLY the evaluation system from course context(s), no other info."""
    results = []
    
    for context in course_contexts:
        lines = []
        # Extract course code and name (first line)
        first_line = context.split('\n')[0] if context else ""
        if first_line and '-' in first_line:
            lines.append(first_line)
        
        # Extract only evaluation system
        eval_match = re.search(r"Evaluation System:(.*?)(?=\n\n|Description:|$)", context, re.DOTALL)
        if eval_match:
            eval_section = eval_match.group(1).strip()
            if eval_section:
                lines.append("")
                lines.append("Evaluation System:")
                lines.append(eval_section)
        
        if lines:
            results.append("\n".join(lines))
    
    return "\n\n".join(results) if results else ""
