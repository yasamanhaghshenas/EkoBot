#!/usr/bin/env python3
"""Quick test of grading detection and extraction."""

from grading_helpers import is_grading_question, extract_grading_only_response

# Test grading question detection
test_questions = [
    "what is the grading system for math 153",
    "how will I be graded in this course",
    "what are the exam weights",
    "tell me about math 153",  # This should NOT trigger grading-only
    "what is the evaluation method",
    "midterm percentage",
]

print("Testing grading question detection:")
for q in test_questions:
    result = is_grading_question(q)
    print(f"  '{q}' -> {result}")

# Test extraction
print("\nTesting extraction (from sample context):")
sample_context = """MATH 153 - Calculus I
Department: CE
Semester: SPRING
Credits: 3
ECTS: 6
Teacher/Lecturer(s): Dr. Öğr. Üyesi Demet Ersoy Özbek

Evaluation System:
- Quizzes / Studio Critiques: 4 time(s), 20% weight
- Midterm: 1 time(s), 30% weight
- Final Exam: 1 time(s), 50% weight

Description: Calculus I provides important tools..."""

result = extract_grading_only_response([sample_context])
print(result)
