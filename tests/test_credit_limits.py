import unittest

from credit_limits import (
    extract_gpa_from_question,
    get_credit_limit_by_gpa,
    is_gpa_credit_limit_question,
)


class CreditLimitTests(unittest.TestCase):
    def test_gpa_2_5_returns_44_ects(self):
        question = "With a 2.5 GPA, how many ECTS credits can I take?"
        self.assertTrue(is_gpa_credit_limit_question(question))
        self.assertEqual(extract_gpa_from_question(question), 2.5)
        self.assertEqual(get_credit_limit_by_gpa(2.5), 44)

    def test_gpa_2_2_returns_38_ects(self):
        question = "How many credits can I take with a GPA of 2.2?"
        self.assertTrue(is_gpa_credit_limit_question(question))
        self.assertEqual(extract_gpa_from_question(question), 2.2)
        self.assertEqual(get_credit_limit_by_gpa(2.2), 38)

    def test_gpa_below_2_returns_32_ects(self):
        question = "My GPA is 1.8; what is my maximum semester course load in ECTS?"
        self.assertTrue(is_gpa_credit_limit_question(question))
        self.assertEqual(extract_gpa_from_question(question), 1.8)
        self.assertEqual(get_credit_limit_by_gpa(1.8), 32)


if __name__ == "__main__":
    unittest.main()
