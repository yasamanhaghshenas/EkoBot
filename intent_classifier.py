INTENT_KEYWORDS = {
    "gpa_credit_limit": ["gpa", "gano", "credit limit", "kredi limiti", "kaç ders alabilirim", "cgpa", "not ortalaması"],
    "attendance_policy": ["attendance", "devam", "missed class", "absence", "make-up", "telafi"],
    "password_help": ["password", "şifre", "sifre", "forgot", "reset", "login", "portal"],
    "student_certificate": ["student certificate", "öğrenci belgesi", "belge", "certificate"],
    "internship": ["internship", "staj", "praktik"],
    "graduation_apply": ["apply for graduation", "mezuniyet başvuru", "how do I graduate", "nasıl mezun"],
    "graduation_requirements": ["graduation requirements", "mezuniyet şartları", "240 ects", "cgpa 2.00"],
    "transcript": ["transcript", "transkript", "not dökümü"],
    "course_planning": ["curriculum", "müfredat", "course plan", "what courses", "ders planı", "semester plan"],
    "out_of_scope": ["poem", "şiir", "joke", "weather", "hava durumu", "love", "recipe", "tarif"],
    "general_greeting": ["hello", "hi", "merhaba", "hey", "selam", "good morning", "günaydın"],
}


def classify_intent(question: str) -> str:
    """Returns the most likely intent string, or 'unknown'."""
    text = (question or "").lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                return intent
    return "unknown"
