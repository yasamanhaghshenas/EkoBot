import re

from langdetect import DetectorFactory

DetectorFactory.seed = 0


_TURKISH_HINTS = (
    "ç", "ğ", "ı", "ö", "ş", "ü",
    "kredi", "ders", "bölüm", "bolum", "fakülte", "fakulte",
    "üniversite", "universite", "ortalama", "gano", "mezun", "staj",
)

_ENGLISH_HINTS = (
    " the ", " and ", "how many", "what is", "what are", "can i take",
    "this semester", "schedule", "credits", "course", "student affairs",
)

_ENGLISH_FORCE_WORDS = (
    "hello", "hi", "yes", "no", "ok", "okay", "thanks", "wtf", "omg", "lol",
    "you", "your", "what", "who", "how", "why", "when", "where", "can", "are",
    "is", "the", "i", "me", "my",
    "im", "good", "fine",
)

_ENGLISH_FORCE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _ENGLISH_FORCE_WORDS) + r")\b",
    re.IGNORECASE,
)


def detect_language(text: str, history: list = []) -> str:
    # For short messages, use history language instead of guessing
    text = (text or "").strip()
    normalized = f" {text.casefold()} "

    if _ENGLISH_FORCE_PATTERN.search(normalized):
        return "en"

    if len(text.strip()) < 12:
        if history:
            # Find last user message longer than 12 chars
            for msg in reversed(history):
                if msg["role"] == "user" and len(msg["content"].strip()) >= 12:
                    try:
                        from langdetect import detect
                        return detect(msg["content"])
                    except:
                        pass
        return "tr"  # default

    try:
        from langdetect import detect
        lang = detect(text)
        return lang if lang in ["tr", "en"] else "tr"
    except:
        return "tr"


def get_fallback_response(session_language: str) -> str:
    lang = (session_language or "tr").strip().lower()
    if lang == "en":
        return (
            "That's outside what I can help with — I'm focused on IEU topics like courses, graduation, internships, and student services. Anything IEU-related I can help you with?"
        )

    return (
        "Bu konuda yardımcı olamıyorum ama IEU ile ilgili her şeyde buradayım — dersler, mezuniyet, staj, öğrenci hizmetleri gibi. Sana başka konularda yardımcı olabilir miyim?"
    )
