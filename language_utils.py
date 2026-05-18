import re

from langdetect import DetectorFactory, LangDetectException, detect_langs

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


def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "tr"

    normalized = f" {text.casefold()} "

    if any(char in normalized for char in _TURKISH_HINTS[:6]):
        return "tr"
    if re.search(r"\b(?:ve|ama|mi|mı|mu|mü|ders|kredi|bölüm|bolum|fakulte|ortalama|mezun|staj)\b", normalized):
        return "tr"

    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "tr"
    except Exception:
        return "tr"

    if not candidates:
        return "tr"

    best = candidates[0]
    if best.lang == "tr" and best.prob >= 0.35:
        return "tr"
    if best.lang == "en" and best.prob >= 0.35:
        return "en"

    if any(hint in normalized for hint in _ENGLISH_HINTS):
        return "en"
    if any(hint in normalized for hint in ("kredi", "ders", "bölüm", "bolum", "gano", "ortalama")):
        return "tr"

    return "tr"


def get_fallback_response(session_language: str) -> str:
    lang = (session_language or "tr").strip().lower()
    if lang == "en":
        return (
            "Sorry, I can't help with that. I'm designed to assist with IEU-related topics such as courses, curriculum, graduation requirements, internships, student services, and university regulations. Feel free to ask me anything about those!"
        )

    return (
        "Üzgünüm, bu konuda sana yardımcı olamıyorum. Ben daha çok IEU ile ilgili konularda yardımcı olmak için tasarlandım: dersler, müfredat, mezuniyet, staj, öğrenci hizmetleri ve üniversite kuralları gibi. Bu konulardan biri hakkında soru sorabilirsin!"
    )
