import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib3
from bs4 import BeautifulSoup


SUPPORT_PAGE_SOURCES = [
    {
        "category": "student_affairs",
        "urls": [
            "https://oim.ieu.edu.tr/en",
            "https://oim.ieu.edu.tr/tr",
        ],
    },
    {
        "category": "academic_calendar",
        "urls": [
            "https://www.ieu.edu.tr/en/akademik-takvim",
            "https://www.ieu.edu.tr/tr/akademik-takvim",
        ],
    },
    {
        "category": "it_support",
        "urls": [
            "https://itsupport.ieu.edu.tr/en/",
            "https://itsupport.ieu.edu.tr/tr/",
            "https://comp.izmirekonomi.edu.tr/en/ekoid",
        ],
    },
    {
        "category": "internship",
        "urls": [
            "https://kariyer.ieu.edu.tr/en",
            "https://kariyer.ieu.edu.tr/tr",
            "https://kariyer.ieu.edu.tr/en/staj-basvuru-kilavuzu",
        ],
    },
]


def _infer_language(source_url, text):
    lower_url = (source_url or "").lower()
    if "/en" in lower_url or lower_url.endswith("/en"):
        return "en"
    if "/tr" in lower_url or lower_url.endswith("/tr"):
        return "tr"

    sample = (text or "")[:1000].lower()
    if any(ch in sample for ch in ["ç", "ğ", "ı", "ö", "ş", "ü"]):
        return "tr"
    return "en"


def _extract_clean_text_from_html(html):
    soup = BeautifulSoup(html or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "footer", "nav", "header"]):
        tag.decompose()

    for node in soup.select(
        ".menu, .navbar, .navigation, .breadcrumbs, .breadcrumb, .cookie, .cookies, "
        ".social, .sidebar, #menu, #nav, #navbar, #footer"
    ):
        node.decompose()

    # Prefer the main content body when available.
    main = soup.find("main") or soup.find("article") or soup.find("section")
    target = main if main is not None else soup.body or soup
    text = target.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _chunk_text_tokens(text, max_tokens=400, overlap=50):
    tokens = (text or "").split()
    if not tokens:
        return []

    chunks = []
    step = max(max_tokens - overlap, 1)
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + max_tokens]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + max_tokens >= len(tokens):
            break
    return chunks


def _scrape_support_page(url, category, verify_ssl=True):
    try:
        response = requests.get(url, timeout=20, verify=verify_ssl)
        response.encoding = "utf-8"
        if response.status_code >= 400:
            return []

        clean_text = _extract_clean_text_from_html(response.text)
        if not clean_text:
            return []

        language = _infer_language(url, clean_text)
        chunks = _chunk_text_tokens(clean_text, max_tokens=400, overlap=50)
        return [
            {
                "text": chunk,
                "metadata": {
                    "category": category,
                    "language": language,
                    "source_url": url,
                },
            }
            for chunk in chunks
        ]
    except Exception:
        return []


def scrape_support_pages(verify_ssl=True):
    """
    Scrape and chunk high-value non-course IEU service pages.
    Returns a list of dicts with `text` and `metadata`.
    """
    items = []
    for source in SUPPORT_PAGE_SOURCES:
        category = source.get("category")
        for url in source.get("urls", []):
            if not url:
                continue
            items.extend(_scrape_support_page(url, category, verify_ssl=verify_ssl))
    return items


def scrape_course(course, verify_ssl=True):
    try:
        url = course["syllabus_url"]
        response = requests.get(url, timeout=15, verify=verify_ssl)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        # check page actually has course content
        page_text = response.text
        if "Dersin Adı" not in page_text and "Course Name" not in page_text and "Ders Tanıtım" not in page_text:
            return None

        code = course["code"]
        department = course["department"]
        text = f"Course Code: {code}\n"
        text += f"Department: {department}\n"

        # label translation map
        label_map = {
            "Dersin Adı": "Course Name",
            "Dersin Dili": "Language",
            "Dersin Türü": "Type",
            "Dersin Düzeyi": "Level",
            "Dersin Amacı": "Course Aim",
            "Ders Tanımı": "Course Description",
            "Dersin Koordinatörü": "Coordinator",
            "Öğretim Eleman(lar)ı": "Instructors",
            "Yerel Kredi": "Credits",
            "AKTS": "ECTS",
            "Yarıyıl": "Semester",
            "Ön-Koşul(lar)": "Prerequisites",
        }

        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])

            # standard 2-column rows
            if len(cells) == 2:
                label = cells[0].get_text(separator=" ", strip=True)
                value = cells[1].get_text(separator=" ", strip=True)
                if label and value and len(label) < 100 and len(value) < 1000:
                    label = label_map.get(label, label)
                    text += f"{label}: {value}\n"

            # course code + semester + credits row (6 columns)
            elif len(cells) == 6:
                code_cell = cells[0].get_text(strip=True)
                semester = cells[1].get_text(strip=True)
                credits = cells[4].get_text(strip=True)
                ects = cells[5].get_text(strip=True)
                if code_cell == code:
                    text += f"Semester: {semester}\n"
                    text += f"Credits: {credits}\n"
                    text += f"ECTS: {ects}\n"

            # weekly topics (3 columns, first is a number)
            elif len(cells) == 3:
                col1 = cells[0].get_text(strip=True)
                col2 = cells[1].get_text(strip=True)
                if col1.isdigit() and col2:
                    text += f"Week {col1}: {col2}\n"

        # evaluation/grading table extraction
        evaluation_text = ""
        eval_table = soup.find("table", {"id": "evaluation_table1"})
        if eval_table:
            for row in eval_table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 3:
                    activity = cells[0].get_text(strip=True)
                    number = cells[1].get_text(strip=True)
                    weight = cells[2].get_text(strip=True)
                    
                    # skip header rows and total row
                    skip_keywords = ["Semester Activities", "Number", "Weightin", 
                                   "Total", "Toplam", "Dönem imi", "Dönem Sonu",
                                   "Weighting of"]
                    if any(k in activity for k in skip_keywords):
                        continue
                    
                    # skip rows where both number and weight are "-"
                    if number == "-" and weight == "-":
                        continue
                    
                    if activity and len(activity) > 2:
                        evaluation_text += f"{activity}: {number} time(s), {weight}% weight\n"

        if evaluation_text:
            text += f"\nEvaluation System:\n{evaluation_text}"

        # special prerequisite extraction (nested table structure)
        # special prerequisite extraction - handles both TR and EN pages
        prereq = "None"
        for row in soup.find_all("tr"):
            row_text = row.get_text(separator=" ", strip=True)
            if "Ön-Koşul" in row_text or "Prerequisite" in row_text:
                raw = row.get_text(separator=" ", strip=True)
       
                raw = raw.replace("Ön-Koşul(lar)", "").replace("Prerequisites", "").strip()
       
                if raw and raw not in ["-", "Yok", "None", "Yok."]:
                  prereq = raw
                break

        text += f"Prerequisites: {prereq}\n"

        return {
            "code": code,
            "department": department,
            "url": url,
            "text": text
        }

    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Scrape course syllabus pages into courses.json")
    parser.add_argument("--input", default="course_codes_full.json")
    parser.add_argument("--output", default="courses.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--insecure", action="store_true", help="Disable SSL certificate verification")
    args = parser.parse_args()

    verify_ssl = not args.insecure
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("WARNING: SSL verification is disabled (--insecure).")

    with open(args.input, "r", encoding="utf-8") as f:
        course_list = json.load(f)

    print(f"Loaded {len(course_list)} courses. Starting scraper...\n")

    unique_courses = {}
    for course in course_list:
        url = course.get("syllabus_url")
        if url and url not in unique_courses:
            unique_courses[url] = course

    print(f"Unique syllabus URLs: {len(unique_courses)}")

    cache = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        future_map = {
            executor.submit(scrape_course, course, verify_ssl=verify_ssl): url
            for url, course in unique_courses.items()
        }
        for future in as_completed(future_map):
            url = future_map[future]
            cache[url] = future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(unique_courses):
                print(f"Progress: {completed}/{len(unique_courses)} unique URLs scraped")
            if args.sleep > 0:
                time.sleep(args.sleep)

    all_courses = []
    failed = 0
    for course in course_list:
        url = course.get("syllabus_url")
        result = cache.get(url)
        if result:
            scraped = dict(result)
            scraped["code"] = course["code"]
            scraped["department"] = course["department"]
            scraped["url"] = url
            all_courses.append(scraped)
        else:
            failed += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_courses, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done!")
    print(f"Successfully scraped: {len(all_courses)} courses")
    print(f"Failed/not found: {failed} courses")


if __name__ == "__main__":
    main()