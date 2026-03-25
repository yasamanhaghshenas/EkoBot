import requests
from bs4 import BeautifulSoup
import json
import time

with open("course_codes_full.json", "r", encoding="utf-8") as f:
    course_list = json.load(f)

print(f"Loaded {len(course_list)} courses. Starting scraper...\n")

def scrape_course(course):
    try:
        url = course["syllabus_url"]
        response = requests.get(url, timeout=10)
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

all_courses = []
failed = 0

for i, course in enumerate(course_list):
    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{len(course_list)} — scraped {len(all_courses)} successfully")

    result = scrape_course(course)
    if result:
        all_courses.append(result)
    else:
        failed += 1

    time.sleep(0.5)  # slightly faster now since we have more courses

with open("courses.json", "w", encoding="utf-8") as f:
    json.dump(all_courses, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done!")
print(f"Successfully scraped: {len(all_courses)} courses")
print(f"Failed/not found: {failed} courses")