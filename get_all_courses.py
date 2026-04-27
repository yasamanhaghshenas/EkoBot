import requests
from bs4 import BeautifulSoup
import json
import re
import time

from catalog import load_ieu_department_directory

# All departments with their curriculum URLs and syllabus URL patterns

departments = [
   
    {"name": "Software Engineering", "curr_url": "https://se.ieu.edu.tr/en/curr", "syllabus_pattern": "https://se.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Computer Engineering", "curr_url": "https://ce.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ce.ieu.edu.tr/en/syllabus_v2/type/read/id/{CODE}+{NUM}"},
    {"name": "Electrical & Electronics Engineering", "curr_url": "https://eee.ieu.edu.tr/en/curr", "syllabus_pattern": "https://eee.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Industrial Engineering", "curr_url": "https://ie.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ie.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Mechanical Engineering", "curr_url": "https://me.ieu.edu.tr/en/curr", "syllabus_pattern": "https://me.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Civil Engineering", "curr_url": "https://cie.ieu.edu.tr/en/curr", "syllabus_pattern": "https://cie.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Biomedical Engineering", "curr_url": "https://bme.ieu.edu.tr/en/curr", "syllabus_pattern": "https://bme.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Aerospace Engineering", "curr_url": "https://ae.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ae.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Mechatronics Engineering", "curr_url": "https://mce.ieu.edu.tr/en/curr", "syllabus_pattern": "https://mce.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Food Engineering", "curr_url": "https://dfe.ieu.edu.tr/en/curr", "syllabus_pattern": "https://dfe.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Genetics and Bioengineering", "curr_url": "https://gbe.ieu.edu.tr/en/curr", "syllabus_pattern": "https://gbe.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── ARTS & SCIENCES ──
    {"name": "Physics", "curr_url": "https://physics.ieu.edu.tr/en/curr", "syllabus_pattern": "https://physics.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Mathematics", "curr_url": "https://dm.ieu.edu.tr/en/curr", "syllabus_pattern": "https://dm.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "English Translation and Interpreting", "curr_url": "https://dti.ieu.edu.tr/en/curr", "syllabus_pattern": "https://dti.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Psychology", "curr_url": "https://psikoloji.ieu.edu.tr/en/curr", "syllabus_pattern": "https://psikoloji.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Sociology", "curr_url": "https://soc.ieu.edu.tr/en/curr", "syllabus_pattern": "https://soc.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── FINE ARTS & DESIGN ──
    {"name": "Industrial Design", "curr_url": "https://ent.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ent.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Interior Architecture", "curr_url": "https://ic.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ic.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Architecture", "curr_url": "https://mmr.ieu.edu.tr/en/curr", "syllabus_pattern": "https://mmr.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Textile and Fashion Design", "curr_url": "https://mt.ieu.edu.tr/en/curr", "syllabus_pattern": "https://mt.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Visual Communication Design", "curr_url": "https://ects.ieu.edu.tr/new/akademik.php?sid=curr&lang=en", "syllabus_pattern": "https://ilt.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── LAW ──
    {"name": "Law", "curr_url": "https://law.ieu.edu.tr/en/curr", "syllabus_pattern": "https://law.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── COMMUNICATION ──
    {"name": "Public Relations and Advertising", "curr_url": "https://pra.ieu.edu.tr/en/curr", "syllabus_pattern": "https://pra.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Cinema and Digital Media", "curr_url": "https://cdm.ieu.edu.tr/en/curr", "syllabus_pattern": "https://cdm.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "New Media and Communication", "curr_url": "https://nmc.ieu.edu.tr/en/curr", "syllabus_pattern": "https://nmc.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── BUSINESS ──
    {"name": "Economics", "curr_url": "https://eco.ieu.edu.tr/en/curr", "syllabus_pattern": "https://eco.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Business Administration", "curr_url": "https://dba.ieu.edu.tr/en/curr", "syllabus_pattern": "https://dba.ieu.edu.tr/en/curr"},
    {"name": "Logistics Management", "curr_url": "https://dlm.ieu.edu.tr/en/curr", "syllabus_pattern": "https://dlm.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Accounting and Auditing", "curr_url": "https://daa.ieu.edu.tr/en/curr", "syllabus_pattern": "https://daa.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Political Science and International Relations", "curr_url": "https://ireu.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ireu.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "International Trade and Finance", "curr_url": "https://itf.ieu.edu.tr/en/curr", "syllabus_pattern": "https://itf.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── HEALTH SCIENCES ──
    {"name": "Physiotherapy and Rehabilitation", "curr_url": "https://ftr.ieu.edu.tr/en/curr", "syllabus_pattern": "https://ftr.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Nursing", "curr_url": "https://nursing.ieu.edu.tr/en/curr", "syllabus_pattern": "https://nursing.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Nutrition and Dietetics", "curr_url": "https://bdb.ieu.edu.tr/en/curr", "syllabus_pattern": "https://bdb.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── MEDICINE ──
    {"name": "Medicine", "curr_url": "https://tip.ieu.edu.tr/en/curr", "syllabus_pattern": "https://tip.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},

    # ── VOCATIONAL SCHOOLS ──
    {"name": "Justice (Vocational)", "curr_url": "https://adl.ieu.edu.tr/adalet/en/curr", "syllabus_pattern": "https://adl.ieu.edu.tr/adalet/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Legal Office Management", "curr_url": "https://adl.ieu.edu.tr/hukuk_buro_yonetimi/en/curr", "syllabus_pattern": "https://adl.ieu.edu.tr/hukuk_buro_yonetimi/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Applied English Translation", "curr_url": "https://vs.ieu.edu.tr/uic/en/curr", "syllabus_pattern": "https://vs.ieu.edu.tr/uic/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Banking and Insurance", "curr_url": "https://bs.ieu.edu.tr/en/curr", "syllabus_pattern": "https://bs.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Computer Programming", "curr_url": "https://vs.ieu.edu.tr/cp/en/curr", "syllabus_pattern": "https://vs.ieu.edu.tr/cp/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Gastronomy and Culinary Arts", "curr_url": "https://culinary.ieu.edu.tr/en/curr", "syllabus_pattern": "https://culinary.ieu.edu.tr/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Child Development", "curr_url": "https://shs.ieu.edu.tr/cd/en/curr", "syllabus_pattern": "https://shs.ieu.edu.tr/cd/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Medical Imaging Techniques", "curr_url": "https://shs.ieu.edu.tr/mit/en/curr", "syllabus_pattern": "https://shs.ieu.edu.tr/mit/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Paramedic", "curr_url": "https://shs.ieu.edu.tr/paramedic/en/curr", "syllabus_pattern": "https://shs.ieu.edu.tr/paramedic/en/syllabus/type/read/id/{CODE}+{NUM}"},
    {"name": "Tourism and Hotel Management", "curr_url": "https://vs.ieu.edu.tr/tourism/en/curr", "syllabus_pattern": "https://vs.ieu.edu.tr/tourism/en/syllabus/type/read/id/{CODE}+{NUM}"},
]


def get_departments():
    directory = load_ieu_department_directory()
    if not directory:
        return departments

    out = []
    seen = set()
    for item in directory:
        name = (item.get("canonical_department") or item.get("source_department") or "").strip()
        curr_url = (item.get("curriculum_url") or "").strip()
        syllabus_pattern = (item.get("syllabus_pattern") or "").strip()
        if not name or not curr_url or not syllabus_pattern:
            continue
        key = (name, curr_url, syllabus_pattern)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "curr_url": curr_url, "syllabus_pattern": syllabus_pattern})

    return out

def get_course_codes_from_page(dept):
    try:
        response = requests.get(dept["curr_url"], timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        full_text = soup.get_text()

        pattern = r'\b([A-Z]{2,5})\s(\d{3})\b'
        matches = re.findall(pattern, full_text)
        codes = list(set([f"{m[0]} {m[1]}" for m in matches]))

        # attach the syllabus pattern to each code
        result = []
        for code in codes:
            parts = code.split(" ")
            prefix = parts[0]
            num = parts[1]
            syllabus_url = dept["syllabus_pattern"].replace("{CODE}", prefix).replace("{NUM}", num)
            result.append({
                "code": code,
                "department": dept["name"],
                "syllabus_url": syllabus_url
            })

        print(f"  ✅ {dept['name']}: {len(result)} courses")
        return result

    except Exception as e:
        print(f"  ❌ {dept['name']}: {e}")
        return []

print("Collecting course codes from all departments...\n")
all_courses = []
seen_codes = set()

for dept in get_departments():
    courses = get_course_codes_from_page(dept)
    for c in courses:
        if c["code"] not in seen_codes:
            seen_codes.add(c["code"])
            all_courses.append(c)
    time.sleep(0.8)

print(f"\nTotal unique courses found: {len(all_courses)}")

with open("course_codes_full.json", "w", encoding="utf-8") as f:
    json.dump(all_courses, f, ensure_ascii=False, indent=2)

print("Saved to course_codes_full.json")
