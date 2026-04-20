import argparse
import json
import re
import time

import requests
import urllib3
from bs4 import BeautifulSoup


def scrape_curriculum_from_url(url, verify_ssl=True):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=20, headers=headers, verify=verify_ssl)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    semesters = {}
    electives = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        first_row_cells = rows[0].find_all(["td", "th"])
        first_row_text = " ".join(c.get_text(" ", strip=True) for c in first_row_cells).strip()
        lowered = first_row_text.lower()

        if not any(kw in lowered for kw in ["semester", "year", "fall", "spring", "guz", "güz", "bahar"]):
            continue

        current_semester = first_row_text
        semesters[current_semester] = []

        for row in rows[2:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            texts = [c.get_text(" ", strip=True) for c in cells]
            code = texts[0]

            if len(texts) >= 3 and texts[1] == "":
                name = texts[2]
            elif len(texts) >= 3 and texts[1].lower() in ["pre.", "prerequisite", "onkosul", "önkoşul"]:
                name = texts[2]
            else:
                name = texts[1]

            if code and name and len(code) >= 3 and any(ch.isdigit() for ch in code):
                semesters[current_semester].append({"code": code, "name": name})

    in_electives_section = False
    for element in soup.find_all(["h2", "h3", "h4", "table"]):
        text = element.get_text(strip=True).lower()

        if "elective course" in text and element.name in ["h2", "h3", "h4"]:
            in_electives_section = True
            continue

        if element.name == "table":
            rows = element.find_all("tr")
            if rows:
                header_cells = rows[0].find_all(["td", "th"])
                header_text = " ".join(c.get_text(" ", strip=True) for c in header_cells).lower()
                if "elective course" in header_text:
                    in_electives_section = True

        if in_electives_section and element.name == "table":
            rows = element.find_all("tr")
            for row in rows[2:]:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 3:
                    code = cells[0].get_text(strip=True)
                    name = cells[2].get_text(strip=True)
                    if code and len(code) >= 3 and any(ch.isdigit() for ch in code):
                        electives.append({"code": code, "name": name})
            break
        breaking_keywords = [ ""]

    semesters = {k: v for k, v in semesters.items() if v}
    return semesters, electives, soup.get_text(" ", strip=True)


def extract_course_codes(text):
    pattern = r"\b([A-Z]{2,6})\s(\d{3})\b"
    found = re.findall(pattern, text)
    return sorted(set([f"{x[0]} {x[1]}" for x in found]))


def build_syllabus_url(pattern, code):
    if not pattern:
        return None
    parts = code.split(" ")
    if len(parts) != 2:
        return None
    return pattern.replace("{CODE}", parts[0]).replace("{NUM}", parts[1])


def is_usable_catalog_row(item, include_common=False, include_graduate=False):
    dept_name = (item.get("department_name") or "").strip().lower()
    sheet = (item.get("sheet") or "").strip().lower()
    curr_url = (item.get("curriculum_url") or "").strip().lower()

    if not curr_url:
        return False

    # Skip placeholders and unresolved templates.
    if "{dept}" in curr_url or dept_name in {"?", "", "none"}:
        return False

    # Skip non-curriculum endpoints by default.
    if ("/curr" not in curr_url) and ("akademik.php" not in curr_url):
        return False

    # Common-pool / orientation rows are useful metadata but not major curricula.
    if ("common courses" in sheet or "gec" in sheet) and not include_common:
        return False

    # Graduate rows need different handling/discovery; exclude from undergrad advisor by default.
    if ("graduate" in sheet or "lisansustu" in curr_url) and not include_graduate:
        return False

    # Exclude preparatory language rows in major planning flow.
    if "english prep" in dept_name or "hazirlik" in dept_name:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Sync program_catalog.json into curriculum.json and course_codes_full.json")
    parser.add_argument("--catalog", default="program_catalog.json")
    parser.add_argument("--curriculum-out", default="curriculum.json")
    parser.add_argument("--codes-out", default="course_codes_full.json")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--include-common", action="store_true", help="Include Common Courses (GEC) rows")
    parser.add_argument("--include-graduate", action="store_true", help="Include Graduate School rows")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL certificate verification for requests")
    args = parser.parse_args()

    verify_ssl = not args.insecure
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("WARNING: SSL verification is disabled (--insecure).")

    with open(args.catalog, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    curriculum_rows = []
    code_rows = []
    seen_code_key = set()
    cache = {}
    failed_urls = set()

    skipped = 0
    kept = 0

    for i, item in enumerate(catalog, start=1):
        if not is_usable_catalog_row(
            item,
            include_common=args.include_common,
            include_graduate=args.include_graduate,
        ):
            skipped += 1
            continue

        kept += 1
        dept_code = (item.get("department_code") or item.get("department_name") or "").strip()
        prog_name = (item.get("program_name") or item.get("department_name") or dept_code).strip()
        entry_year = item.get("entry_year")
        curr_url = item.get("curriculum_url")
        syllabus_pattern = item.get("syllabus_pattern")

        if not curr_url:
            continue

        print(f"[{i}/{len(catalog)}] Syncing curriculum: {prog_name} ({dept_code})")

        if curr_url in failed_urls:
            continue

        if curr_url in cache:
            semesters, electives, page_text = cache[curr_url]
        else:
            try:
                semesters, electives, page_text = scrape_curriculum_from_url(curr_url, verify_ssl=verify_ssl)
                cache[curr_url] = (semesters, electives, page_text)
            except Exception as e:
                print(f"  Failed: {e}")
                failed_urls.add(curr_url)
                continue

        if semesters:
            curriculum_rows.append(
                {
                    "department": dept_code,
                    "program_name": prog_name,
                    "entry_year": entry_year,
                    "url": curr_url,
                    "semesters": semesters,
                    "electives": electives,
                }
            )

        if syllabus_pattern:
            codes = extract_course_codes(page_text)
            for code in codes:
                key = (code, dept_code, entry_year)
                if key in seen_code_key:
                    continue
                seen_code_key.add(key)
                code_rows.append(
                    {
                        "code": code,
                        "department": dept_code,
                        "program_name": prog_name,
                        "entry_year": entry_year,
                        "syllabus_url": build_syllabus_url(syllabus_pattern, code),
                    }
                )

        time.sleep(max(args.sleep, 0.0))

    with open(args.curriculum_out, "w", encoding="utf-8") as f:
        json.dump(curriculum_rows, f, ensure_ascii=False, indent=2)

    with open(args.codes_out, "w", encoding="utf-8") as f:
        json.dump(code_rows, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Catalog Sync Complete")
    print("=" * 60)
    print(f"Catalog input rows  : {len(catalog)}")
    print(f"Rows kept          : {kept}")
    print(f"Rows skipped       : {skipped}")
    print(f"Unique curr URLs   : {len(cache)}")
    print(f"Failed curr URLs   : {len(failed_urls)}")
    print(f"Curriculum records : {len(curriculum_rows)}")
    print(f"Course code rows   : {len(code_rows)}")
    print(f"Saved curriculum   : {args.curriculum_out}")
    print(f"Saved code list    : {args.codes_out}")


if __name__ == "__main__":
    main()
