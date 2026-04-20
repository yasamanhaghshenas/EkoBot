import requests
from bs4 import BeautifulSoup
import json

# All department curriculum URLs from Tina's research
DEPARTMENTS = {
    "CE": "https://ects.ieu.edu.tr/new/akademik.php?section=ce.cs.ieu.edu.tr&sid=curr&lang=en",
    "SE": "https://ects.ieu.edu.tr/new/akademik.php?section=se.cs.ieu.edu.tr&sid=curr&lang=en",
    "EEE": "https://ects.ieu.edu.tr/new/akademik.php?section=ete.cs.ieu.edu.tr&sid=curr&lang=en",
    "IE": "https://ects.ieu.edu.tr/new/akademik.php?section=is.cs.ieu.edu.tr&sid=curr&lang=en",
    "GBE": "https://ects.ieu.edu.tr/new/akademik.php?section=gbe.ieu.edu.tr&sid=curr&lang=en",
    "AE": "https://ects.ieu.edu.tr/new/akademik.php?section=ae.ieu.edu.tr&sid=curr&lang=en",
    "CIVE": "https://ects.ieu.edu.tr/new/akademik.php?section=cie.ieu.edu.tr&sid=curr&lang=en",
    "ME": "https://ects.ieu.edu.tr/new/akademik.php?section=me.ieu.edu.tr&sid=curr&lang=en",
    "BME": "https://bme.ieu.edu.tr/en/curr",
    "MCE": "https://mce.ieu.edu.tr/en/curr",
    "FE": "https://dfe.ieu.edu.tr/en/curr",
}

def scrape_curriculum(dept_code, url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=15, headers=headers)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, "html.parser")

        curriculum = {
            "department": dept_code,
            "url": url,
            "semesters": {}
        }
        curriculum["electives"] = []

        # parse each table as one possible semester block
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            first_row_cells = rows[0].find_all(["td", "th"])
            first_row_text = " ".join(c.get_text(" ", strip=True) for c in first_row_cells).strip()
            lowered = first_row_text.lower()

            # CE/SE curriculum pages keep semester label in the first row of each table.
            if not any(kw in lowered for kw in ["semester", "year", "fall", "spring", "güz", "bahar"]):
                continue

            current_semester = first_row_text
            curriculum["semesters"][current_semester] = []

            # start from row 2 to skip semester label + header row
            for row in rows[2:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue

                texts = [c.get_text(" ", strip=True) for c in cells]
                code = texts[0]

                # Some tables store prerequisite as second column and name as third.
                if len(texts) >= 3 and texts[1] == "":
                    name = texts[2]
                elif len(texts) >= 3 and texts[1].lower() in ["pre.", "prerequisite", "önkoşul"]:
                    name = texts[2]
                else:
                    name = texts[1]

                if code and name and len(code) >= 3 and any(c.isdigit() for c in code):
                    curriculum["semesters"][current_semester].append({
                        "code": code,
                        "name": name
                    })

        # find elective courses section at bottom of page
        all_text_elements = soup.find_all(["h2", "h3", "h4", "th", "td"])
        in_electives_section = False

        for element in soup.find_all(["h2", "h3", "h4", "table"]):
            text = element.get_text(strip=True).lower()
            
            if "elective course" in text and element.name in ["h2","h3","h4"]:
                in_electives_section = True
                continue

            # In ECTS pages the elective section title is often the first row of a table.
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
                        if code and len(code) >= 3 and any(c.isdigit() for c in code):
                            curriculum["electives"].append({
                                "code": code,
                                "name": name
                            })
                break

        # remove empty semesters
        curriculum["semesters"] = {
            k: v for k, v in curriculum["semesters"].items() if v
        }
        
        return curriculum

    except Exception as e:
        print(f"  Failed for {dept_code}: {e}")
        return None

if __name__ == "__main__":
    results = []
    for dept, url in DEPARTMENTS.items():
        print(f"Scraping curriculum for {dept}...")
        data = scrape_curriculum(dept, url)
        if data and data["semesters"]:
            results.append(data)
            total = sum(len(v) for v in data["semesters"].values())
            print(f"  Found {len(data['semesters'])} semesters, {total} courses")
        else:
            print(f"  No data found for {dept}")

    with open("curriculum.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(results)} departments to curriculum.json")