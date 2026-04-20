import requests
from bs4 import BeautifulSoup
import json

DEPARTMENTS = {
    "CE": "ce",
    "BME": "bme",
    "FE": "fe",
    "GBE": "gbe",
    "AE": "ae",
    "CIVE": "cive",
    "MCE": "mce",
    "SE": "se", 
    "EEE": "eee",
    "IE": "ie",
    "ME": "me",
    "FTR": "ftr",
}

def scrape_graduation_requirements(dept_code, subdomain):
    url = f"https://{subdomain}.ieu.edu.tr/en/bologna/aid/9"
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; EkoBot/1.0)"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # IEU pages can vary; try several likely content containers.
        content = (
            soup.find("div", class_="col-md-9")
            or soup.find("div", class_="col-lg-9")
            or soup.select_one("main")
            or soup.select_one("div.container")
        )

        if content:
            clean_text = content.get_text(strip=True, separator=" ")
            if not clean_text:
                return None
            return {
                "department": dept_code,
                "url": url,
                "text": clean_text
            }
    except Exception as e:
        print(f"Failed for {dept_code}: {e}")
    return None

if __name__ == "__main__":
    results = []
    for dept, sub in DEPARTMENTS.items():
        print(f"Scraping {dept}...")
        data = scrape_graduation_requirements(dept, sub)
        if data:
            results.append(data)
            print(f"  Got: {data['text'][:100]}...")
    
    with open("dept_info.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Saved {len(results)} departments to dept_info.json")