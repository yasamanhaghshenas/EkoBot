# save as check_course.py and run it
import requests
from bs4 import BeautifulSoup

code = "SE 116"
url = f"https://se.ieu.edu.tr/tr/syllabus/type/read/id/{code.replace(' ', '+')}"
response = requests.get(url, timeout=10)
response.encoding = "utf-8"
soup = BeautifulSoup(response.text, "html.parser")

# print all table rows to see what we get
for row in soup.find_all("tr"):
    text = row.get_text(separator=" | ", strip=True)
    if text:
        print(text)