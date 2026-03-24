import requests
from bs4 import BeautifulSoup

url = "https://chinhsachonline.chinhphu.vn/danh-sach-cau-hoi.htm"
headers = {"User-Agent": "Mozilla/5.0"}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")

# Find container of questions
items = soup.select(".question-box, .box-cau-hoi, .item") # Let's print all parent elements of a.question-title
print("--- Searching for list items ---")

questions = soup.select("a.question-title")
for idx, q in enumerate(questions[:3]):
    print(f"\n[Question {idx+1}]")
    print(f"Title: {q.get_text(strip=True)}")
    
    # Let's print the parent or siblings to find date
    parent = q.parent
    if parent:
        print(f"Parent Class: {parent.get('class')}")
        # Print inner text of parent
        print(f"Parent Text: {parent.get_text(strip=True)[:200]}...")
        
    grandparent = parent.parent if parent else None
    if grandparent:
         print(f"Grandparent Class: {grandparent.get('class')}")
         print(f"Grandparent Text: {grandparent.get_text(strip=True)[:300]}...")
