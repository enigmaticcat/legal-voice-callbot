import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime

BASE_URL = "https://chinhsachonline.chinhphu.vn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_question_links(page=1):
    """
    Lấy danh sách link và ngày tháng của câu hỏi từ 1 trang danh mục.
    """
    if page == 1:
        url = f"{BASE_URL}/danh-sach-cau-hoi.htm"
    else:
        url = f"{BASE_URL}/danh-sach-cau-hoi/trang-{page}.htm"
    # print(f"Quét danh sách trang {page}...")
    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    
    results = []
    
    # Tìm tất cả các container của câu hỏi
    items = soup.select(".content")
    
    for item in items:
        a = item.select_one("a.question-title")
        if not a:
            continue
            
        href = a.get("href")
        if not href:
            continue
            
        link = BASE_URL + href if href.startswith("/") else href
        
        # Bóc tách ngày từ text (Regex \d{2}/\d{2}/\d{4})
        full_text = item.get_text(strip=True)
        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', full_text)
        date_str = date_match.group(1) if date_match else None
        
        results.append({
            "link": link,
            "date": date_str,
            "title": a.get_text(strip=True)
        })
    
    # Kiểm tra có trang kế tiếp hay không
    next_page = soup.select_one(f'a[href*="trang-{page+1}"]')
    has_next = next_page is not None
    if not has_next:
        next_btn = soup.select_one("a.pagination-number.page-next")
        has_next = next_btn is not None
        
    return results, has_next

def get_question_detail(url):
    """Lấy nội dung chi tiết 1 câu hỏi"""
    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    
    data = {"url": url}
    
    # Tiêu đề
    title = soup.select_one("h1.detail__title")
    data["title"] = title.get_text(strip=True) if title else ""
    
    # Lĩnh vực / bộ ngành
    category = soup.select_one("a.detail__category")
    data["category"] = category.get_text(strip=True) if category else ""
    
    # Nội dung câu hỏi 
    question_body = soup.select_one(".detail__cquestion")
    data["question"] = question_body.get_text(separator=" ", strip=True) if question_body else ""
    
    # Nội dung trả lời (nằm trong .detail__reply -> .detail__rcontent)
    answer_body = soup.select_one(".detail__rcontent")
    data["answer"] = answer_body.get_text(separator=" ", strip=True) if answer_body else ""
    
    return data

def crawl_all(target_date_str="01/01/2024", max_pages=None):
    all_data = []
    page = 1
    target_date = datetime.strptime(target_date_str, "%d/%m/%Y")
    output_file = "/Users/nguyenthithutam/Desktop/Callbot/cauhoichinhsach_2024.json"
    
    print(f"🚀 Bắt đầu crawl dữ liệu mới từ ngày: {target_date_str}")
    stop_crawling = False
    
    while True:
        print(f"Quét trang {page}...")
        results, has_next = get_question_links(page)
        print(f"  → Tìm thấy {len(results)} câu hỏi trên trang {page}")
        
        if not results:
            print("  ⚠️ Không tìm thấy link câu hỏi nào nữa. Dừng.")
            break

        for idx, res_item in enumerate(results):
            link = res_item["link"]
            date_str = res_item["date"]
            title = res_item["title"]
            
            if date_str:
                try:
                    current_date = datetime.strptime(date_str, "%d/%m/%Y")
                    # Nếu ngày nhỏ hơn target_date thì DỪNG 
                    if current_date < target_date:
                        print(f"\n🛑 Phát hiện ngày {date_str} < mốc {target_date_str}. Dừng crawl.")
                        stop_crawling = True
                        break
                except ValueError:
                    pass

            # Chỉ đọc file nếu link hợp lệ
            try:
                print(f"    [{idx+1}/{len(results)}] Đang đọc: {title[:40]}... ({date_str})")
                detail = get_question_detail(link)
                detail["date"] = date_str  # Lưu metadata ngày
                
                if detail.get("question") and detail.get("answer"):
                    all_data.append(detail)
                    # Ghi file tạm thời liên tục để tránh mất dữ liệu nếu crash
                    if len(all_data) % 10 == 0:
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(all_data, f, ensure_ascii=False, indent=2)
                else:
                    print(f"      ⚠️ Trống dữ liệu: {link}")
                time.sleep(0.5)  # Tránh trigger firewall
            except Exception as e:
                 print(f"      ❌ Lỗi khi đọc {link}: {e}")
                 
        if stop_crawling or not has_next or (max_pages and page >= max_pages):
            break
            
        page += 1
        time.sleep(1.2)
        
    # Ghi kết quả cuối cùng
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã hoàn tất! Crawl thành công {len(all_data)} câu hỏi. Lưu tại: {output_file}")
    return all_data

if __name__ == "__main__":
    crawl_all(target_date_str="01/01/2024")