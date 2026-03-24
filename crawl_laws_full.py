import csv
import os
import time
import random
from playwright.sync_api import sync_playwright

CSV_PATH = '/Users/nguyenthithutam/Desktop/Callbot/missing_laws.csv'
OUTPUT_DIR = '/Users/nguyenthithutam/Desktop/Callbot/extracted_texts/'
LOG_PATH = '/Users/nguyenthithutam/Desktop/Callbot/crawl_progress.log'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(message):
    print(message)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def crawl():
    log_message("=== KHỞI ĐỘNG CRAWLER (ISOLATED CONTEXT) ===")
    
    laws_to_crawl = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            laws_to_crawl.append(row[1])
            
    log_message(f"Tìm thấy {len(laws_to_crawl)} văn bản cần thu thập.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for index, law_id in enumerate(laws_to_crawl, 1):
            file_safe_name = law_id.replace('/', '_').replace(' ', '_')
            output_file = os.path.join(OUTPUT_DIR, f"{file_safe_name}.txt")
            
            if os.path.exists(output_file):
                log_message(f"[{index}/{len(laws_to_crawl)}] {law_id} - Đã tồn tại, bỏ qua.")
                continue

            log_message(f"[{index}/{len(laws_to_crawl)}] Đang xử lý: {law_id}")
            
            # TẠO CONTEXT MỚI CHO TỪNG VÒNG LẶP ĐỂ XÓA COOKIE / CACHE
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                viewport={'width': 1280, 'height': 800}
            )
            page = context.new_page()

            try:
                page.goto("https://luatvietnam.vn/", timeout=30000)
                
                search_selector = '#search_q'
                page.wait_for_selector(search_selector, timeout=5000)
                page.fill(search_selector, law_id)
                page.click('button.btn-search-lvn')
                page.wait_for_timeout(5000) 
                
                all_links = page.locator('a')
                count = all_links.count()
                
                target_link = None
                for i in range(count):
                    link = all_links.nth(i)
                    text = link.inner_text().strip()
                    title_attr = link.get_attribute('title') or ""
                    
                    if law_id in text or law_id in title_attr:
                        if len(text) > 20: 
                            target_link = link
                            break

                if target_link:
                    target_link.click()
                    page.wait_for_timeout(5000) 
                    
                    extracted_text = ""
                    selectors = ['#divNoiDung', '.content-details', '.box-content']
                    
                    for sel in selectors:
                        elements = page.locator(sel)
                        if elements.count() > 0:
                            extracted_text = elements.first.inner_text()
                            if "CỘNG HÒA" in extracted_text:
                                break
                    
                    if not extracted_text:
                         divs = page.locator('div')
                         for j in range(divs.count()):
                              txt = divs.nth(j).inner_text()
                              if "CỘNG HÒA" in txt and len(txt) > 2000:
                                   extracted_text = txt
                                   break
                                   
                    if extracted_text and len(extracted_text.strip()) > 100:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_text)
                        log_message(f"  -> LƯU THÀNH CÔNG: {output_file}")
                    else:
                        log_message(f"  -> CẢNH BÁO: Không lấy được nội dung cho {law_id}")
                else:
                    log_message(f"  -> KHÔNG tìm thấy kết quả cho {law_id}")

            except Exception as e:
                log_message(f"  -> LỖI khi tải {law_id}: {str(e)}")

            # Đóng page/context ngay lập tức
            page.close()
            context.close()

            delay = random.uniform(5, 10)
            time.sleep(delay)

        browser.close()
    log_message("=== HOÀN TẤT CRAWLER ===")

if __name__ == "__main__":
    crawl()
