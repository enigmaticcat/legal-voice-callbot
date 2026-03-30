"""
crawl_qa.py
===========
Crawl Q&A dinh dưỡng từ:
  1. VnExpress  — static  (~nhiều categories sức khỏe, lọc dinh dưỡng)
  2. Vinmec     — Playwright
  3. BV Thu Cúc — Playwright

Output: qa_corpus.jsonl
  { "source", "url", "question", "answer", "doctor", "category", "crawled_at" }

Cài đặt:
  pip install playwright beautifulsoup4 requests
  playwright install chromium
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path("/Users/nguyenthithutam/Desktop/Callbot/qa_corpus.jsonl")
LOG_FILE    = Path("/Users/nguyenthithutam/Desktop/Callbot/crawl_qa.log")

MIN_ANSWER_LENGTH = 100
DELAY = 0.8

NUTRITION_KEYWORDS = [
    "dinh dưỡng", "ăn uống", "thực phẩm", "chế độ ăn", "bữa ăn",
    "vitamin", "protein", "canxi", "sắt", "kẽm", "omega", "calo",
    "béo phì", "giảm cân", "thừa cân", "suy dinh dưỡng",
    "sữa", "rau", "trái cây", "thịt", "cá", "đậu", "hạt",
    "đường", "tinh bột", "chất xơ", "cholesterol",
    "ăn dặm", "ăn chay", "ăn kiêng", "khẩu phần",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── Writer ───────────────────────────────────────────────────────────────────

class QAWriter:
    def __init__(self, path: Path):
        self.path  = path
        self.count = 0
        self.seen  = set()
        self._load_existing()
        self._fh   = open(path, "a", encoding="utf-8")

    def _load_existing(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        self.seen.add(doc.get("url", doc.get("question", "")))
                        self.count += 1
                    except Exception:
                        pass
            if self.count:
                log.info(f"▶  Resume — {self.count} Q&A already saved")

    def write(self, doc: dict) -> bool:
        key = doc.get("url") or doc.get("question", "")[:100]
        if key in self.seen:
            return False
        answer = doc.get("answer", "")
        if len(answer) < MIN_ANSWER_LENGTH:
            return False
        self.seen.add(key)
        self._fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % 20 == 0:
            self._fh.flush()
        return True

    def close(self):
        self._fh.flush()
        self._fh.close()


def is_nutrition_related(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in NUTRITION_KEYWORDS)


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def get(url: str) -> Optional[requests.Response]:
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r
        except Exception as e:
            log.warning(f"  ⚠️  {attempt+1}/2 failed: {url} — {e}")
            time.sleep(2)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — VnExpress (static)
# ═══════════════════════════════════════════════════════════════════════════════

# All health Q&A categories on VnExpress
VNE_QA_CATEGORIES = [
    "https://vnexpress.net/suc-khoe/cac-benh/benh-dinh-duong/hoi-dap",
    "https://vnexpress.net/suc-khoe/cac-benh/hoi-dap",
    "https://vnexpress.net/suc-khoe/song-khoe/hoi-dap",
    "https://vnexpress.net/suc-khoe/cham-soc-nguoi-benh/hoi-dap",
]


def vne_parse_qa_page(url: str, base_cat: str) -> tuple[list[dict], Optional[str]]:
    """Parse one listing page. Returns (list of QA dicts, next_page_url)."""
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")
    results = []

    # Each Q&A lives inside div.list-answer > div.ask + div.fck_detail
    for container in soup.select("div.list-answer"):
        ask_el = container.select_one("div.ask")
        if not ask_el:
            continue
        qa_id = ask_el.get("id", "").replace("question_", "")

        # Prefer full question (content_more), fallback to short (content_less)
        q_full = ask_el.select_one(".content_more")
        q_short = ask_el.select_one(".content_less")
        q_el = q_full if q_full else q_short
        if not q_el:
            continue
        # Strip button text ("Ẩn bớt", "Xem thêm")
        for btn in q_el.select(".btn-view"):
            btn.decompose()
        question = q_el.get_text(separator=" ", strip=True).strip()

        # Answer block is sibling div.fck_detail
        ans_block = container.select_one("div.fck_detail")
        if not ans_block:
            continue
        ans_el = ans_block.select_one("div.answer")
        if not ans_el:
            continue

        doctor_el = ans_el.select_one(".name, .user_traloi")
        doctor = doctor_el.get_text(strip=True) if doctor_el else ""
        if doctor_el:
            doctor_el.decompose()
        answer = ans_el.get_text(separator="\n", strip=True)

        if not question or not answer:
            continue
        if not is_nutrition_related(question + " " + answer):
            continue

        qa_url = f"https://vnexpress.net/suc-khoe/hoi-dap/cau-hoi-{qa_id}" if qa_id else url
        results.append({
            "source":     "vnexpress",
            "url":        qa_url,
            "question":   clean_text(question),
            "answer":     clean_text(answer),
            "doctor":     doctor,
            "category":   "dinh-duong",
            "crawled_at": datetime.now().isoformat(),
        })

    # Next page: only follow if it stays within the same category base
    next_url = None
    next_el = soup.select_one("link[rel='next']")
    if next_el:
        candidate = next_el.get("href", "")
        # Must stay within same category path
        if base_cat.split("/hoi-dap")[0] in candidate or "/hoi-dap-p" in candidate:
            next_url = candidate

    return results, next_url


def crawl_vnexpress_qa(writer: QAWriter):
    log.info("=" * 55)
    log.info("▶  VnExpress Q&A")
    seen_pages = set()

    for cat_url in VNE_QA_CATEGORIES:
        page_url = cat_url
        page_num = 0
        base_cat = cat_url
        empty_streak = 0
        while page_url:
            if page_url in seen_pages:
                break
            seen_pages.add(page_url)
            page_num += 1

            qas, next_url = vne_parse_qa_page(page_url, base_cat)
            saved = sum(1 for qa in qas if writer.write(qa))
            log.info(f"  {page_url.split('vnexpress.net')[1]} → {len(qas)} nutrition Q&A, saved {saved}")

            if len(qas) == 0:
                empty_streak += 1
                if empty_streak >= 2:
                    log.info(f"  2 empty pages in a row — skipping rest of {cat_url.split('vnexpress.net')[1]}")
                    break
            else:
                empty_streak = 0

            page_url = next_url
            time.sleep(DELAY)

    log.info(f"  VnExpress done. Total Q&A so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — Vinmec (Playwright)
# ═══════════════════════════════════════════════════════════════════════════════

VINMEC_QA_BASE = "https://www.vinmec.com/vie/hoi-dap-bac-si/"
VINMEC_NUTRITION_FILTER = "https://www.vinmec.com/vie/hoi-dap-bac-si/?category=dinh-duong"


async def vinmec_get_qa_urls(page: Page, max_pages: int = 30) -> list[str]:
    """Scroll/paginate to collect Q&A article URLs."""
    urls = set()
    try:
        await page.goto(VINMEC_QA_BASE, wait_until="domcontentloaded", timeout=30_000)
    except PWTimeout:
        log.warning("  Vinmec base page timeout")
        return []

    for p in range(1, max_pages + 1):
        # Collect links matching Q&A pattern
        hrefs = await page.eval_on_selector_all("a[href*='/vie/hoi-dap-bac-si/']", "els => els.map(e => e.href)")
        before = len(urls)
        for href in hrefs:
            # Individual Q&A pages have a slug (not just the base category)
            if re.search(r"/vie/hoi-dap-bac-si/[^/]+-\d+", href):
                urls.add(href)

        # Try next page button or scroll
        next_btn = await page.query_selector("a.next, a[rel='next'], .pagination a.next, li.next a")
        if next_btn:
            await next_btn.click()
            await asyncio.sleep(1.5)
        else:
            # Try scrolling
            prev_h = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1.5)
            new_h = await page.evaluate("document.body.scrollHeight")
            if new_h == prev_h:
                break

        log.info(f"  Vinmec listing p{p}: +{len(urls)-before} URLs → total {len(urls)}")

    return list(urls)


async def vinmec_parse_qa(page: Page, url: str) -> Optional[dict]:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
    except PWTimeout:
        return None

    # Question
    question = ""
    for sel in [".question-content", ".ask-question", "h1.title", "h1"]:
        el = await page.query_selector(sel)
        if el:
            question = (await el.inner_text()).strip()
            break

    # Answer
    answer = ""
    doctor = ""
    for sel in [".answer-content", ".doctor-answer", ".fck_detail", ".answer"]:
        el = await page.query_selector(sel)
        if el:
            # Get doctor
            d_el = await el.query_selector(".doctor-name, .name, .author")
            if d_el:
                doctor = (await d_el.inner_text()).strip()
            answer = (await el.inner_text()).strip()
            break

    if not question or not answer or len(answer) < MIN_ANSWER_LENGTH:
        return None
    if not is_nutrition_related(question + " " + answer):
        return None

    return {
        "source":     "vinmec",
        "url":        url,
        "question":   clean_text(question),
        "answer":     clean_text(answer),
        "doctor":     doctor,
        "category":   "dinh-duong",
        "crawled_at": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — BV Thu Cúc (Playwright)
# ═══════════════════════════════════════════════════════════════════════════════

THUCUC_QA_BASE = "https://benhvienthucuc.vn/hoi-dap-chuyen-gia/dinh-duong"


async def thucuc_collect_urls(page: Page, max_clicks: int = 30) -> list[str]:
    urls = set()
    try:
        await page.goto(THUCUC_QA_BASE, wait_until="domcontentloaded", timeout=30_000)
    except PWTimeout:
        log.warning("  Thu Cuc page timeout")
        return []

    for click in range(max_clicks):
        hrefs = await page.eval_on_selector_all("a[href*='/hoi-dap-chuyen-gia/']", "els => els.map(e => e.href)")
        before = len(urls)
        for href in hrefs:
            # Individual Q&A — exclude listing/category pages
            if re.search(r"/hoi-dap-chuyen-gia/(?!dinh-duong/?$)[^/]+/?$", href):
                urls.add(href.rstrip("/"))

        # Try next/pagination
        next_btn = await page.query_selector(
            "a.next, a[rel='next'], .pagination a:has-text('Tiếp'), "
            "nav a:has-text('›'), .wp-pagenavi a.nextpostslink"
        )
        if not next_btn:
            break
        try:
            await next_btn.click()
            await asyncio.sleep(1.5)
            log.info(f"  Thu Cuc click {click+1}: +{len(urls)-before} → total {len(urls)}")
        except Exception:
            break

    log.info(f"  Thu Cuc collected {len(urls)} Q&A URLs")
    return list(urls)


async def thucuc_parse_qa(page: Page, url: str) -> Optional[dict]:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
    except PWTimeout:
        return None

    question = ""
    for sel in ["h1.entry-title", "h1.post-title", "h1"]:
        el = await page.query_selector(sel)
        if el:
            question = (await el.inner_text()).strip()
            break

    answer = ""
    doctor = ""
    for sel in [".entry-content", ".post-content", "article .content", "article"]:
        el = await page.query_selector(sel)
        if el:
            await el.evaluate("e => e.querySelectorAll('.sharedaddy,.related-posts,.post-tags').forEach(x=>x.remove())")
            d_el = await el.query_selector(".doctor-name, .author, strong:first-of-type")
            if d_el:
                doctor = (await d_el.inner_text()).strip()
            answer = (await el.inner_text()).strip()
            break

    if not question or not answer or len(answer) < MIN_ANSWER_LENGTH:
        return None

    return {
        "source":     "benhvienthucuc",
        "url":        url,
        "question":   clean_text(question),
        "answer":     clean_text(answer),
        "doctor":     doctor,
        "category":   "dinh-duong",
        "crawled_at": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    writer = QAWriter(OUTPUT_FILE)
    log.info("=" * 55)
    log.info("▶  crawl_qa.py — VnExpress + Vinmec + BV Thu Cúc")
    log.info(f"   Output: {OUTPUT_FILE}")
    log.info("=" * 55)

    # ── 1. VnExpress (static) ──────────────────────────────────────────────────
    crawl_vnexpress_qa(writer)

    # ── 2 & 3. Playwright sources ─────────────────────────────────────────────
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="vi-VN",
        )

        # ── Vinmec ──────────────────────────────────────────────────────────
        log.info("\n" + "=" * 55)
        log.info("▶  Vinmec Q&A")
        list_page = await context.new_page()
        vinmec_urls = await vinmec_get_qa_urls(list_page, max_pages=50)
        await list_page.close()
        log.info(f"  Collected {len(vinmec_urls)} Vinmec Q&A URLs")

        art_page = await context.new_page()
        for url in vinmec_urls:
            if url in writer.seen:
                continue
            doc = await vinmec_parse_qa(art_page, url)
            if doc and writer.write(doc):
                log.info(f"    ✅ [{writer.count}] {doc['question'][:65]}")
            await asyncio.sleep(0.5)
        await art_page.close()

        # ── BV Thu Cúc ──────────────────────────────────────────────────────
        log.info("\n" + "=" * 55)
        log.info("▶  BV Thu Cúc Q&A")
        list_page2 = await context.new_page()
        thucuc_urls = await thucuc_collect_urls(list_page2, max_clicks=50)
        await list_page2.close()
        log.info(f"  Collected {len(thucuc_urls)} Thu Cuc Q&A URLs")

        art_page2 = await context.new_page()
        for url in thucuc_urls:
            if url in writer.seen:
                continue
            doc = await thucuc_parse_qa(art_page2, url)
            if doc and writer.write(doc):
                log.info(f"    ✅ [{writer.count}] {doc['question'][:65]}")
            await asyncio.sleep(0.5)
        await art_page2.close()

        await browser.close()

    writer.close()
    log.info(f"\n{'='*55}")
    log.info(f"✅  DONE — {writer.count} Q&A saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
