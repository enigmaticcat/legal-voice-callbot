"""
crawl_thucuc_qa.py
==================
Crawl Q&A dinh dưỡng từ benhvienthucuc.vn (TCI Hospital).
Static site (Next.js SSR) — không cần Playwright.

URL pattern:
  Listing : https://benhvienthucuc.vn/hoi-dap-chuyen-gia/dinh-duong?page=N
  Article  : https://benhvienthucuc.vn/hoi-dap-chuyen-gia/dinh-duong/{slug}

Output: thucuc_qa.jsonl
  { source, url, question, answer, category, crawled_at }
"""

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

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL    = "https://benhvienthucuc.vn"
LIST_BASE   = "https://benhvienthucuc.vn/hoi-dap-chuyen-gia/dinh-duong"
OUTPUT_FILE = Path("/Users/nguyenthithutam/Desktop/Callbot/thucuc_qa.jsonl")
LOG_FILE    = Path("/Users/nguyenthithutam/Desktop/Callbot/crawl_thucuc.log")

MIN_ANSWER_LENGTH   = 200
DELAY_BETWEEN_PAGES = 1.5
DELAY_BETWEEN_ARTS  = 0.8

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

# ─── HTTP ─────────────────────────────────────────────────────────────────────

def get(url: str) -> Optional[requests.Response]:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r
        except Exception as e:
            log.warning(f"  ⚠️  Attempt {attempt+1}/3 failed: {url} — {e}")
            time.sleep(2)
    return None

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
                        self.seen.add(doc["url"])
                        self.count += 1
                    except Exception:
                        pass
            if self.count:
                log.info(f"▶  Resume — {self.count} Q&A already in {self.path.name}")

    def write(self, doc: dict) -> bool:
        if doc["url"] in self.seen:
            return False
        if len(doc.get("answer", "")) < MIN_ANSWER_LENGTH:
            return False
        self.seen.add(doc["url"])
        self._fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % 20 == 0:
            self._fh.flush()
        return True

    def close(self):
        self._fh.flush()
        self._fh.close()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# ─── Listing page → article URLs ──────────────────────────────────────────────

# Article URL: exactly 3 path segments ending with a non-empty slug
ARTICLE_RE = re.compile(
    r"^https?://benhvienthucuc\.vn/hoi-dap-chuyen-gia/dinh-duong/[^/?#]+$"
)

def get_article_urls(page: int) -> list[str]:
    url = LIST_BASE if page == 1 else f"{LIST_BASE}?page={page}"
    r = get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Make absolute
        if href.startswith("/"):
            href = BASE_URL + href
        if ARTICLE_RE.match(href.rstrip("/")):
            urls.append(href.rstrip("/"))
    return list(dict.fromkeys(urls))  # deduplicate, preserve order

# ─── Article page → Q&A doc ───────────────────────────────────────────────────

def parse_article(url: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    # Title (short form) = h1 inside the gray question box
    gray_box = soup.select_one("div.bg-gray-100.rounded-lg")
    if not gray_box:
        log.warning(f"  ❌ No question box: {url}")
        return None

    h1 = gray_box.select_one("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Patient question (human-style text) = <p> inside the gray box
    q_para = gray_box.select_one("p")
    question = q_para.get_text(separator="\n", strip=True) if q_para else title
    if not question:
        return None

    # Answer = singular-content div
    content_div = soup.select_one("div.singular-content")
    if not content_div:
        content_div = soup.select_one("article .entry-content, .post-content")
    if not content_div:
        log.warning(f"  ❌ No answer content: {url}")
        return None

    # Strip noise inside answer
    for el in content_div.select(
        "script, style, .sharedaddy, .related-posts, .post-tags, "
        ".wp-block-buttons, nav, .breadcrumb, .comment-section, "
        "#comments, .sidebar, footer"
    ):
        el.decompose()

    answer = content_div.get_text(separator="\n", strip=True)
    answer = clean_text(answer)

    if len(answer) < MIN_ANSWER_LENGTH:
        log.debug(f"  ⏭  Too short ({len(answer)}c): {url}")
        return None

    return {
        "source":     "benhvienthucuc",
        "url":        url,
        "title":      title,
        "question":   question,
        "answer":     answer,
        "category":   "dinh-duong",
        "crawled_at": datetime.now().isoformat(),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def detect_total_pages() -> int:
    r = get(LIST_BASE)
    if not r:
        return 13  # fallback from known value
    soup = BeautifulSoup(r.text, "html.parser")
    max_page = 1
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]page=(\d+)", a["href"])
        if m:
            max_page = max(max_page, int(m.group(1)))
    return max_page


def main():
    writer = QAWriter(OUTPUT_FILE)
    log.info("=" * 60)
    log.info("▶  crawl_thucuc_qa.py — benhvienthucuc.vn")
    log.info(f"   Output: {OUTPUT_FILE}")
    log.info("=" * 60)

    total_pages = detect_total_pages()
    log.info(f"   Total pages detected: {total_pages}")

    for page in range(1, total_pages + 1):
        urls = get_article_urls(page)
        log.info(f"  Page {page}/{total_pages} → {len(urls)} articles")

        for url in urls:
            if url in writer.seen:
                continue
            doc = parse_article(url)
            if doc and writer.write(doc):
                log.info(f"    ✅ [{writer.count}] {doc['question'][:65]}")
            time.sleep(DELAY_BETWEEN_ARTS)

        time.sleep(DELAY_BETWEEN_PAGES)

    writer.close()
    log.info(f"\n{'='*60}")
    log.info(f"✅  DONE — {writer.count} Q&A docs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
