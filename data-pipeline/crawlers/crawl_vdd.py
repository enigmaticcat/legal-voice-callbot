"""
crawl_vdd.py
============
Crawl viendinhduong.vn (Viện Dinh Dưỡng Quốc Gia) — static site, no Playwright needed.

~700 bài từ /vi/article/tin-tuc?page=1..27
Output: vdd_corpus.jsonl
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

BASE_URL    = "https://viendinhduong.vn"
LIST_URL    = "https://viendinhduong.vn/vi/article/tin-tuc"
OUTPUT_FILE = Path("/Users/nguyenthithutam/Desktop/Callbot/vdd_corpus.jsonl")
LOG_FILE    = Path("/Users/nguyenthithutam/Desktop/Callbot/crawl_vdd.log")

MIN_CONTENT_LENGTH  = 300
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
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r
        except Exception as e:
            log.warning(f"  ⚠️  Attempt {attempt+1}/2 failed: {url} — {e}")
            time.sleep(2)
    return None

# ─── Writer ───────────────────────────────────────────────────────────────────

class CorpusWriter:
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
                log.info(f"▶  Resume — {self.count} docs already in {self.path.name}")

    def write(self, doc: dict) -> bool:
        if doc["url"] in self.seen:
            return False
        if len(doc.get("content", "")) < MIN_CONTENT_LENGTH:
            return False
        self.seen.add(doc["url"])
        self._fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % 50 == 0:
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

def get_article_urls(page: int) -> list[str]:
    url = LIST_URL if page == 1 else f"{LIST_URL}?page={page}"
    r = get(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.select("a[href*='/vi/article/tin-tuc/']"):
        href = a.get("href", "")
        # Only article detail pages (have hex ID at end)
        if re.search(r"-[0-9a-f]{24}$", href):
            full = href if href.startswith("http") else BASE_URL + href
            urls.append(full)
    return list(dict.fromkeys(urls))  # deduplicate, preserve order

# ─── Article page → doc ───────────────────────────────────────────────────────

def parse_article(url: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title_el = soup.select_one("div.news-title")
    if not title_el:
        log.warning(f"    ❌ No title: {url}")
        return None
    title = title_el.get_text(strip=True)
    if not title:
        return None

    # Description (intro paragraph) + body
    parts = []
    desc = soup.select_one("div.news-description")
    if desc:
        parts.append(desc.get_text(separator="\n", strip=True))

    body = soup.select_one("div.news-details")
    if not body:
        log.warning(f"    ❌ No body: {url}")
        return None

    # Strip noise inside body
    for el in body.select("script, style, .new-share-fb, .news-views, figure.ads, .relate-news"):
        el.decompose()

    parts.append(body.get_text(separator="\n", strip=True))
    content = "\n\n".join(p for p in parts if p)

    # Author (append as attribution)
    author_el = soup.select_one("div.author-name")
    if author_el:
        author = author_el.get_text(strip=True)
        if author:
            content += f"\n\nTác giả: {author}"

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    return {
        "source":     "viendinhduong",
        "url":        url,
        "title":      title,
        "content":    clean_text(content),
        "category":   "dinh-duong",
        "crawled_at": datetime.now().isoformat(),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    writer = CorpusWriter(OUTPUT_FILE)
    log.info("=" * 60)
    log.info("▶  crawl_vdd.py — viendinhduong.vn")
    log.info(f"   Output: {OUTPUT_FILE}")
    log.info("=" * 60)

    # Detect total pages from page 1
    r0 = get(LIST_URL)
    total_pages = 1
    if r0:
        soup0 = BeautifulSoup(r0.text, "html.parser")
        # Look for last page number in pagination
        page_links = soup0.select("a[href*='?page=']")
        for a in page_links:
            m = re.search(r"\?page=(\d+)", a.get("href", ""))
            if m:
                total_pages = max(total_pages, int(m.group(1)))
    log.info(f"   Total pages detected: {total_pages}")

    for page in range(1, total_pages + 1):
        urls = get_article_urls(page)
        log.info(f"  Page {page}/{total_pages} → {len(urls)} articles")

        for url in urls:
            if url in writer.seen:
                continue
            doc = parse_article(url)
            if doc and writer.write(doc):
                log.info(f"    ✅ [{writer.count}] {doc['title'][:65]}")
            time.sleep(DELAY_BETWEEN_ARTS)

        time.sleep(DELAY_BETWEEN_PAGES)

    writer.close()
    log.info(f"\n{'='*60}")
    log.info(f"✅  DONE — {writer.count} docs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
