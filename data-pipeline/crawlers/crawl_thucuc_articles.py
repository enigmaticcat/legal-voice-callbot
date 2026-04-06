"""
crawl_thucuc_articles.py
========================
Crawler bài viết dinh dưỡng từ benhvienthucuc.vn/song-khoe/kham-dinh-duong

Cấu trúc site:
  Category: /song-khoe/kham-dinh-duong?page=N  (N = 1..~30, ~20 bài/trang)
  Article:  /song-khoe/kham-dinh-duong/<slug>
  Content:  <div class="singular-content">...</div>

Output: thucuc_articles_corpus.jsonl (nối vào nutrition_corpus.jsonl sau)
  {
    "source":     "benhvienthucuc",
    "url":        "https://benhvienthucuc.vn/...",
    "title":      "...",
    "content":    "...",
    "category":   "dinh-duong",
    "crawled_at": "2026-..."
  }

Chạy:
  pip install requests beautifulsoup4
  python crawl_thucuc_articles.py
  python crawl_thucuc_articles.py --max-pages 5   # test nhanh
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_URL     = "https://benhvienthucuc.vn"
CATEGORY_URL = "https://benhvienthucuc.vn/song-khoe/kham-dinh-duong"
SOURCE       = "benhvienthucuc"
CATEGORY     = "dinh-duong"

OUTPUT_FILE = Path(__file__).resolve().parent.parent.parent / "thucuc_articles_corpus.jsonl"
LOG_FILE    = Path(__file__).resolve().parent.parent.parent / "crawl_thucuc_articles.log"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

DELAY_PAGES    = 1.5   # seconds between category page requests
DELAY_ARTICLES = 0.8   # seconds between article requests
MIN_CONTENT    = 200   # skip articles shorter than this
SAVE_EVERY     = 20    # flush to disk every N articles

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── HTTP helper ─────────────────────────────────────────────────────────────

def get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r
        except Exception as e:
            log.warning(f"  Attempt {attempt+1}/3 failed {url}: {e}")
            time.sleep(2)
    return None


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# ─── Writer (resume-safe) ─────────────────────────────────────────────────────

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
            log.info(f"Resume: {self.count} docs already in {self.path.name}")

    def write(self, doc: dict) -> bool:
        if doc["url"] in self.seen:
            return False
        if len(doc["content"]) < MIN_CONTENT:
            log.debug(f"  Skip (too short): {doc['url']}")
            return False
        self.seen.add(doc["url"])
        self._fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % SAVE_EVERY == 0:
            self._fh.flush()
        return True

    def close(self):
        self._fh.flush()
        self._fh.close()

# ─── Crawler logic ────────────────────────────────────────────────────────────

def get_article_slugs(page: int) -> List[str]:
    """
    Fetch category page N and return all article relative URLs.
    Pattern: /song-khoe/kham-dinh-duong/<slug>  (slug ≥ 10 chars, only a-z0-9-)
    """
    url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}?page={page}"
    r = get(url)
    if not r:
        return []
    slugs = re.findall(
        r'href="(/song-khoe/kham-dinh-duong/[a-z0-9][a-z0-9-]{9,})"',
        r.text,
    )
    return list(dict.fromkeys(slugs))  # deduplicate, preserve order


def parse_article(article_url: str) -> Optional[dict]:
    """
    Fetch and parse one article.
    Returns corpus doc or None if content too short / fetch failed.
    """
    r = get(article_url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""
    if not title:
        return None

    # Content — singular-content div
    content_div = soup.find(class_="singular-content")
    if not content_div:
        log.debug(f"  No singular-content: {article_url}")
        return None

    # Remove unwanted nested blocks (related articles, ads, author boxes)
    for tag in content_div.find_all(
        class_=re.compile(r"related|suggest|ads|author|social|share|comment", re.I)
    ):
        tag.decompose()

    content = clean_text(content_div.get_text(separator="\n"))

    if len(content) < MIN_CONTENT:
        return None

    return {
        "source":     SOURCE,
        "url":        article_url,
        "title":      title,
        "content":    content,
        "category":   CATEGORY,
        "crawled_at": datetime.now().isoformat(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=35,
                        help="Max category pages to crawl (default 35, ~700 articles)")
    parser.add_argument("--out", default=str(OUTPUT_FILE),
                        help="Output JSONL file path")
    args = parser.parse_args()

    out_path = Path(args.out)
    writer   = CorpusWriter(out_path)

    log.info(f"Starting crawl: {CATEGORY_URL}")
    log.info(f"Output: {out_path}")

    total_written = 0

    for page in range(1, args.max_pages + 1):
        log.info(f"── Page {page}/{args.max_pages} ──")
        slugs = get_article_slugs(page)

        if not slugs:
            log.info(f"  No articles found on page {page}, stopping.")
            break

        log.info(f"  Found {len(slugs)} articles")

        for slug in slugs:
            article_url = urljoin(BASE_URL, slug)

            if article_url in writer.seen:
                log.debug(f"  Skip (seen): {article_url}")
                continue

            doc = parse_article(article_url)
            if doc and writer.write(doc):
                total_written += 1
                log.info(f"  [{total_written}] {doc['title'][:60]}")
            else:
                log.debug(f"  Skip: {article_url}")

            time.sleep(DELAY_ARTICLES)

        time.sleep(DELAY_PAGES)

    writer.close()
    log.info(f"Done. Written {total_written} new articles → {out_path}")
    log.info(f"Total in file (incl. resumed): {writer.count}")


if __name__ == "__main__":
    main()
