"""
crawl_skds_playwright.py
========================
Crawl suckhoedoisong.vn/dinh-duong bằng Playwright (JS-rendered).

Lý do cần Playwright:
  - Trang dùng infinite scroll / nút "Xem thêm bài viết"
  - Nội dung bài viết render bằng JS
  - Static requests chỉ trả về HTML framework trống

Output: skds_corpus.jsonl  (tách riêng khỏi nutrition_corpus.jsonl)

Cài đặt (chỉ cần 1 lần):
  pip install playwright
  playwright install chromium

Chạy:
  python3 crawl_skds_playwright.py
"""

import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path("/Users/nguyenthithutam/Desktop/Callbot/skds_corpus.jsonl")
LOG_FILE    = Path("/Users/nguyenthithutam/Desktop/Callbot/crawl_skds.log")

MIN_CONTENT_LENGTH = 300   # chars
MAX_CLICKS         = 80    # "Xem thêm" clicks per category (~40 articles/click → ~3200/cat)
ARTICLE_TIMEOUT    = 20_000  # ms
SCROLL_PAUSE       = 1.5   # seconds between scroll/click actions

SKDS_CATEGORIES = [
    ("https://suckhoedoisong.vn/dinh-duong.htm",                          "dinh-duong"),
    ("https://suckhoedoisong.vn/dinh-duong/dinh-duong-me-va-be.htm",      "me-va-be"),
    ("https://suckhoedoisong.vn/dinh-duong/dinh-duong-nguoi-cao-tuoi.htm","nguoi-cao-tuoi"),
    ("https://suckhoedoisong.vn/dinh-duong/dinh-duong-hoc-duong.htm",     "hoc-duong"),
    ("https://suckhoedoisong.vn/dinh-duong/che-do-an-nguoi-benh.htm",     "nguoi-benh"),
    ("https://suckhoedoisong.vn/dinh-duong/canh-giac-thuc-pham.htm",      "canh-giac"),
    ("https://suckhoedoisong.vn/dinh-duong/thuc-pham-chuc-nang.htm",      "thuc-pham-chuc-nang"),
]

ARTICLE_URL_RE = re.compile(r"-169\d{15,18}\.htm$")

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


def make_doc(url: str, title: str, content: str, category: str) -> dict:
    return {
        "source":     "suckhoedoisong",
        "url":        url,
        "title":      title.strip(),
        "content":    clean_text(content),
        "category":   category,
        "crawled_at": datetime.now().isoformat(),
    }


# ─── Category page: collect article URLs ──────────────────────────────────────

async def collect_article_urls(page: Page, cat_url: str) -> list[str]:
    """
    Navigate to category page, then repeatedly click 'Xem thêm bài viết'
    (or scroll) to load all articles. Returns deduplicated article URLs.
    """
    log.info(f"  → Loading category: {cat_url}")
    try:
        await page.goto(cat_url, wait_until="domcontentloaded", timeout=30_000)
    except PWTimeout:
        log.warning(f"  ⚠️  Timeout loading {cat_url}")
        return []

    urls: set[str] = set()
    clicks = 0

    while clicks < MAX_CLICKS:
        # Collect all article links currently on page
        hrefs = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href)"
        )
        before = len(urls)
        for href in hrefs:
            if ARTICLE_URL_RE.search(href):
                urls.add(href)
        new_found = len(urls) - before

        # Try to click "Xem thêm" button
        btn = await page.query_selector(
            "a.list__viewmore, button.list__viewmore, "
            "a.xem-them, .viewmore a, a:has-text('Xem thêm bài')"
        )
        if not btn:
            # Try scrolling to trigger infinite scroll
            prev_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(SCROLL_PAUSE)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                log.info(f"    No more content after {clicks} clicks/scrolls, {len(urls)} URLs")
                break
            clicks += 1
            continue

        try:
            await btn.scroll_into_view_if_needed()
            await btn.click()
            await asyncio.sleep(SCROLL_PAUSE)
            clicks += 1
            log.info(f"    Click {clicks}: +{new_found} → total {len(urls)} URLs")
        except Exception as e:
            log.warning(f"    Click failed: {e}")
            break

    log.info(f"  Category done: {len(urls)} article URLs collected")
    return list(urls)


# ─── Article page: parse content ──────────────────────────────────────────────

async def parse_article(page: Page, url: str, category: str) -> Optional[dict]:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=ARTICLE_TIMEOUT)
    except PWTimeout:
        log.warning(f"      ⏱  Timeout: {url}")
        return None
    except Exception as e:
        log.warning(f"      ❌ Error loading {url}: {e}")
        return None

    # Title
    title = ""
    for sel in ["h1.detail-title", "h1.title-detail", "h1"]:
        el = await page.query_selector(sel)
        if el:
            title = (await el.inner_text()).strip()
            break
    if not title:
        return None

    # Body — remove noise elements first
    await page.evaluate(
        "document.querySelectorAll('script,style,.box-tinlienquan,.relate-news,.box-comment,.ads,iframe,.share-icon,.author-block,.tags-news,figure.ads').forEach(el => el.remove())"
    )

    content = ""
    for sel in [".detail-content", ".detail__content", ".detail-body", "article .content"]:
        el = await page.query_selector(sel)
        if el:
            content = (await el.inner_text()).strip()
            break

    if not content or len(content) < MIN_CONTENT_LENGTH:
        return None

    return make_doc(url, title, content, category)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    writer = CorpusWriter(OUTPUT_FILE)
    log.info("=" * 60)
    log.info("▶  crawl_skds_playwright.py — suckhoedoisong.vn")
    log.info(f"   Output: {OUTPUT_FILE}")
    log.info("=" * 60)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="vi-VN",
        )

        for cat_url, cat_name in SKDS_CATEGORIES:
            log.info(f"\n{'='*50}")
            log.info(f"Category: {cat_name}")

            # Step 1: collect all article URLs from category page
            cat_page = await context.new_page()
            article_urls = await collect_article_urls(cat_page, cat_url)
            await cat_page.close()

            # Step 2: parse each article
            art_page = await context.new_page()
            for i, url in enumerate(article_urls, 1):
                if url in writer.seen:
                    continue
                doc = await parse_article(art_page, url, cat_name)
                if doc and writer.write(doc):
                    log.info(f"      ✅ [{writer.count}] {doc['title'][:65]}")
                elif doc is None:
                    log.debug(f"      skip {url}")
                await asyncio.sleep(0.5)

                # Progress every 100
                if i % 100 == 0:
                    log.info(f"    Progress: {i}/{len(article_urls)} — corpus: {writer.count}")

            await art_page.close()
            log.info(f"  Category '{cat_name}' done. Corpus total: {writer.count}")

        await browser.close()

    writer.close()
    log.info(f"\n{'='*60}")
    log.info(f"✅  DONE — {writer.count} docs saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
