"""
crawl_nutrition.py
==================
Multi-source Vietnamese nutrition crawler for RAG corpus.

Sources:
  Tier 1:
    - suckhoedoisong.vn/dinh-duong      (~5,000 bài)
    - vnexpress.net/suc-khoe/dinh-duong (~3,000 bài)
  Tier 2:
    - hellobacsi.com/dinh-duong         (~1,500 bài)
    - viendinhduong.vn                  (~2,000 bài)

Output: nutrition_corpus.jsonl (one JSON doc per line)
  {
    "id":      "skds_001",
    "source":  "suckhoedoisong",
    "url":     "https://...",
    "title":   "...",
    "content": "...",   # full article text, cleaned
    "category": "...",  # sub-category if available
    "crawled_at": "2026-03-25T..."
  }
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import sys
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Optional, Tuple, List

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path("/Users/nguyenthithutam/Desktop/Callbot/nutrition_corpus.jsonl")
LOG_FILE    = Path("/Users/nguyenthithutam/Desktop/Callbot/crawl_nutrition.log")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

DELAY_BETWEEN_PAGES    = 1.5   # seconds
DELAY_BETWEEN_ARTICLES = 0.8   # seconds
MIN_CONTENT_LENGTH     = 200   # chars — skip stub pages
SAVE_EVERY             = 20    # flush to disk every N articles

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

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get(url: str, timeout: int = 10) -> Optional[requests.Response]:
    """Safe GET with retry (2 attempts, 10s timeout)."""
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r
        except Exception as e:
            log.warning(f"  ⚠️  Attempt {attempt+1}/2 failed for {url}: {e}")
            time.sleep(2)
    return None


def clean_text(text: str) -> str:
    """Remove excessive whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def make_doc(source: str, url: str, title: str, content: str, category: str = "") -> dict:
    return {
        "source":     source,
        "url":        url,
        "title":      title.strip(),
        "content":    clean_text(content),
        "category":   category.strip(),
        "crawled_at": datetime.now().isoformat(),
    }


# ─── Writer ──────────────────────────────────────────────────────────────────

class CorpusWriter:
    def __init__(self, path: Path):
        self.path   = path
        self.count  = 0
        self.seen   = set()     # deduplicate by URL
        self._load_existing()
        self._fh    = open(path, "a", encoding="utf-8")

    def _load_existing(self):
        """Resume: skip already-crawled URLs."""
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        self.seen.add(doc["url"])
                        self.count += 1
                    except Exception:
                        pass
            log.info(f"▶  Resuming — {self.count} docs already in {self.path.name}")

    def write(self, doc: dict) -> bool:
        """Returns True if written, False if duplicate/too short."""
        if doc["url"] in self.seen:
            return False
        if len(doc["content"]) < MIN_CONTENT_LENGTH:
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


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — suckhoedoisong.vn
# ═══════════════════════════════════════════════════════════════════════════════

SKDS_CATEGORIES = [
    ("https://suckhoedoisong.vn/dinh-duong.htm",                     "dinh-duong"),
    ("https://suckhoedoisong.vn/dinh-duong/dinh-duong-me-va-be.htm", "me-va-be"),
    ("https://suckhoedoisong.vn/dinh-duong/dinh-duong-nguoi-cao-tuoi.htm", "nguoi-cao-tuoi"),
    ("https://suckhoedoisong.vn/dinh-duong/dinh-duong-hoc-duong.htm","hoc-duong"),
    ("https://suckhoedoisong.vn/dinh-duong/che-do-an-nguoi-benh.htm","nguoi-benh"),
    ("https://suckhoedoisong.vn/dinh-duong/canh-giac-thuc-pham.htm", "canh-giac"),
    ("https://suckhoedoisong.vn/dinh-duong/thuc-pham-chuc-nang.htm", "thuc-pham-chuc-nang"),
]



def skds_get_links_from_page(url: str) -> Tuple[List[str], Optional[str]]:
    """Returns (list_of_article_urls, next_page_url or None).
    
    SKDS uses .box-category-link-title for article links.
    URL pattern: /slug-169XXXXXXXXXXXXXXXXXX.htm (18-char numeric ID)
    """
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    # Primary selector confirmed via browser inspection
    # Using both .detail-title based variants and regex fallback
    for a in soup.select(".box-category-link-title, .box-home-focus-link-title, .box-category-link-with-avatar"):
        href = a.get("href", "")
        if re.search(r"-169\d{15,18}\.htm", href):
            full = urljoin("https://suckhoedoisong.vn", href)
            links.append(full)

    # Fallback to any <a> with matching pattern
    for a in soup.find_all("a", href=re.compile(r"-169\d{15,18}\.htm")):
        href = a.get("href", "")
        full = urljoin("https://suckhoedoisong.vn", href)
        links.append(full)

    next_url = None
    # SKDS often has a "SEE MORE" button or /p2 suffix
    next_btn = soup.select_one("a.next, a[rel='next'], .pagination .next a, a.list__viewmore")
    if next_btn:
        href = next_btn.get("href", "")
        if href and href != "#" and "javascript" not in href:
            next_url = urljoin("https://suckhoedoisong.vn", href)

    return list(dict.fromkeys(links)), next_url


def skds_parse_article(url: str, category: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    # Title selector fixed to .detail-title (single hyphen)
    title_el = soup.select_one("h1.detail-title, .detail-title, h1")
    if not title_el:
        log.warning(f"      ❌ Title not found for {url}")
        return None
    title = title_el.get_text(strip=True)

    # Main body fixed to .detail-content (single hyphen)
    body = soup.select_one(".detail-content, .detail__content, .detail-body, article")
    if not body:
        log.warning(f"      ❌ Body not found for {url}")
        return None

    # Remove unwanted tags
    for el in body.select("script, style, .box-tinlienquan, .ads, iframe, figure.ads, .relate-news, .box-comment"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)
    return make_doc("suckhoedoisong", url, title, content, category)


def crawl_suckhoedoisong(writer: CorpusWriter, max_pages_per_cat: int = 50):
    log.info("=" * 60)
    log.info("▶  Crawling suckhoedoisong.vn ...")

    for cat_url, cat_name in SKDS_CATEGORIES:
        log.info(f"  Category: {cat_name}")
        page_url = cat_url
        page_num = 0

        while page_url and page_num < max_pages_per_cat:
            page_num += 1
            links, next_url = skds_get_links_from_page(page_url)
            log.info(f"    Page {page_num} → {len(links)} links")

            for art_url in links:
                if art_url in writer.seen:
                    continue
                doc = skds_parse_article(art_url, cat_name)
                if doc and writer.write(doc):
                    log.info(f"      ✅ [{writer.count}] {doc['title'][:60]}")
                time.sleep(DELAY_BETWEEN_ARTICLES)

            # Stop if no links found and no explicit next page
            if not links and not next_url:
                log.info(f"    No more content for category {cat_name}. Stopping.")
                break

            # If no explicit next page found, try /pN suffix pattern
            if not next_url and links and page_num < max_pages_per_cat:
                base = cat_url.replace(".htm", "")
                candidate = f"{base}/p{page_num + 1}.htm"
                r_test = get(candidate)
                if r_test and r_test.status_code == 200 and len(r_test.content) > 10000:
                    next_url = candidate
                else:
                    break  # No more pages

            page_url = next_url
            time.sleep(DELAY_BETWEEN_PAGES)

    log.info(f"  suckhoedoisong done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — vnexpress.net
# ═══════════════════════════════════════════════════════════════════════════════

VNE_CATEGORIES = [
    ("https://vnexpress.net/suc-khoe/dinh-duong",          "dinh-duong"),
    ("https://vnexpress.net/suc-khoe/song-khoe",           "song-khoe"),
]

NUTRITION_KEYWORDS = [
    "ăn", "thực phẩm", "dinh dưỡng", "vitamin", "protein", "calo",
    "chất xơ", "khoáng chất", "omega", "béo", "tinh bột", "đường",
    "rau", "trái cây", "hoa quả", "uống", "chế độ ăn", "bữa ăn",
    "sữa", "trứng", "thịt", "cá", "tôm", "cua", "đậu", "hạt",
]


def vne_get_links_from_page(url: str) -> Tuple[List[str], Optional[str]]:
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    # Be more specific to avoid picking up the same link multiple times from different blocks
    for a in soup.select("article h3.title-news a[href], .item-news h3 a[href], h2.title-news a[href]"):
        href = a.get("href", "")
        if re.match(r"https://vnexpress\.net/[^/]+-\d+\.html", href):
            # Exclude non-article pages like tags or categories
            if not any(x in href for x in ["/tag-", "/topic-", "/video/"]):
                links.append(href)

    next_url = None
    next_btn = soup.select_one("a.next-page, a[rel='next'], .pagination .next")
    if next_btn:
        href = next_btn.get("href", "")
        if href:
            next_url = urljoin("https://vnexpress.net", href)

    return list(dict.fromkeys(links)), next_url


def vne_is_nutrition_related(title: str, content: str) -> bool:
    combined = (title + " " + content[:500]).lower()
    return any(kw in combined for kw in NUTRITION_KEYWORDS)


def vne_parse_article(url: str, category: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1.title-detail, h1")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    body = soup.select_one(".fck_detail, article.fck_detail, .sidebar-1")
    if not body:
        return None
    for el in body.select("script, style, .box-morelink, .item_slide_show"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)

    if not vne_is_nutrition_related(title, content):
        return None

    return make_doc("vnexpress", url, title, content, category)


def crawl_vnexpress(writer: CorpusWriter, max_pages_per_cat: int = 50):
    log.info("=" * 60)
    log.info("▶  Crawling vnexpress.net ...")

    for cat_url, cat_name in VNE_CATEGORIES:
        log.info(f"  Category: {cat_name}")
        page_url = cat_url
        page_num = 0

        while page_url and page_num < max_pages_per_cat:
            page_num += 1
            links, next_url = vne_get_links_from_page(page_url)
            log.info(f"    Page {page_num} → {len(links)} links")

            for art_url in links:
                if art_url in writer.seen:
                    continue
                doc = vne_parse_article(art_url, cat_name)
                if doc and writer.write(doc):
                    log.info(f"      ✅ [{writer.count}] {doc['title'][:60]}")
                time.sleep(DELAY_BETWEEN_ARTICLES)

            page_url = next_url
            time.sleep(DELAY_BETWEEN_PAGES)

    log.info(f"  vnexpress done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — hellobacsi.com
# ═══════════════════════════════════════════════════════════════════════════════

HB_CATEGORIES = [
    # HB articles
    ("https://hellobacsi.com/an-uong-lanh-manh/",           "dinh-duong"),
]


def hb_get_links_from_page(url: str) -> Tuple[List[str], Optional[str]]:
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("a.HgbW6-m, article h2 a, article h3 a, a[href*='/']"):
        href = a.get("href", "")
        if not href or href.startswith(("#", "javascript:")): continue
        full = urljoin("https://hellobacsi.com", href)
        # Article URLs often have depth 3+ (/cat/subcat/slug)
        parts = [p for p in full.split("/") if p]
        if full.startswith("https://hellobacsi.com") and len(parts) >= 4:
            if not any(k in full for k in ["/categories", "/videos", "/bot", "/spotlight", "/health-tools"]):
                links.append(full)

    next_url = None
    # Check for ?page=N pattern
    if "?" in url:
        base, query = url.split("?", 1)
        if "page=" in query:
            p_match = re.search(r"page=(\d+)", query)
            if p_match:
                next_page = int(p_match.group(1)) + 1
                next_url = f"{base}?page={next_page}"
        else:
            next_url = f"{url}&page=2"
    else:
        next_url = f"{url}?page=2"

    return list(dict.fromkeys(links)), next_url


def hb_parse_article(url: str, category: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1")
    if not title_el:
        log.warning(f"      ❌ HB Title not found: {url}")
        return None
    title = title_el.get_text(strip=True)

    # Selector div.css-1r0x5v6 found via browser (Mantine)
    body = soup.select_one("div.css-1r0x5v6, .entry-content, article")
    if not body:
        log.warning(f"      ❌ HB Body not found: {url}")
        return None

    # Exclusions
    for el in body.select("aside, header, footer, nav, .mantine-Breadcrumbs-root, .css-1y0xxz8, .cart-care-button"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)
    
    # Filter out short/junk content (minigames, communities)
    if len(content) < 800:
        return None
        
    return make_doc("hellobacsi", url, title, content, category)


def crawl_hellobacsi(writer: CorpusWriter, max_pages_per_cat: int = 30):
    log.info("=" * 60)
    log.info("▶  Crawling hellobacsi.com ...")

    for cat_url, cat_name in HB_CATEGORIES:
        log.info(f"  Category: {cat_name}")
        page_url = cat_url
        page_num = 0

        while page_url and page_num < max_pages_per_cat:
            page_num += 1
            links, next_url = hb_get_links_from_page(page_url)
            log.info(f"    Page {page_num} → {len(links)} links")

            for art_url in links:
                if art_url in writer.seen:
                    continue
                doc = hb_parse_article(art_url, cat_name)
                if doc and writer.write(doc):
                    log.info(f"      ✅ [{writer.count}] {doc['title'][:60]}")
                time.sleep(DELAY_BETWEEN_ARTICLES)

            page_url = next_url
            time.sleep(DELAY_BETWEEN_PAGES)

    log.info(f"  hellobacsi done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — viendinhduong.vn
# ═══════════════════════════════════════════════════════════════════════════════

VDD_CATEGORIES = [
    ("https://viendinhduong.vn/vi/article/tin-tuc",         "tin-tuc"),
    ("https://viendinhduong.vn/vi/article/hoi-dap",         "hoi-dap"),
    ("https://viendinhduong.vn/vi/article/kien-thuc",       "kien-thuc"),
]


def vdd_get_links_from_page(url: str) -> Tuple[List[str], Optional[str]]:
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    # Pattern found via browser: /vi/article/ category / article-slug
    for a in soup.select("a[href*='/vi/article/']"):
        href = a.get("href", "")
        full = urljoin("https://viendinhduong.vn", href)
        if len(full) > 50: # Avoid general category links
            links.append(full)

    next_url = None
    # Check for ?page=N pattern
    if "page=" in url:
        p_match = re.search(r"page=(\d+)", url)
        if p_match:
            next_page = int(p_match.group(1)) + 1
            next_url = re.sub(r"page=\d+", f"page={next_page}", url)
    else:
        sep = "&" if "?" in url else "?"
        next_url = f"{url}{sep}page=2"

    return list(dict.fromkeys(links)), next_url


def vdd_parse_article(url: str, category: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1, .article-title, .post-title")
    if not title_el:
        log.warning(f"      ❌ VDD Title not found: {url}")
        return None
    title = title_el.get_text(strip=True)

    # Selector .article-content found via browser
    body = soup.select_one(".article-content, #main-content, main")
    if not body:
        log.warning(f"      ❌ VDD Body not found: {url}")
        return None

    # Exclusions
    for el in body.select(".sidebar, .breadcrumb, .toolbar-detail, .meta-detail"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)
    return make_doc("viendinhduong", url, title, content, category)


def crawl_viendinhduong(writer: CorpusWriter, max_pages_per_cat: int = 50):
    log.info("=" * 60)
    log.info("▶  Crawling viendinhduong.vn ...")

    for cat_url, cat_name in VDD_CATEGORIES:
        log.info(f"  Category: {cat_name}")
        page_url = cat_url
        page_num = 0

        while page_url and page_num < max_pages_per_cat:
            page_num += 1
            links, next_url = vdd_get_links_from_page(page_url)
            log.info(f"    Page {page_num} → {len(links)} links")

            for art_url in links:
                if art_url in writer.seen:
                    continue
                doc = vdd_parse_article(art_url, cat_name)
                if doc and writer.write(doc):
                    log.info(f"      ✅ [{writer.count}] {doc['title'][:60]}")
                time.sleep(DELAY_BETWEEN_ARTICLES)

            page_url = next_url
            time.sleep(DELAY_BETWEEN_PAGES)

    log.info(f"  viendinhduong done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5 — vinmec.com
# Category listing: /vie/dinh-duong/          (228 articles, 4 pages)
# Pagination:       /vie/dinh-duong/page_2    (no trailing slash)
# Article links:    /vie/bai-viet/<slug>      (confirmed via browser)
# Article selectors: h1.single-title, #main-article
# ═══════════════════════════════════════════════════════════════════════════════

VINMEC_CATEGORIES = [
    ("https://www.vinmec.com/vie/dinh-duong/", "dinh-duong"),
]


def vinmec_get_links(url: str) -> Tuple[List[str], Optional[str]]:
    """Collect /vie/bai-viet/<slug> links from a /vie/dinh-duong/page_N listing."""
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select(".mini-post a[href], .col-3 a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        full = urljoin("https://www.vinmec.com", href)
        # Only article detail pages: /vie/bai-viet/<slug>  (no further sub-path)
        if re.match(r"https://www\.vinmec\.com/vie/bai-viet/[^/]+$", full):
            links.append(full)

    next_url = None
    # Pattern: /vie/dinh-duong/  → /vie/dinh-duong/page_2  → /vie/dinh-duong/page_3
    m = re.search(r"/page_(\d+)$", url.rstrip("/"))
    if m:
        next_page = int(m.group(1)) + 1
        next_url = re.sub(r"/page_\d+$", f"/page_{next_page}", url.rstrip("/"))
    else:
        next_url = url.rstrip("/") + "/page_2"

    return list(dict.fromkeys(links)), next_url


def vinmec_parse_article(url: str, category: str) -> Optional[dict]:
    # Skip quiz/interactive pages — no article content
    if "trac-nghiem" in url:
        return None

    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1.single-title, h1")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    body = soup.select_one("#main-article")
    if not body:
        return None

    # Remove nav, ads, CTA, related blocks, TOC toggle
    for el in body.select(
        "script, style, aside, .ads, .social-share, .box-tinlienquan, "
        ".related-posts, .breadcrumb, .block-share, .article-footer, "
        ".muc-luc, nav, .nav, [class*='related'], [class*='booking'], "
        "[class*='hotline'], [class*='appointment']"
    ):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)

    # Strip leading nav noise (☰ / Mục lục lines at top)
    content = re.sub(r"^(☰\s*\n?|Mục lục\s*\n?)+", "", content).strip()

    # Strip trailing CTA block (Đặt lịch / HOTLINE / MyVinmec)
    content = re.sub(
        r"\nĐể đặt lịch khám.*?(MyVinmec|ứng dụng)[\s\S]*$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    return make_doc("vinmec", url, title, content, category)


def crawl_vinmec(writer: CorpusWriter, max_pages_per_cat: int = 10):
    log.info("=" * 60)
    log.info("▶  Crawling vinmec.com /vie/dinh-duong/ ...")
    for cat_url, cat_name in VINMEC_CATEGORIES:
        _crawl_generic(writer, cat_url, cat_name,
                       get_links_fn=vinmec_get_links,
                       parse_fn=vinmec_parse_article,
                       max_pages=max_pages_per_cat)
    log.info(f"  vinmec done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 6 — nutrihome.vn
# Pagination: /tin-tuc/page/2/  (115 pages confirmed)
# Article links: /<slug>/ (root-level slugs, not under /tin-tuc/)
# ═══════════════════════════════════════════════════════════════════════════════

NUTRIHOME_BASE = "https://nutrihome.vn"
NUTRIHOME_CATEGORIES = [
    ("https://nutrihome.vn/tin-tuc/",               "tin-tuc"),
    ("https://nutrihome.vn/tag/dinh-duong/",         "dinh-duong"),
]


def nutrihome_get_links(url: str) -> Tuple[List[str], Optional[str]]:
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("h2 a[href], h3 a[href], .entry-title a[href], .post-title a[href]"):
        href = a.get("href", "")
        full = urljoin(NUTRIHOME_BASE, href)
        if full.startswith(NUTRIHOME_BASE) and full != url:
            links.append(full)

    next_url = None
    # Pattern: /tin-tuc/page/2/, /tag/dinh-duong/page/2/
    if re.search(r"/page/(\d+)/?$", url):
        m = re.search(r"/page/(\d+)/?$", url)
        next_page = int(m.group(1)) + 1
        next_url = re.sub(r"/page/\d+/?$", f"/page/{next_page}/", url)
    else:
        next_url = url.rstrip("/") + "/page/2/"

    return list(dict.fromkeys(links)), next_url


def nutrihome_parse_article(url: str, category: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1, .entry-title, .post-title")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    body = soup.select_one(".entry-content, .post-content, .article-content, article")
    if not body:
        return None

    for el in body.select("script, style, .sharedaddy, .related-posts, .wp-block-group"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)
    return make_doc("nutrihome", url, title, content, category)


def crawl_nutrihome(writer: CorpusWriter, max_pages_per_cat: int = 115):
    log.info("=" * 60)
    log.info("▶  Crawling nutrihome.vn ...")
    for cat_url, cat_name in NUTRIHOME_CATEGORIES:
        _crawl_generic(writer, cat_url, cat_name,
                       get_links_fn=nutrihome_get_links,
                       parse_fn=nutrihome_parse_article,
                       max_pages=max_pages_per_cat)
    log.info(f"  nutrihome done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 7 — medlatec.vn
# NOTE: Site blocks automated crawlers (403 + robots.txt ClaudeBot block).
# Code kept for reference but excluded from ALL_SOURCES and default crawl.
# ═══════════════════════════════════════════════════════════════════════════════

MEDLATEC_HEADERS = {
    **HEADERS,
    "Referer": "https://www.google.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

MEDLATEC_CATEGORIES = [
    ("https://medlatec.vn/tin-tuc/dinh-duong",      "dinh-duong"),
    ("https://medlatec.vn/tin-tuc/suc-khoe",        "suc-khoe"),
]


def medlatec_get(url: str) -> Optional[requests.Response]:
    """GET with Medlatec-specific headers to bypass 403."""
    for attempt in range(2):
        try:
            r = requests.get(url, headers=MEDLATEC_HEADERS, timeout=12)
            if r.status_code == 403:
                log.warning(f"  ⚠️  Medlatec 403 for {url} — skipping")
                return None
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r
        except Exception as e:
            log.warning(f"  ⚠️  Medlatec attempt {attempt+1}/2 failed: {e}")
            time.sleep(3)
    return None


def medlatec_get_links(url: str) -> Tuple[List[str], Optional[str]]:
    r = medlatec_get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("a[href*='/tin-tuc/']"):
        href = a.get("href", "")
        full = urljoin("https://medlatec.vn", href)
        # Article pages have deeper path: /tin-tuc/<category>/<slug>
        if re.match(r"https://medlatec\.vn/tin-tuc/[^/]+/[^/]+", full):
            links.append(full)

    next_url = None
    if re.search(r"[?&]page=(\d+)", url):
        m = re.search(r"[?&]page=(\d+)", url)
        next_page = int(m.group(1)) + 1
        next_url = re.sub(r"([?&])page=\d+", rf"\g<1>page={next_page}", url)
    else:
        sep = "&" if "?" in url else "?"
        next_url = f"{url}{sep}page=2"

    return list(dict.fromkeys(links)), next_url


def medlatec_parse_article(url: str, category: str) -> Optional[dict]:
    r = medlatec_get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1, .article-title, .post-title")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    body = soup.select_one(".article-content, .post-content, .entry-content, main")
    if not body:
        return None

    for el in body.select("script, style, .related-posts, aside, .breadcrumb"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)
    return make_doc("medlatec", url, title, content, category)


def crawl_medlatec(writer: CorpusWriter, max_pages_per_cat: int = 50):
    log.info("=" * 60)
    log.info("▶  Crawling medlatec.vn ...")
    for cat_url, cat_name in MEDLATEC_CATEGORIES:
        _crawl_generic(writer, cat_url, cat_name,
                       get_links_fn=medlatec_get_links,
                       parse_fn=medlatec_parse_article,
                       max_pages=max_pages_per_cat)
    log.info(f"  medlatec done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 8 — diag.vn
# Pagination: /blog/page/2/  (262 pages confirmed)
# Article links: /blog/<category>/<slug>/
# ═══════════════════════════════════════════════════════════════════════════════

DIAG_CATEGORIES = [
    ("https://diag.vn/blog/category/blog/dinh-duong/", "dinh-duong"),
    ("https://diag.vn/blog/",                          "general"),
]


def diag_get_links(url: str) -> Tuple[List[str], Optional[str]]:
    r = get(url)
    if not r:
        return [], None
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("h2 a[href], h3 a[href], .post-title a[href], article a[href]"):
        href = a.get("href", "")
        full = urljoin("https://diag.vn", href)
        # Article: /blog/<cat>/<slug>/ — 3 path segments minimum
        if re.match(r"https://diag\.vn/blog/[^/]+/[^/]+/?$", full):
            links.append(full)

    next_url = None
    if re.search(r"/page/(\d+)/?$", url):
        m = re.search(r"/page/(\d+)/?$", url)
        next_page = int(m.group(1)) + 1
        next_url = re.sub(r"/page/\d+/?$", f"/page/{next_page}/", url)
    else:
        next_url = url.rstrip("/") + "/page/2/"

    return list(dict.fromkeys(links)), next_url


def diag_parse_article(url: str, category: str) -> Optional[dict]:
    r = get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1, .post-title, .entry-title")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    body = soup.select_one(".entry-content, .post-content, .article-body, article")
    if not body:
        return None

    for el in body.select("script, style, .related-posts, .social-share, aside"):
        el.decompose()

    content = body.get_text(separator="\n", strip=True)
    return make_doc("diag", url, title, content, category)


def crawl_diag(writer: CorpusWriter, max_pages_per_cat: int = 100):
    log.info("=" * 60)
    log.info("▶  Crawling diag.vn ...")
    for cat_url, cat_name in DIAG_CATEGORIES:
        _crawl_generic(writer, cat_url, cat_name,
                       get_links_fn=diag_get_links,
                       parse_fn=diag_parse_article,
                       max_pages=max_pages_per_cat)
    log.info(f"  diag done. Total corpus so far: {writer.count}")


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC CRAWLER HELPER (used by all sources above)
# ═══════════════════════════════════════════════════════════════════════════════

def _crawl_generic(writer: CorpusWriter, cat_url: str, cat_name: str,
                   get_links_fn, parse_fn, max_pages: int):
    """Shared crawl loop: paginate → collect links → parse articles."""
    log.info(f"  Category: {cat_name}")
    page_url = cat_url
    page_num = 0
    empty_pages = 0  # stop after 2 consecutive empty pages

    while page_url and page_num < max_pages:
        page_num += 1
        links, next_url = get_links_fn(page_url)
        log.info(f"    Page {page_num} → {len(links)} links")

        if not links:
            empty_pages += 1
            if empty_pages >= 2:
                log.info(f"    2 empty pages in a row — stopping {cat_name}")
                break
        else:
            empty_pages = 0

        for art_url in links:
            if art_url in writer.seen:
                continue
            doc = parse_fn(art_url, cat_name)
            if doc and writer.write(doc):
                log.info(f"      ✅ [{writer.count}] {doc['title'][:60]}")
            time.sleep(DELAY_BETWEEN_ARTICLES)

        page_url = next_url
        time.sleep(DELAY_BETWEEN_PAGES)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_SOURCES = [
    "suckhoedoisong", "vnexpress", "hellobacsi", "viendinhduong",
    "vinmec", "nutrihome", "diag",
    # "medlatec",  # excluded: site blocks automated crawlers (403 + robots.txt)
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vietnamese Nutrition Corpus Crawler")
    parser.add_argument(
        "--sources", nargs="+",
        choices=ALL_SOURCES + ["medlatec", "all"],
        default=["all"],
        help="Which sources to crawl (default: all)"
    )
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max pages per category (default: 50)")
    args = parser.parse_args()

    sources = args.sources
    if "all" in sources:
        sources = ALL_SOURCES

    writer = CorpusWriter(OUTPUT_FILE)
    start  = datetime.now()

    try:
        if "suckhoedoisong" in sources:
            crawl_suckhoedoisong(writer, max_pages_per_cat=args.max_pages)

        if "vnexpress" in sources:
            crawl_vnexpress(writer, max_pages_per_cat=args.max_pages)

        if "hellobacsi" in sources:
            crawl_hellobacsi(writer, max_pages_per_cat=args.max_pages)

        if "viendinhduong" in sources:
            crawl_viendinhduong(writer, max_pages_per_cat=args.max_pages)

        if "vinmec" in sources:
            crawl_vinmec(writer, max_pages_per_cat=args.max_pages)

        if "nutrihome" in sources:
            crawl_nutrihome(writer, max_pages_per_cat=args.max_pages)

        if "medlatec" in sources:
            crawl_medlatec(writer, max_pages_per_cat=args.max_pages)

        if "diag" in sources:
            crawl_diag(writer, max_pages_per_cat=args.max_pages)

    except KeyboardInterrupt:
        log.info("⏹  Interrupted by user.")
    finally:
        writer.close()
        elapsed = datetime.now() - start
        log.info("=" * 60)
        log.info(f"✅ Done! Total documents: {writer.count}")
        log.info(f"⏱  Elapsed: {elapsed}")
        log.info(f"📄 Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
