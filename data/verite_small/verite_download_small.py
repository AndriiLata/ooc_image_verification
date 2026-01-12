#!/usr/bin/env python3
"""
verite_download_small_fixed.py

Robust downloader for VERITE_articles.csv

- Retries + backoff
- Browser-like headers
- Fallback URL fixes for common CDN patterns (Snopes mediaproxy, strip query params)
- Converts downloaded image to JPEG so filenames match images/true_<id>.jpg, images/false_<id>.jpg
"""

import csv
import io
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

# ---------------- Config ----------------
ARTICLES_CSV = "VERITE_articles.csv"
IMAGES_DIR = Path("images")

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

SLEEP_BETWEEN = 0.15   # be polite
MAX_BYTES = 25 * 1024 * 1024  # safety: 25MB

# A realistic desktop UA helps a lot for 403
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "User-Agent": UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# ---------------- Session with retries ----------------
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(BASE_HEADERS)
    return s

# ---------------- Helpers ----------------
def strip_query(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return url

def snopes_mediaproxy(url: str, width: int = 1200) -> str:
    # Convert: https://media.snopes.com/... -> https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/...
    return f"https://mediaproxy.snopes.com/width/{width}/{url}"

def candidate_urls(url: str) -> List[str]:
    """
    Generate fallback URLs from a single URL.
    Order matters: try best first.
    """
    if not url:
        return []

    urls = [url]

    # 1) If it contains lots of transforming params, try without query string
    if "?" in url:
        urls.append(strip_query(url))

    # 2) Snopes: direct media sometimes 403; mediaproxy often works
    if "media.snopes.com/" in url:
        urls.insert(0, snopes_mediaproxy(url, 1200))
        urls.insert(1, snopes_mediaproxy(url, 600))

    return list(dict.fromkeys(urls))

def download_bytes(session: requests.Session, url: str, referer: Optional[str] = None) -> Optional[bytes]:
    """
    Download bytes with streaming + size cap.
    Returns None if fails.
    """
    headers = {}
    if referer:
        headers["Referer"] = referer

    try:
        with session.get(url, timeout=TIMEOUT, stream=True, headers=headers, allow_redirects=True) as r:
            # If forbidden, try again with referer = origin homepage
            if r.status_code == 403 and referer is None:
                parsed = urlparse(url)
                origin = f"{parsed.scheme}://{parsed.netloc}/"
                return download_bytes(session, url, referer=origin)

            r.raise_for_status()

            content = bytearray()
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > MAX_BYTES:
                    raise RuntimeError(f"Image too large (> {MAX_BYTES} bytes)")
            return bytes(content)

    except Exception:
        return None

def save_as_jpeg(img_bytes: bytes, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    img.save(dest_path, format="JPEG", quality=92, optimize=True)

def download_image(session: requests.Session, url: str, dest_path: Path) -> bool:
    """
    Try multiple candidate URLs until one works.
    Returns True if saved, False otherwise.
    """
    if dest_path.exists():
        print(f"[SKIP] {dest_path} already exists.")
        return True

    if not url:
        print(f"[WARN] Empty URL for {dest_path.name}, skipping.")
        return False

    for u in candidate_urls(url):
        print(f"[GET ] {u}")
        b = download_bytes(session, u)
        if not b:
            continue

        # Validate & convert to JPEG
        try:
            save_as_jpeg(b, dest_path)
            print(f"[OK  ] Saved to {dest_path}")
            return True
        except Exception as e:
            # bytes were not a valid image (HTML error page, etc.)
            print(f"[ERR ] Not an image / cannot decode -> {dest_path}: {e}")
            continue

    print(f"[FAIL] Could not download any working version for -> {dest_path}")
    return False

# ---------------- Main ----------------
def main():
    IMAGES_DIR.mkdir(exist_ok=True)

    csv_path = Path(ARTICLES_CSV)
    if not csv_path.exists():
        print(f"[FATAL] CSV file not found: {ARTICLES_CSV}")
        return

    session = make_session()

    ok, fail = 0, 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = (row.get("id") or "").strip()
            if row_id == "":
                print("[WARN] Row without 'id' value, skipping.")
                continue

            true_url = (row.get("true_url") or "").strip()
            false_url = (row.get("false_url") or "").strip()

            true_dest = IMAGES_DIR / f"true_{row_id}.jpg"
            false_dest = IMAGES_DIR / f"false_{row_id}.jpg"

            if true_url:
                ok += int(download_image(session, true_url, true_dest))
                fail += int(not true_dest.exists())
            else:
                print(f"[WARN] No true_url for id={row_id}")

            time.sleep(SLEEP_BETWEEN)

            if false_url:
                ok += int(download_image(session, false_url, false_dest))
                fail += int(not false_dest.exists())
            else:
                print(f"[WARN] No false_url for id={row_id}")

            time.sleep(SLEEP_BETWEEN)

    print("\nDone.")
    print(f"Saved OK: {ok}")
    print(f"Missing : {fail}")

if __name__ == "__main__":
    main()
