#!/usr/bin/env python3
"""
verite_download_small.py

Download images for the VERITE dataset from VERITE_articles_small.csv
and save them into the local "images" folder so they match the
paths used in VERITE_small.csv (images/true_<id>.jpg and images/false_<id>.jpg).
"""

import csv
import os
from pathlib import Path

import requests

# Config
ARTICLES_CSV = "VERITE_articles_small.csv"
IMAGES_DIR = Path("images")
TIMEOUT = 20  # seconds for HTTP requests


def download_image(url: str, dest_path: Path) -> None:
    """
    Download an image from `url` and save it to `dest_path`.

    - Skips download if dest_path already exists.
    - Prints a short log line for success or failure.
    """
    if dest_path.exists():
        print(f"[SKIP] {dest_path} already exists.")
        return

    if not url:
        print(f"[WARN] Empty URL for {dest_path.name}, skipping.")
        return

    try:
        print(f"[GET ] {url}")
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()

        # Make sure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "wb") as f:
            f.write(resp.content)

        print(f"[OK  ] Saved to {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"[ERR ] Failed to download {url} -> {dest_path}: {e}")


def main():
    # Ensure images directory exists
    IMAGES_DIR.mkdir(exist_ok=True)

    if not Path(ARTICLES_CSV).exists():
        print(f"[FATAL] CSV file not found: {ARTICLES_CSV}")
        return

    with open(ARTICLES_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # In your CSV, "id" is the index that matches the _<id> in VERITE_small.csv
            row_id = row.get("id", "").strip()

            if row_id == "":
                print("[WARN] Row without 'id' value, skipping.")
                continue

            true_url = (row.get("true_url") or "").strip()
            false_url = (row.get("false_url") or "").strip()

            # Destination file paths as used in VERITE_small.csv
            true_dest = IMAGES_DIR / f"true_{row_id}.jpg"
            false_dest = IMAGES_DIR / f"false_{row_id}.jpg"

            # Download images
            if true_url:
                download_image(true_url, true_dest)
            else:
                print(f"[WARN] No true_url for id={row_id}")

            if false_url:
                download_image(false_url, false_dest)
            else:
                print(f"[WARN] No false_url for id={row_id}")


if __name__ == "__main__":
    main()
