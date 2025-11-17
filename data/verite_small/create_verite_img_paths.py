#!/usr/bin/env python3
"""
create_image_url_map.py

Creates a CSV file mapping:
  - true_<id>.jpg → true_url
  - false_<id>.jpg → false_url

Input:  VERITE_articles_small.csv
Output: image_url_map.csv
"""

import csv
from pathlib import Path

ARTICLES_CSV = "VERITE_articles_small.csv"
OUTPUT_CSV = "image_url_map.csv"


def main():
    if not Path(ARTICLES_CSV).exists():
        print(f"[FATAL] {ARTICLES_CSV} not found.")
        return

    rows_out = []

    with open(ARTICLES_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id", "").strip()
            if row_id == "":
                continue

            true_url = (row.get("true_url") or "").strip()
            false_url = (row.get("false_url") or "").strip()

            # Add true image mapping
            if true_url:
                rows_out.append({
                    "image_name": f"true_{row_id}.jpg",
                    "image_url": true_url
                })

            # Add false image mapping
            if false_url:
                rows_out.append({
                    "image_name": f"false_{row_id}.jpg",
                    "image_url": false_url
                })

    # Write output CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "image_url"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] Created {OUTPUT_CSV} with {len(rows_out)} rows.")


if __name__ == "__main__":
    main()
