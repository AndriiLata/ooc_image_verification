# scripts/00_collect_evidence.py
import os, io, json, time, hashlib, yaml, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from PIL import Image
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# ----------------- repo root & helpers -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]

def rel_to_repo(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)

load_dotenv()

# ----------------- read config -----------------
cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))

# original VERITE root from config (still there if you need it)
ver_root = (REPO_ROOT / cfg["paths"]["verite"]).resolve()

# SMALL dataset lives under: REPO_ROOT / data / verite_small
small_root = (REPO_ROOT / "data" / "verite_small").resolve()
if not small_root.exists():
    raise SystemExit(f"verite_small folder not found at: {small_root}")

evidence_root_cfg = (
    cfg["paths"].get("evidence_root")
    or cfg["paths"].get("evidence_out")
    or "./data/evidence"
)
evidence_root = (REPO_ROOT / evidence_root_cfg).resolve()
evidence_root.mkdir(parents=True, exist_ok=True)

# SerpAPI-related config
K = int(cfg.get("evidence", {}).get("k_results", 3))  # still max cap if needed
UA = cfg.get("evidence", {}).get("user_agent", "ooc-verifier/1.0")
TIMEOUT = int(cfg.get("evidence", {}).get("timeout", 12))
PROVIDER = cfg.get("evidence", {}).get("provider", "serpapi").lower()

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()

# ----------------- utils -----------------
def _headers():
    return {"User-Agent": UA}

def save_image_bytes(img_bytes: bytes, out_dir: Path) -> Optional[Path]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        name = hashlib.sha1(img_bytes).hexdigest()[:12] + ".jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / name
        img.save(fp, "JPEG", quality=92)
        return fp
    except Exception:
        return None

def save_image_from_url(url: str, out_dir: Path) -> Optional[Path]:
    if not url:
        return None
    try:
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        return save_image_bytes(r.content, out_dir)
    except Exception:
        return None

def fetch_title(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return (soup.title.string or "").strip() if soup.title else ""
    except Exception:
        return ""

# ----------------- SerpAPI: text -> image -----------------
def serpapi_image_search(query: str, num: int = 1) -> List[Dict]:
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_images",
        "q": query,
        "ijn": "0",
        "api_key": SERPAPI_KEY,
        "safe": "active"
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    out = []
    for it in (js.get("images_results") or [])[:max(1, num)]:
        out.append({
            "contentUrl": it.get("original") or it.get("thumbnail"),
            "hostPageUrl": it.get("link"),
            "title": it.get("title") or ""
        })
    return out

# ----------------- SerpAPI: reverse via image_url (GET) -----------------
def _parse_reverse_pages(js: dict, num: int) -> List[Dict]:
    pages, blocks = [], []
    for key in ("image_results", "organic_results", "image_sources", "sources"):
        if isinstance(js.get(key), list):
            blocks.extend(js[key])
    for it in blocks:
        pages.append({
            "hostPageUrl": it.get("link") or it.get("source") or it.get("url"),
            "title": it.get("title") or it.get("snippet") or "",
            "snippet": it.get("snippet") or ""
        })
        if len(pages) >= max(1, num):
            break
    # dedupe by URL
    seen, out = set(), []
    for p in pages:
        u = p.get("hostPageUrl")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(p)
    return out

def _parse_reverse_similar_images(js: dict, num: int) -> List[Dict]:
    sims, blocks = [], []
    for key in ("inline_images", "visual_matches", "images_results", "image_results"):
        if isinstance(js.get(key), list):
            blocks.extend(js[key])
    for it in blocks:
        content = it.get("original") or it.get("image") or it.get("thumbnail")
        if not content:
            continue
        sims.append({
            "contentUrl": content,
            "hostPageUrl": it.get("source") or it.get("link") or it.get("displayed_link"),
            "title": it.get("title") or it.get("snippet") or ""
        })
        if len(sims) >= max(1, num):
            break
    return sims

def serpapi_reverse_from_image_url(image_url: str, num: int = 1) -> Tuple[List[Dict], List[Dict]]:
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_reverse_image",
        "image_url": image_url,
        "api_key": SERPAPI_KEY
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    pages = _parse_reverse_pages(js, num)
    sims = _parse_reverse_similar_images(js, num)
    return pages, sims

# ----------------- args -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=None, help="Max number of article rows (ids) to process.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip ids that already have meta.json.")
    ap.add_argument("--skip_reverse", action="store_true", help="Skip reverse-image search step.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ----------------- main -----------------
def main():
    if PROVIDER != "serpapi":
        raise SystemExit(f"Unknown provider: {PROVIDER}")
    if not SERPAPI_KEY:
        raise SystemExit("Set SERPAPI_KEY in .env for provider=serpapi")

    args = parse_args()

    # Load VERITE_articles_small.csv
    art_csv = small_root / "VERITE_articles_small.csv"
    if not art_csv.exists():
        raise SystemExit(f"VERITE_articles_small.csv not found at: {art_csv}")

    art_df = pd.read_csv(art_csv)
    if "id" not in art_df.columns:
        raise SystemExit("VERITE_articles_small.csv must contain an 'id' column")

    # Normalise id to int
    art_df["id"] = pd.to_numeric(art_df["id"], errors="coerce").astype("Int64")
    art_df = art_df.dropna(subset=["id"]).reset_index(drop=True)
    art_df["id"] = art_df["id"].astype(int)

    if args.max_pairs is not None:
        art_df = art_df.iloc[:args.max_pairs].reset_index(drop=True)

    rows = []

    print(f"[info] Processing {len(art_df)} ids from VERITE_articles_small.csv using SerpAPI")

    for _, arow in tqdm(art_df.iterrows(), total=len(art_df)):
        vid = int(arow["id"])

        pair_dir = evidence_root / f"{vid:06d}"
        meta_fp = pair_dir / "meta.json"

        if args.skip_existing and meta_fp.exists():
            # read existing for aggregated CSV if possible
            try:
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                captions_list = meta.get("captions", [])
                images_list = meta.get("images_paths", [])
                rows.append({
                    "match_index": vid,
                    "captions": json.dumps(captions_list, ensure_ascii=False),
                    "images_paths": json.dumps(images_list, ensure_ascii=False),
                })
                continue
            except Exception:
                pass

        # dirs
        img_from_img_dir = pair_dir / "from_image" / "images"
        img_from_txt_dir = pair_dir / "from_text" / "images"
        img_from_img_dir.mkdir(parents=True, exist_ok=True)
        img_from_txt_dir.mkdir(parents=True, exist_ok=True)

        true_caption = str(arow.get("true_caption", "") or "").strip()
        false_caption = str(arow.get("false_caption", "") or "").strip()
        true_url = str(arow.get("true_url", "") or "").strip()
        false_url = str(arow.get("false_url", "") or "").strip()

        captions_accum: List[str] = []
        images_accum: List[str] = []
        text_search_meta: List[Dict] = []   # NEW: detailed text search metadata

        # ---------- TEXT SEARCH: helper ----------
        def run_text_search(query: str, query_type: str):
            nonlocal captions_accum, images_accum, text_search_meta
            if not query:
                return
            try:
                hits = serpapi_image_search(query, num=1)
            except Exception as e:
                print(f"[warn] text search ({query_type}) failed (id={vid}): {e}")
                hits = []
            if hits:
                h = hits[0]
                content_url = h.get("contentUrl", "")
                host_page = h.get("hostPageUrl", "")
                hit_title = h.get("title") or fetch_title(host_page)

                if hit_title:
                    captions_accum.append(hit_title)

                fp = save_image_from_url(content_url, img_from_txt_dir)
                downloaded_path = rel_to_repo(fp) if fp else ""

                if downloaded_path:
                    images_accum.append(downloaded_path)

                text_search_meta.append({
                    "query_type": query_type,          # "true_caption" or "false_caption"
                    "query": query,
                    "result_content_url": content_url,
                    "result_hostPageUrl": host_page,
                    "result_title": hit_title,
                    "downloaded_image": downloaded_path
                })
            time.sleep(0.6)

        # ---------- TEXT SEARCH: true_caption ----------
        run_text_search(true_caption, "true_caption")

        # ---------- TEXT SEARCH: false_caption ----------
        run_text_search(false_caption, "false_caption")

        reverse_meta = {"true": {}, "false": {}}

        # ---------- REVERSE SEARCH: true_url ----------
        if true_url and not args.skip_reverse:
            try:
                pages, sims = serpapi_reverse_from_image_url(true_url, num=1)
            except Exception as e:
                print(f"[warn] reverse (true_url) failed (id={vid}): {e}")
                pages, sims = ([], [])
            # first similar image if available
            if sims:
                s = sims[0]
                vm_title = s.get("title") or fetch_title(s.get("hostPageUrl", ""))
                if vm_title:
                    captions_accum.append(vm_title)
                fp = save_image_from_url(s.get("contentUrl", ""), img_from_img_dir)
                if fp:
                    images_accum.append(rel_to_repo(fp))
                reverse_meta["true"] = {
                    "image_url": s.get("contentUrl", ""),
                    "hostPageUrl": s.get("hostPageUrl", ""),
                    "title": vm_title,
                }
            elif pages:
                # fallback: use page title/snippet even if we don't have a direct image url
                p = pages[0]
                title = p.get("title") or fetch_title(p.get("hostPageUrl", ""))
                if title:
                    captions_accum.append(title)
                reverse_meta["true"] = {
                    "image_url": "",
                    "hostPageUrl": p.get("hostPageUrl", ""),
                    "title": title,
                }
            time.sleep(0.6)

        # ---------- REVERSE SEARCH: false_url ----------
        if false_url and not args.skip_reverse:
            try:
                pages, sims = serpapi_reverse_from_image_url(false_url, num=1)
            except Exception as e:
                print(f"[warn] reverse (false_url) failed (id={vid}): {e}")
                pages, sims = ([], [])
            if sims:
                s = sims[0]
                vm_title = s.get("title") or fetch_title(s.get("hostPageUrl", ""))
                if vm_title:
                    captions_accum.append(vm_title)
                fp = save_image_from_url(s.get("contentUrl", ""), img_from_img_dir)
                if fp:
                    images_accum.append(rel_to_repo(fp))
                reverse_meta["false"] = {
                    "image_url": s.get("contentUrl", ""),
                    "hostPageUrl": s.get("hostPageUrl", ""),
                    "title": vm_title,
                }
            elif pages:
                p = pages[0]
                title = p.get("title") or fetch_title(p.get("hostPageUrl", ""))
                if title:
                    captions_accum.append(title)
                reverse_meta["false"] = {
                    "image_url": "",
                    "hostPageUrl": p.get("hostPageUrl", ""),
                    "title": title,
                }
            time.sleep(0.6)

        # local source images (from your small dataset)
        src_true = (small_root / "images" / f"true_{vid}.jpg").resolve()
        src_false = (small_root / "images" / f"false_{vid}.jpg").resolve()

        # unique + clean
        captions_list = list(
            dict.fromkeys([c for c in captions_accum if isinstance(c, str) and c.strip()])
        )
        images_list = list(
            dict.fromkeys([p for p in images_accum if isinstance(p, str) and p.strip()])
        )

        # write meta.json with extra info for debugging
        pair_dir.mkdir(parents=True, exist_ok=True)
        meta_fp.write_text(
            json.dumps(
                {
                    "id": vid,
                    "true_caption": true_caption,
                    "false_caption": false_caption,
                    "true_url": true_url,
                    "false_url": false_url,
                    "source_image_true": str(src_true) if src_true.exists() else "",
                    "source_image_false": str(src_false) if src_false.exists() else "",
                    "reverse": reverse_meta,
                    "text_search": text_search_meta,       # NEW: detailed text search info
                    "captions": captions_list,             # aggregated evidence captions
                    "images_paths": images_list,           # aggregated evidence images
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # row for encoder step
        rows.append(
            {
                "match_index": vid,
                "captions": json.dumps(captions_list, ensure_ascii=False),
                "images_paths": json.dumps(images_list, ensure_ascii=False),
            }
        )

    out_csv = evidence_root / "collected_evidence_small.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Evidence saved under {rel_to_repo(evidence_root)}")
    print(f"✓ CSV for encoder: {rel_to_repo(out_csv)}")


if __name__ == "__main__":
    main()
