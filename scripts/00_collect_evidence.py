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

# small dataset lives under: REPO_ROOT / data / verite_small
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
UA = cfg.get("evidence", {}).get("user_agent", "ooc-verifier/1.0")
TIMEOUT = int(cfg.get("evidence", {}).get("timeout", 12))
PROVIDER = cfg.get("evidence", {}).get("provider", "serpapi").lower()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()

# Quality controls
MIN_EDGE = int(cfg.get("evidence", {}).get("min_image_edge_hq", 600))  # require HQ
# Text->image: request large images
GOOGLE_IMAGES_TBS = cfg.get("evidence", {}).get("google_images_tbs", "isz:l")  # large :contentReference[oaicite:3]{index=3}

# ----------------- utils -----------------
def _headers():
    return {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

def save_image_bytes(img_bytes: bytes, out_dir: Path, min_edge: int = MIN_EDGE) -> Optional[Path]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if min(w, h) < min_edge:
            return None
        name = hashlib.sha1(img_bytes).hexdigest()[:12] + ".jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / name
        img.save(fp, "JPEG", quality=92)
        return fp
    except Exception:
        return None

def save_image_from_url(url: str, out_dir: Path, min_edge: int = MIN_EDGE) -> Optional[Path]:
    if not url:
        return None
    try:
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        # some sites lie about content-type; PIL will fail if not an image anyway
        return save_image_bytes(r.content, out_dir, min_edge=min_edge)
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

# ----------------- SerpAPI: text -> image (HQ) -----------------
def serpapi_image_search(query: str, num: int = 1) -> List[Dict]:
    """
    Use Google Images engine with a "large" filter (tbs=isz:l) to reduce low-res hits.
    :contentReference[oaicite:4]{index=4}
    """
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_images",
        "q": query,
        "ijn": "0",
        "api_key": SERPAPI_KEY,
        "safe": "active",
        "tbs": GOOGLE_IMAGES_TBS,
        "device": "desktop",
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()

    out = []
    for it in (js.get("images_results") or [])[:max(1, num)]:
        out.append({
            "contentUrl": it.get("original") or "",
            "hostPageUrl": it.get("link") or "",
            "title": it.get("title") or ""
        })
    return out

# ----------------- SerpAPI: reverse search (HQ) via Google Lens -----------------
def serpapi_reverse_from_image_url(image_url: str, num: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """
    Use Google Lens API (engine=google_lens, type=visual_matches).
    Lens 'visual_matches' items often include:
      - image (full-res)
      - image_width / image_height
      - link (host page)
    :contentReference[oaicite:5]{index=5}
    Returns:
      pages: list of {hostPageUrl, title, snippet}
      sims:  list of {contentUrl, hostPageUrl, title, w, h}
    """
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_lens",
        "type": "visual_matches",
        "url": image_url,          # IMPORTANT: param name is 'url' for google_lens :contentReference[oaicite:6]{index=6}
        "api_key": SERPAPI_KEY,
        "safe": "active",
        "device": "desktop",
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()

    sims = []
    for it in (js.get("visual_matches") or [])[:max(1, num)]:
        sims.append({
            "contentUrl": it.get("image") or "",           # HQ image URL :contentReference[oaicite:7]{index=7}
            "hostPageUrl": it.get("link") or "",
            "title": it.get("title") or "",
            "w": int(it.get("image_width") or 0),
            "h": int(it.get("image_height") or 0),
        })

    pages = []
    for it in (js.get("visual_matches") or [])[:max(1, num)]:
        pages.append({
            "hostPageUrl": it.get("link") or "",
            "title": it.get("title") or "",
            "snippet": "",
        })

    # dedupe pages by url
    seen, out_pages = set(), []
    for p in pages:
        u = p.get("hostPageUrl")
        if not u or u in seen:
            continue
        seen.add(u)
        out_pages.append(p)

    return out_pages, sims

# ----------------- args -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=None, help="Max number of article rows (ids) to process.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip ids that already have meta.json.")
    ap.add_argument("--skip_reverse", action="store_true", help="Skip reverse-image search step.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def sample_ids(vid: int) -> Dict[str, str]:
    """Must match 01_extract_verite.py ids."""
    return {"true": f"{vid}_true", "mis": f"{vid}_mis", "ooc": f"{vid}_ooc"}

def _dedupe_str_list(xs: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in xs if isinstance(x, str) and x.strip()]))

def pick_best_sim(sims: List[Dict]) -> Optional[Dict]:
    """Pick the highest-resolution candidate (prefer large min(w,h), then area)."""
    if not sims:
        return None
    sims2 = []
    for s in sims:
        w, h = int(s.get("w") or 0), int(s.get("h") or 0)
        if min(w, h) >= MIN_EDGE and s.get("contentUrl"):
            sims2.append(s)
    if not sims2:
        return None
    sims2.sort(key=lambda x: (min(int(x.get("w") or 0), int(x.get("h") or 0)),
                              int(x.get("w") or 0) * int(x.get("h") or 0)),
              reverse=True)
    return sims2[0]

# ----------------- main -----------------
def main():
    if PROVIDER != "serpapi":
        raise SystemExit(f"Unknown provider: {PROVIDER}")
    if not SERPAPI_KEY:
        raise SystemExit("Set SERPAPI_KEY in .env for provider=serpapi")

    args = parse_args()

    # Load VERITE_articles.csv
    art_csv = small_root / "VERITE_articles.csv"
    if not art_csv.exists():
        raise SystemExit(f"VERITE_articles.csv not found at: {art_csv}")

    art_df = pd.read_csv(art_csv)
    if "id" not in art_df.columns:
        raise SystemExit("VERITE_articles.csv must contain an 'id' column")

    art_df["id"] = pd.to_numeric(art_df["id"], errors="coerce")
    art_df = art_df.dropna(subset=["id"]).reset_index(drop=True)
    art_df["id"] = art_df["id"].astype(int)

    if args.max_pairs is not None:
        art_df = art_df.iloc[:args.max_pairs].reset_index(drop=True)

    rows = []

    print(f"[info] Processing {len(art_df)} ids using SerpAPI")
    print(f"[info] Reverse uses Google Lens visual_matches (HQ image URLs). MIN_EDGE={MIN_EDGE}px")

    for _, arow in tqdm(art_df.iterrows(), total=len(art_df)):
        vid = int(arow["id"])

        pair_dir = evidence_root / f"{vid:06d}"
        meta_fp = pair_dir / "meta.json"

        if args.skip_existing and meta_fp.exists():
            try:
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                if isinstance(meta.get("samples"), dict):
                    for k, sid in sample_ids(vid).items():
                        s = meta["samples"].get(k, {})
                        rows.append({
                            "match_index": sid,
                            "captions": json.dumps(s.get("captions", []), ensure_ascii=False),
                            "images_paths": json.dumps(s.get("images_paths", []), ensure_ascii=False),
                        })
                continue
            except Exception:
                pass

        img_from_img_dir = pair_dir / "from_image" / "images"
        img_from_txt_dir = pair_dir / "from_text" / "images"
        img_from_img_dir.mkdir(parents=True, exist_ok=True)
        img_from_txt_dir.mkdir(parents=True, exist_ok=True)

        true_caption = str(arow.get("true_caption", "") or "").strip()
        false_caption = str(arow.get("false_caption", "") or "").strip()
        true_url = str(arow.get("true_url", "") or "").strip()
        false_url = str(arow.get("false_url", "") or "").strip()

        text_hits: Dict[str, Dict] = {"true_caption": {}, "false_caption": {}}
        reverse_hits: Dict[str, Dict] = {"true_url": {}, "false_url": {}}

        text_search_meta: List[Dict] = []
        reverse_meta = {"true": {}, "false": {}}

        # ---------- TEXT SEARCH ----------
        def run_text_search(query: str, query_type: str):
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

                fp = save_image_from_url(content_url, img_from_txt_dir, min_edge=MIN_EDGE)
                downloaded_path = rel_to_repo(fp) if fp else ""

                text_hits[query_type] = {
                    "query": query,
                    "result_content_url": content_url,
                    "result_hostPageUrl": host_page,
                    "result_title": hit_title,
                    "downloaded_image": downloaded_path,
                }

                text_search_meta.append({
                    "query_type": query_type,
                    "query": query,
                    "result_content_url": content_url,
                    "result_hostPageUrl": host_page,
                    "result_title": hit_title,
                    "downloaded_image": downloaded_path
                })
            time.sleep(0.4)

        run_text_search(true_caption, "true_caption")
        run_text_search(false_caption, "false_caption")

        # ---------- REVERSE SEARCH (HQ via Google Lens) ----------
        def run_reverse(image_url: str, which: str):
            if not image_url or args.skip_reverse:
                return
            try:
                pages, sims = serpapi_reverse_from_image_url(image_url, num=5)
            except Exception as e:
                print(f"[warn] reverse ({which}) failed (id={vid}): {e}")
                pages, sims = ([], [])

            best = pick_best_sim(sims)  # HQ only
            key = "true" if which == "true_url" else "false"

            if best:
                host = best.get("hostPageUrl", "")
                title = best.get("title") or fetch_title(host)
                fp = save_image_from_url(best.get("contentUrl", ""), img_from_img_dir, min_edge=MIN_EDGE)
                downloaded_path = rel_to_repo(fp) if fp else ""

                reverse_hits[which] = {
                    "input_image_url": image_url,
                    "chosen_image_url": best.get("contentUrl", ""),
                    "hostPageUrl": host,
                    "title": title,
                    "w": int(best.get("w") or 0),
                    "h": int(best.get("h") or 0),
                    "downloaded_image": downloaded_path,
                }
                reverse_meta[key] = reverse_hits[which]
            else:
                # No HQ candidate found -> leave empty (better than downloading thumbnails)
                reverse_hits[which] = {"input_image_url": image_url, "note": "no_hq_visual_match_found"}
                reverse_meta[key] = reverse_hits[which]

            time.sleep(0.4)

        run_reverse(true_url, "true_url")
        run_reverse(false_url, "false_url")

        # local source images (from your small dataset)
        src_true = (small_root / "images" / f"true_{vid}.jpg").resolve()
        src_false = (small_root / "images" / f"false_{vid}.jpg").resolve()

        # ---------- Assemble evidence per sample ----------
        def assemble(sample_kind: str) -> Tuple[List[str], List[str]]:
            caps: List[str] = []
            imgs: List[str] = []

            if sample_kind == "true":
                tc = text_hits.get("true_caption", {})
                rv = reverse_hits.get("true_url", {})
            elif sample_kind == "mis":
                tc = text_hits.get("false_caption", {})
                rv = reverse_hits.get("true_url", {})
            elif sample_kind == "ooc":
                tc = text_hits.get("true_caption", {})
                rv = reverse_hits.get("false_url", {})
            else:
                tc, rv = {}, {}

            if tc.get("result_title"):
                caps.append(tc["result_title"])
            if rv.get("title"):
                caps.append(rv["title"])

            if tc.get("downloaded_image"):
                imgs.append(tc["downloaded_image"])
            if rv.get("downloaded_image"):
                imgs.append(rv["downloaded_image"])

            return _dedupe_str_list(caps), _dedupe_str_list(imgs)

        samples_out = {}
        for kind, sid in sample_ids(vid).items():
            caps_list, imgs_list = assemble(kind)
            samples_out[kind] = {"sample_id": sid, "captions": caps_list, "images_paths": imgs_list}
            rows.append({
                "match_index": sid,
                "captions": json.dumps(caps_list, ensure_ascii=False),
                "images_paths": json.dumps(imgs_list, ensure_ascii=False),
            })

        agg_caps = _dedupe_str_list(samples_out["true"]["captions"] + samples_out["mis"]["captions"] + samples_out["ooc"]["captions"])
        agg_imgs = _dedupe_str_list(samples_out["true"]["images_paths"] + samples_out["mis"]["images_paths"] + samples_out["ooc"]["images_paths"])

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
                    "text_search": text_search_meta,
                    "samples": samples_out,
                    "captions": agg_caps,
                    "images_paths": agg_imgs,
                    "quality": {
                        "min_edge_required": MIN_EDGE,
                        "reverse_engine": "google_lens visual_matches",
                        "text_engine": f"google_images tbs={GOOGLE_IMAGES_TBS}",
                        "note": "Reverse downloads only HQ 'image' URLs from Lens visual_matches; thumbnails are ignored."
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    out_csv = evidence_root / "collected_evidence_small.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Evidence saved under {rel_to_repo(evidence_root)}")
    print(f"✓ CSV for encoder: {rel_to_repo(out_csv)}")

if __name__ == "__main__":
    main()
