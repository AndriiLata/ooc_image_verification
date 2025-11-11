# scripts/00_collect_evidence.py
import os, io, json, time, base64, hashlib, yaml
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from PIL import Image
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import argparse, random

# ----------------- repo root & small helpers -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # one level up from scripts/

def rel_to_repo(p: Path) -> str:
    """Return path string relative to the repo root (fallback to absolute)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)

load_dotenv()

# ----------------- read config -----------------
cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
ver_root = (REPO_ROOT / cfg["paths"]["verite"]).resolve()

# allow both keys: evidence_root (new) or evidence_out (old), default to ./data/evidence
evidence_root_cfg = (
    cfg["paths"].get("evidence_root")
    or cfg["paths"].get("evidence_out")
    or "./data/evidence"
)
evidence_root = (REPO_ROOT / evidence_root_cfg).resolve()
evidence_root.mkdir(parents=True, exist_ok=True)

K = int(cfg.get("evidence", {}).get("k_results", 3))
UA = cfg.get("evidence", {}).get("user_agent", "ooc-verifier/1.0")
TIMEOUT = int(cfg.get("evidence", {}).get("timeout", 12))
PROVIDER = cfg.get("evidence", {}).get("provider", "serpapi").lower()  # we implement serpapi here

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# ----------------- utils -----------------
def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]

def _headers():
    return {"User-Agent": UA}

def save_image_bytes(img_bytes: bytes, out_dir: Path) -> Path | None:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        name = _sha1(img_bytes) + ".jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / name
        img.save(fp, "JPEG", quality=92)
        return fp
    except Exception:
        return None

def save_image_from_url(url: str, out_dir: Path) -> Path | None:
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

# ----------------- SerpAPI providers -----------------
def serpapi_image_search(query: str, num: int) -> List[Dict]:
    """Google Images via SerpAPI (text -> images)."""
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_images",
        "q": query,
        "ijn": "0",
        "api_key": SERPAPI_KEY,
        "num": max(1, num),
        "safe": "active"
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    out = []
    for it in (js.get("images_results") or [])[:num]:
        out.append({
            "contentUrl": it.get("original") or it.get("thumbnail"),
            "hostPageUrl": it.get("link"),
            "title": it.get("title") or ""
        })
    return out

def _serpapi_reverse_from_image_url(image_url: str, num: int) -> Tuple[List[Dict], List[Dict]]:
    """Reverse image using a public image URL (no upload)."""
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
    pages = []
    for it in (js.get("best_guess", {}).get("pages", []) or [])[:num]:
        pages.append({"hostPageUrl": it.get("link"), "title": it.get("title") or ""})
    sim = []
    for it in (js.get("visual_matches") or [])[:num]:
        sim.append({
            "contentUrl": it.get("original") or it.get("thumbnail"),
            "hostPageUrl": it.get("source"),
            "title": it.get("title") or ""
        })
    return pages, sim

def _serpapi_reverse_from_local_file(img_path: Path, num: int) -> Tuple[List[Dict], List[Dict]]:
    """Reverse image via base64 upload (can 404 on SerpAPI lately)."""
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_reverse_image",
        "encoded_image": b64,
        "api_key": SERPAPI_KEY
    }
    r = requests.post(url, data=params, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    pages = []
    for it in (js.get("best_guess", {}).get("pages", []) or [])[:num]:
        pages.append({"hostPageUrl": it.get("link"), "title": it.get("title") or ""})
    sim = []
    for it in (js.get("visual_matches") or [])[:num]:
        sim.append({
            "contentUrl": it.get("original") or it.get("thumbnail"),
            "hostPageUrl": it.get("source"),
            "title": it.get("title") or ""
        })
    return pages, sim

def serpapi_reverse_with_fallback(local_img: Path, articles_row: dict | None, num: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Try base64 upload first. If it 404s, and we have a public image URL in VERITE_articles
    (true_url or false_url), retry using image_url.
    """
    # 1) base64 upload
    try:
        return _serpapi_reverse_from_local_file(local_img, num)
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status not in (400, 404, 422):
            raise
    except Exception:
        pass

    # 2) fallback via image_url if available
    if articles_row:
        for col in ("true_url", "false_url"):
            image_url = str(articles_row.get(col, "") or "").strip()
            if image_url:
                try:
                    return _serpapi_reverse_from_image_url(image_url, num)
                except Exception:
                    continue

    # 3) give up
    return ([], [])

# ----------------- args -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=10, help="Process at most N VERITE rows")
    ap.add_argument("--max_unique_images", type=int, default=10, help="Reverse-search at most N unique images")
    ap.add_argument("--k_results", type=int, default=None, help="Override K results per direction")
    ap.add_argument("--use_article_captions", action="store_true", help="Also use true/false/query from VERITE_articles.csv")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if meta.json already exists for a pair")
    ap.add_argument("--skip_reverse", action="store_true", help="Do not perform reverse-image search (only text->image)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ----------------- main -----------------
def main():
    args = parse_args()
    random.seed(args.seed)

    # Load VERITE
    ver_csv = ver_root / "VERITE.csv"
    df = pd.read_csv(ver_csv)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["id"] = df.index.astype(int)

    # Sample a small subset to save quota
    idxs = list(df.index)
    random.shuffle(idxs)
    df = df.loc[idxs[:args.max_pairs]].reset_index(drop=True)

    # Optional article captions for text queries
    art_csv = ver_root / "VERITE_articles.csv"
    articles_for_text = pd.read_csv(art_csv) if (args.use_article_captions and art_csv.exists()) else None
    if articles_for_text is not None and "id" in articles_for_text.columns:
        articles_for_text["id"] = pd.to_numeric(articles_for_text["id"], errors="coerce")

    # Articles for reverse fallback URLs (always load if present)
    articles_by_id: Dict[int, dict] = {}
    if art_csv.exists():
        art_df = pd.read_csv(art_csv)
        if "id" in art_df.columns:
            for _, arow in art_df.iterrows():
                try:
                    articles_by_id[int(pd.to_numeric(arow["id"], errors="coerce"))] = arow.to_dict()
                except Exception:
                    continue

    # Unique images (limit for reverse)
    unique_paths = df["image_path"].astype(str).unique().tolist()
    unique_paths = unique_paths[:args.max_unique_images]
    reverse_cache: Dict[str, Tuple[List[Dict], List[Dict]]] = {}

    # K override
    K_local = args.k_results if args.k_results is not None else K

    print(f"[{PROVIDER}] reverse-image for {len(unique_paths)} unique images (K={K_local}) …")
    for rel in tqdm(unique_paths):
        local = (ver_root / rel.strip("./")).resolve()
        if not local.exists():
            reverse_cache[rel] = ([], [])
            continue
        if args.skip_reverse:
            reverse_cache[rel] = ([], [])
            continue
        try:
            if PROVIDER == "serpapi":
                # find one VERITE id that uses this image (to grab article URLs if available)
                sample_row = df[df["image_path"] == rel].iloc[0]
                vid_for_this_img = int(sample_row["id"])
                art_row = articles_by_id.get(vid_for_this_img)
                pages, sims = serpapi_reverse_with_fallback(local, art_row, K_local)
            else:
                raise RuntimeError(f"Unknown provider: {PROVIDER}")
        except Exception as e:
            print(f"[warn] reverse search failed for {rel}: {e}")
            pages, sims = ([], [])
        reverse_cache[rel] = (pages, sims)
        time.sleep(0.6)  # gentle rate limiting

    rows = []
    print("Collecting per-pair evidence …")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        vid = int(r["id"])
        rel_img = str(r["image_path"])
        pair_dir = evidence_root / f"{vid:06d}"
        meta_fp = pair_dir / "meta.json"
        pages_json = pair_dir / "from_image" / "pages.json"

        if args.skip_existing and meta_fp.exists():
            # Reconstruct row cheaply from disk
            try:
                pages_info = []
                if pages_json.exists():
                    pages_info = json.loads(pages_json.read_text(encoding="utf-8")).get("pagesIncluding", [])
                captions_list = [p.get("title","") for p in pages_info if p.get("title")]
                images_list = []
                for sub in ("from_image/images", "from_text/images"):
                    d = pair_dir / sub
                    if d.exists():
                        images_list += [rel_to_repo(p) for p in d.glob("*.jpg")]
                rows.append({
                    "match_index": vid,
                    "captions": json.dumps(captions_list, ensure_ascii=False),
                    "images_paths": json.dumps(images_list, ensure_ascii=False),
                })
                continue
            except Exception:
                pass  # fall through to rebuild

        # Ensure dirs
        img_from_img_dir = pair_dir / "from_image" / "images"
        img_from_txt_dir = pair_dir / "from_text" / "images"
        img_from_img_dir.mkdir(parents=True, exist_ok=True)
        img_from_txt_dir.mkdir(parents=True, exist_ok=True)

        # Build text queries
        qlist = []
        cap = str(r.get("caption", "") or "").strip()
        if cap:
            qlist.append(cap)
        if articles_for_text is not None:
            a = articles_for_text[articles_for_text["id"] == vid]
            if len(a):
                for col in ("true_caption","false_caption","query"):
                    v = a.iloc[0].get(col, "")
                    if isinstance(v, str) and v.strip():
                        qlist.append(v.strip())
        # Deduplicate & filter trivial
        qlist = list(dict.fromkeys([q for q in qlist if len(q) > 3]))

        # Reverse image results
        pages, sims = reverse_cache.get(rel_img, ([], []))
        pages_info = []
        sim_img_paths = []
        captions_accum = []  # <-- collect titles from all sources

        # 1) Pages including (reverse)
        for p in pages[:K_local]:
            url = p.get("hostPageUrl")
            title = p.get("title") or fetch_title(url)
            if title:
                captions_accum.append(title)
            pages_info.append({"url": url, "title": title})

        # 2) Visual matches (reverse): save image + capture page title
        for s in sims[:K_local]:
            # page title from host page (prefer s['title'], else fetch)
            vm_title = s.get("title") or fetch_title(s.get("hostPageUrl", ""))
            if vm_title:
                captions_accum.append(vm_title)
            fp = save_image_from_url(s.get("contentUrl",""), img_from_img_dir)
            if fp:
                sim_img_paths.append(rel_to_repo(fp))

        # 3) Text -> image: first result per query; also capture page title
        txt_img_paths = []
        if PROVIDER == "serpapi":
            for q in qlist:
                try:
                    hits = serpapi_image_search(q, num=1)
                except Exception as e:
                    print(f"[warn] text search failed (id={vid}): {e}")
                    hits = []
                if hits:
                    h = hits[0]
                    # capture title (prefer hit title, else fetch host page)
                    hit_title = h.get("title") or fetch_title(h.get("hostPageUrl", ""))
                    if hit_title:
                        captions_accum.append(hit_title)
                    fp = save_image_from_url(h.get("contentUrl",""), img_from_txt_dir)
                    if fp:
                        txt_img_paths.append(rel_to_repo(fp))
                time.sleep(0.6)

        # Sidecar JSONs
        (pair_dir / "from_image").mkdir(parents=True, exist_ok=True)
        pages_json.write_text(
            json.dumps({"pagesIncluding": pages_info}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        meta_fp.write_text(
            json.dumps({
                "verite_id": vid,
                "source_image": str((ver_root / rel_img).resolve()),
                "text_queries": qlist,
                "n_similar_images": len(sim_img_paths),
                "n_text_images": len(txt_img_paths)
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Row for encoder step
        # Use unique, non-empty captions collected from all sources
        captions_list = list(dict.fromkeys([c for c in captions_accum if isinstance(c, str) and c.strip()]))
        images_list = sim_img_paths + txt_img_paths
        rows.append({
            "match_index": vid,
            "captions": json.dumps(captions_list, ensure_ascii=False),
            "images_paths": json.dumps(images_list, ensure_ascii=False),
        })

    out_csv = evidence_root / "collected_evidence.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Evidence saved under {rel_to_repo(evidence_root)}")
    print(f"✓ CSV for encoder: {rel_to_repo(out_csv)}")

if __name__ == "__main__":
    if PROVIDER == "serpapi" and not SERPAPI_KEY:
        raise SystemExit("Set SERPAPI_KEY in .env for provider=serpapi")
    main()
