# scripts/00_collect_evidence.py
import os, io, json, time, hashlib, yaml, argparse, random
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
ver_root = (REPO_ROOT / cfg["paths"]["verite"]).resolve()

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
def serpapi_image_search(query: str, num: int) -> List[Dict]:
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

def serpapi_reverse_from_image_url(image_url: str, num: int) -> Tuple[List[Dict], List[Dict]]:
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
    ap.add_argument("--max_pairs", type=int, default=10)
    ap.add_argument("--max_unique_images", type=int, default=10)
    ap.add_argument("--k_results", type=int, default=None)
    ap.add_argument("--use_article_captions", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--skip_reverse", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ----------------- main -----------------
def main():
    if PROVIDER != "serpapi":
        raise SystemExit(f"Unknown provider: {PROVIDER}")
    if not SERPAPI_KEY:
        raise SystemExit("Set SERPAPI_KEY in .env for provider=serpapi")

    args = parse_args()
    random.seed(args.seed)

    # Load VERITE rows
    ver_csv = ver_root / "VERITE.csv"
    df = pd.read_csv(ver_csv)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["id"] = df.index.astype(int)

    # Optional article captions (for text queries)
    art_csv = ver_root / "VERITE_articles.csv"
    articles_for_text = pd.read_csv(art_csv) if (args.use_article_captions and art_csv.exists()) else None
    if articles_for_text is not None and "id" in articles_for_text.columns:
        articles_for_text["id"] = pd.to_numeric(articles_for_text["id"], errors="coerce")

    # Article URLs (for reverse)
    articles_by_id: Dict[int, dict] = {}
    if art_csv.exists():
        art_df = pd.read_csv(art_csv)
        if "id" in art_df.columns:
            for _, arow in art_df.iterrows():
                try:
                    articles_by_id[int(pd.to_numeric(arow["id"], errors="coerce"))] = arow.to_dict()
                except Exception:
                    continue

    # sample subset
    idxs = list(df.index)
    random.shuffle(idxs)
    df = df.loc[idxs[:args.max_pairs]].reset_index(drop=True)

    # Unique images (limit reverse)
    unique_paths = df["image_path"].astype(str).unique().tolist()
    unique_paths = unique_paths[:args.max_unique_images]
    reverse_cache: Dict[str, Tuple[List[Dict], List[Dict]]] = {}

    K_local = args.k_results if args.k_results is not None else K

    print(f"[serpapi] reverse-image via image_url for {len(unique_paths)} unique images (K={K_local}) …")
    for rel in tqdm(unique_paths):
        # find any VERITE id that uses this image to get its true/false URL
        try:
            sample_row = df[df["image_path"] == rel].iloc[0]
            vid = int(sample_row["id"])
            art_row = articles_by_id.get(vid, {})
            # prefer true_url, then false_url
            image_url = str(art_row.get("true_url") or art_row.get("false_url") or "").strip()
        except Exception:
            image_url = ""

        if (not image_url) or args.skip_reverse:
            reverse_cache[rel] = ([], [])
            continue

        try:
            pages, sims = serpapi_reverse_from_image_url(image_url, K_local)
        except Exception as e:
            print(f"[warn] reverse failed for {rel}: {e}")
            pages, sims = ([], [])
        reverse_cache[rel] = (pages, sims)
        time.sleep(0.6)

    rows = []
    print("Collecting per-pair evidence …")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        vid = int(r["id"])
        rel_img = str(r["image_path"])

        pair_dir = evidence_root / f"{vid:06d}"
        meta_fp = pair_dir / "meta.json"
        pages_json = pair_dir / "from_image" / "pages.json"

        if args.skip_existing and meta_fp.exists():
            try:
                pages_info = []
                if pages_json.exists():
                    pages_info = json.loads(pages_json.read_text(encoding="utf-8")).get("pagesIncluding", [])
                captions_list = [p.get("title","") or p.get("snippet","") for p in pages_info if (p.get("title") or p.get("snippet"))]
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
                pass

        # dirs
        img_from_img_dir = pair_dir / "from_image" / "images"
        img_from_txt_dir = pair_dir / "from_text" / "images"
        img_from_img_dir.mkdir(parents=True, exist_ok=True)
        img_from_txt_dir.mkdir(parents=True, exist_ok=True)

        # build text queries
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
        qlist = list(dict.fromkeys([q for q in qlist if len(q) > 3]))

        # reverse results from cache
        pages, sims = reverse_cache.get(rel_img, ([], []))
        pages_info, sim_img_paths, captions_accum = [], [], []

        # pages including
        for p in pages[:K_local]:
            url = p.get("hostPageUrl")
            title = p.get("title") or fetch_title(url)
            snippet = p.get("snippet") or ""
            if title:
                captions_accum.append(title)
            elif snippet:
                captions_accum.append(snippet)
            pages_info.append({"url": url, "title": title, "snippet": snippet})

        # visual matches
        for s in sims[:K_local]:
            vm_title = s.get("title") or fetch_title(s.get("hostPageUrl", ""))
            if vm_title:
                captions_accum.append(vm_title)
            fp = save_image_from_url(s.get("contentUrl",""), img_from_img_dir)
            if fp:
                sim_img_paths.append(rel_to_repo(fp))

        # text -> image (first per query)
        txt_img_paths = []
        for q in qlist:
            try:
                hits = serpapi_image_search(q, num=1)
            except Exception as e:
                print(f"[warn] text search failed (id={vid}): {e}")
                hits = []
            if hits:
                h = hits[0]
                hit_title = h.get("title") or fetch_title(h.get("hostPageUrl", ""))
                if hit_title:
                    captions_accum.append(hit_title)
                fp = save_image_from_url(h.get("contentUrl",""), img_from_txt_dir)
                if fp:
                    txt_img_paths.append(rel_to_repo(fp))
            time.sleep(0.6)

        # write sidecars
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

        # row for encoder step
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
    main()
