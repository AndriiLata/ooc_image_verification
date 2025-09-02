import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def _rank(query_vec: np.ndarray, candidates: np.ndarray, cand_ids: list, topk: int = 5):
    sims = cosine_similarity(query_vec.reshape(1,-1), candidates)[0]
    idx = np.argsort(-sims)[:topk]
    return [cand_ids[i] for i in idx], sims[idx].tolist()

def rank_evidence(samples_df: pd.DataFrame,
                  q_img_map: pd.DataFrame, q_txt_map: pd.DataFrame,
                  ev_img_mat: np.ndarray, ev_img_ids: list,
                  ev_txt_mat: np.ndarray, ev_txt_ids: list,
                  topk: int = 1):
    """
    samples_df must have columns: id, image_id (string ids).
    q_img_map/q_txt_map are DataFrames (dims x items) indexed by embedding dim.
    """
    out_rows = []
    for _, r in samples_df.iterrows():
        qid = str(r["id"]); iid = str(r["image_id"])
        q_img = q_img_map[qid].values
        q_txt = q_txt_map[iid].values  # consistent with their code: text keyed by id, image by image_id

        img_rank, img_s = _rank(q_img, ev_img_mat, ev_img_ids, topk)
        txt_rank, txt_s = _rank(q_txt, ev_txt_mat, ev_txt_ids, topk)
        out_rows.append({
            "id": qid, "image_id": iid,
            "img_ranked_items": img_rank, "img_sim_scores": img_s,
            "txt_ranked_items": txt_rank, "txt_sim_scores": txt_s
        })
    return pd.DataFrame(out_rows)
