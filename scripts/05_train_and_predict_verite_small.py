#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

def main():
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
    tag = cfg["encoder"]["version"].replace("-", "").replace("/", "")
    outs_root = REPO_ROOT / cfg["paths"]["outputs"]

    fp = outs_root / "verite_small" / f"decision_features_verite_small_{tag}.csv"
    if not fp.exists():
        raise SystemExit(f"Missing: {fp} (run 04_build_decision_features_verite_small.py first)")

    df = pd.read_csv(fp)

    # Features you created
    feature_cols = [
        "sim_img_txt",
        "sim_img_top_txt",
        "sim_txt_top_img",
        "sim_txt_top_txt",
        "sim_img_top_img",
        "gap_img_eviTxt",
        "gap_txt_eviImg",
        "gap_txt_eviTxt",
        "gap_img_eviImg",
    ]

    # Replace missing with 0 (simple baseline)
    X = df[feature_cols].fillna(0.0).values
    y = df["falsified"].values

    # Tiny dataset => this is just a sanity check, not real science yet
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.33, random_state=0, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Simple fast model
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Predict
    prob = clf.predict_proba(X)[:, 1]   # probability of falsified
    pred = (prob >= 0.5).astype(int)

    df_out = df.copy()
    df_out["pred_falsified"] = pred
    df_out["pred_prob_falsified"] = prob
    df_out["pred_label"] = df_out["pred_falsified"].map({0: "TRUE", 1: "MISUSE"})

    # Evaluate on test split
    test_pred = pred[idx_test]
    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    print(f"Accuracy (toy split): {acc:.3f}")
    print(f"F1 (toy split):       {f1:.3f}")
    print("\nReport:\n", classification_report(y, pred, target_names=["TRUE","MISUSE"]))

    out_fp = outs_root / "verite_small" / f"predictions_verite_small_{tag}.csv"
    df_out.to_csv(out_fp, index=False)
    print(f"\n✓ Wrote predictions: {out_fp}")
    print("\nSample outputs:")
    print(df_out[["id","label","falsified","pred_label","pred_prob_falsified"]].head(12))

if __name__ == "__main__":
    main()
