import yaml, numpy as np, pandas as pd
from ooclab.muse.sims import build_muse_table, SIM_NAMES
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

cfg = yaml.safe_load(open("config/default.yaml"))
tag = cfg["encoder"]["version"].replace("-","").replace("/",""); outs = cfg["paths"]["outputs"]

def load_maps():
    ids = np.load(f"{outs}/vn/visualnews_item_ids_{tag}.npy")
    I   = np.load(f"{outs}/vn/visualnews_clip_image_embeddings_{tag}.npy").T
    T   = np.load(f"{outs}/vn/visualnews_clip_text_embeddings_{tag}.npy").T
    img_map = pd.DataFrame(I, index=ids).T; img_map.columns = img_map.columns.astype(str)
    txt_map = pd.DataFrame(T, index=ids).T; txt_map.columns = txt_map.columns.astype(str)

    Xti = np.load(f"{outs}/evidence/X_text_ids_{tag}.npy")
    Xte = np.load(f"{outs}/evidence/X_text_embeddings_{tag}.npy").T
    Xii = np.load(f"{outs}/evidence/X_image_ids_{tag}.npy")
    Xie = np.load(f"{outs}/evidence/X_image_embeddings_{tag}.npy").T
    Xtxt_map = pd.DataFrame(Xte, index=Xti).T; Xtxt_map.columns = Xtxt_map.columns.astype(str)
    Ximg_map = pd.DataFrame(Xie, index=Xii).T; Ximg_map.columns = Ximg_map.columns.astype(str)
    return img_map, txt_map, Ximg_map, Xtxt_map

def run_rf_dt(train, test, sims=SIM_NAMES):
    Xtr, ytr = train[sims].values, train["falsified"].values
    Xte, yte = test[sims].values, test["falsified"].values
    for name, model in [("DT", DecisionTreeClassifier(max_depth=7, random_state=0)),
                        ("RF", RandomForestClassifier(random_state=0))]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        print(name, "Acc:", round(metrics.accuracy_score(yte, pred)*100, 2))

def main():
    img_map, txt_map, Ximg_map, Xtxt_map = load_maps()
    train = pd.read_csv(f"{outs}/ranked_train_{tag}.csv")
    test  = pd.read_csv(f"{outs}/ranked_test_{tag}.csv")

    muse_train = build_muse_table(train, img_map, txt_map, Ximg_map, Xtxt_map, use_evidence=1)
    muse_test  = build_muse_table(test,  img_map, txt_map, Ximg_map, Xtxt_map, use_evidence=1)
    run_rf_dt(muse_train, muse_test)

if __name__ == "__main__":
    main()
