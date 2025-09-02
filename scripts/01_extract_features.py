import os, yaml, pandas as pd
from ooclab.features.extract_main import extract_for_dataframe, save_embeddings

cfg = yaml.safe_load(open("config/default.yaml"))
encv = cfg["encoder"]["version"]; dev = cfg["encoder"]["device"]
root = cfg["paths"]["root"]; outs = cfg["paths"]["outputs"]

# Example: VisualNews merged DataFrame with columns: id, image_path, caption
df = pd.read_json(os.path.join(root, "visualnews", "origin", "data.json"))
ids, I, T = extract_for_dataframe(df, img_root=os.path.join(root, "visualnews", "origin"),
                                  id_col="id", img_path_col="image_path", text_col="caption",
                                  encoder_version=encv, device=dev, batch=cfg["run"]["batch_size"])
save_embeddings(os.path.join(outs, "vn"), "visualnews", ids, I, T, encv)
