import yaml, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from ooclab.muse.mlp_model import MUSEMLP
from ooclab.train.loop import train_epoch, evaluate

cfg = yaml.safe_load(open("config/default.yaml"))
tag = cfg["encoder"]["version"].replace("-","").replace("/",""); outs = cfg["paths"]["outputs"]
device = torch.device(cfg["encoder"]["device"] if torch.cuda.is_available() else "cpu")

def load_muse(split):
    df = pd.read_csv(f"{outs}/muse_{split}_{tag}.csv") if False else None  # or compute on the fly
    return df

# Example with prebuilt dataframes `muse_train`, `muse_valid`, `muse_test`
def to_loader(df, sims, bs):
    X = torch.tensor(df[sims].values, dtype=torch.float32)
    y = torch.tensor(df["falsified"].values, dtype=torch.float32)
    ds = TensorDataset(X,y)
    return DataLoader(ds, batch_size=bs, shuffle=True)

def main(muse_train, muse_valid, muse_test, sims):
    model = MUSEMLP(in_dim=len(sims)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def wrap(dl):
        # tiny adapter so we can reuse train loop
        def _in(model, batch, device):
            X, y = batch[0].to(device), batch[1].to(device)
            return model(X), y
        dl.dataset.__class__.logits_in = staticmethod(_in)  # not elegant, but short
        return dl

    tr = wrap(to_loader(muse_train, sims, cfg["muse_mlp"]["batch_size"]))
    va = wrap(to_loader(muse_valid, sims, cfg["muse_mlp"]["batch_size"]))
    te = wrap(to_loader(muse_test,  sims, cfg["muse_mlp"]["batch_size"]))

    best, patience, best_state = -1, 10, None
    for _ in range(cfg["muse_mlp"]["epochs"]):
        train_epoch(model, tr, opt, device)
        scores = evaluate(model, va, device)
        if scores["Accuracy"] > best:
            best = scores["Accuracy"]; best_state = model.state_dict().copy(); patience = 10
        else:
            patience -= 1
        if patience == 0: break

    model.load_state_dict(best_state)
    print("Test:", evaluate(model, te, device))

if __name__ == "__main__":
    pass
