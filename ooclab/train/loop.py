import time, numpy as np, torch, torch.nn.functional as F
from sklearn import metrics

def train_epoch(model, loader, opt, device):
    model.train(); epoch_loss = 0.0
    for batch in loader:
        opt.zero_grad()
        logits, labels = batch["logits_in"](model, batch, device)  # see loaders/tensors.py
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward(); opt.step()
        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        logits, labels = batch["logits_in"](model, batch, device)
        ys.append(labels.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    return {
        "Accuracy": metrics.accuracy_score(y, pred),
        "AUC": metrics.roc_auc_score(y, p),
        "Precision": metrics.precision_score(y, pred, zero_division=0),
        "Recall": metrics.recall_score(y, pred, zero_division=0),
        "F1": metrics.f1_score(y, pred, zero_division=0),
    }
