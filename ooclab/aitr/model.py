import torch, torch.nn as nn, torch.nn.functional as F

def combine_features(a, b):
    # concat + add + sub + mul (same as paper’s idea)
    cat = torch.cat([a, b], dim=1)
    add = a + b
    sub = a - b
    mul = a * b
    return torch.cat([cat, add, sub, mul], dim=1)

class AITR(nn.Module):
    def __init__(self, emb_dim=768, heads=[1,2,4,8], tf_dim=1024,
                 use_muse=True, use_evidence=1, pooling="attention"):
        super().__init__()
        self.use_muse = use_muse
        self.use_evidence = use_evidence
        token_dim = emb_dim
        self.cls_token = nn.Parameter(torch.randn(token_dim))

        layers = []
        for h in heads:
            layers.append(nn.TransformerEncoderLayer(d_model=token_dim, nhead=h,
                                                     dim_feedforward=tf_dim, batch_first=True, norm_first=True))
        self.tf = nn.ModuleList(layers)
        self.pooling = pooling
        if pooling == "weighted":
            self.wp = nn.Linear(token_dim, token_dim)
            self.softmax = nn.Softmax(dim=-1)
        elif pooling == "attention":
            self.q = nn.Linear(token_dim, token_dim)
            self.k = nn.Linear(token_dim, token_dim)
            self.v = nn.Linear(token_dim, token_dim)
            self.softmax = nn.Softmax(dim=-1)

        # simple projection after TF
        self.clf = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 1))

    def forward(self, img, txt, X_img=None, X_txt=None, muse_vec=None):
        B = img.size(0)
        fused = combine_features(img, txt).view(B, -1, img.size(1))  # [B, 4, D]
        x = torch.cat([self.cls_token.expand(B,1,-1), fused], dim=1)
        if self.use_evidence and X_img is not None and X_txt is not None:
            x = torch.cat([x, X_img, X_txt], dim=1)  # evidence already [B, k, D]

        for layer in self.tf:
            x = layer(x)
        # pool across layers’ cls outputs (simplified: pool across sequence positions of final layer)
        if self.pooling == "max":
            x, _ = torch.max(x, dim=1)
        elif self.pooling == "weighted":
            w = self.softmax(self.wp(x))
            x = torch.sum(w * x, dim=1)
        else:
            Q, K, V = self.q(x), self.k(x), self.v(x)
            att = self.softmax(Q @ K.transpose(-2,-1) / (Q.size(-1)**0.5))
            x = (att @ V).mean(1)

        # Optionally concatenate MUSE signal
        if self.use_muse and muse_vec is not None:
            # project muse to same size and add
            x = x + muse_vec

        return self.clf(x).squeeze(1)  # logits
