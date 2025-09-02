import torch
from torch.utils.data import Dataset, DataLoader

class TensorRows(Dataset):
    def __init__(self, df, img_map, txt_map, Ximg_map, Xtxt_map, use_evidence=1):
        self.df = df.reset_index(drop=True)
        self.img_map, self.txt_map = img_map, txt_map
        self.Ximg_map, self.Xtxt_map = Ximg_map, Xtxt_map
        self.k = use_evidence

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        s = self.df.iloc[i]
        img = torch.tensor(self.img_map[str(s.image_id)].values, dtype=torch.float32)
        txt = torch.tensor(self.txt_map[str(s.id)].values, dtype=torch.float32)
        lbl = torch.tensor(float(s.falsified), dtype=torch.float32)
        if self.k and len(s.img_ranked_items)>0:
            Ximg = torch.tensor(self.Ximg_map[s.img_ranked_items[:self.k]].T.values, dtype=torch.float32)
        else:
            Ximg = torch.zeros((self.k, img.shape[0]))
        if self.k and len(s.txt_ranked_items)>0:
            Xtxt = torch.tensor(self.Xtxt_map[s.txt_ranked_items[:self.k]].T.values, dtype=torch.float32)
        else:
            Xtxt = torch.zeros((self.k, img.shape[0]))

        return {"img": img, "txt": txt, "label": lbl, "Ximg": Ximg, "Xtxt": Xtxt}

def make_loader(df, img_map, txt_map, Ximg_map, Xtxt_map, batch_size=512, shuffle=False, use_evidence=1):
    ds = TensorRows(df, img_map, txt_map, Ximg_map, Xtxt_map, use_evidence)
    collate = lambda batch: {
        "img": torch.stack([b["img"] for b in batch]),
        "txt": torch.stack([b["txt"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "Ximg": torch.stack([b["Ximg"] for b in batch]),
        "Xtxt": torch.stack([b["Xtxt"] for b in batch]),
        "logits_in": lambda model, bt, device: (
            model(bt["img"].to(device), bt["txt"].to(device), bt["Ximg"].to(device), bt["Xtxt"].to(device)),
            bt["label"].to(device)
        )
    }
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=2, pin_memory=True)
