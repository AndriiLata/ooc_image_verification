
from typing import Tuple, List
import torch
import open_clip
from PIL import Image

class ClipEncoder:
    def __init__(self, version: str = "ViT-L/14", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_name = "ViT-L-14" if "L/14" in version else "ViT-B-32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)

    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            batch = torch.stack([self.preprocess(img.convert("RGB")) for img in pil_images]).to(self.device)
            feats = self.model.encode_image(batch)
            return feats.float()

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            toks = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(toks)
            return feats.float()
