import torch
import clip
from PIL import Image
import numpy as np

class ImageEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode(self, img):
        try:
            img_t = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(img_t)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            emb = emb.cpu().numpy()[0]

            if np.isnan(emb).any():
                return None

            return emb

        except Exception as e:
            print("Erreur ImageEncoder:", e)
            return None
