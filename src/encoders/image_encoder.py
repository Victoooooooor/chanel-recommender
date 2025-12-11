import torch
import clip
from PIL import Image

class ImageEncoder:
    # def __init__(self, device="cpu"):
    #     self.device = device
    #     self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    # def encode(self, img):
    #     img_t = self.preprocess(img).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         emb = self.model.encode_image(img_t)
    #     return emb.cpu().numpy()[0]

    def __init__(self):
        pass
    
    def encode(self, img):
        # Encodeur image désactivé temporairement
        return None
