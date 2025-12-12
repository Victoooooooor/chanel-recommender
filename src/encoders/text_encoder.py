from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class TextEncoder:
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

            emb = pooled.numpy()[0]

            if np.isnan(emb).any():
                return None

            return emb

        except Exception as e:
            print("Erreur TextEncoder:", e)
            return None
