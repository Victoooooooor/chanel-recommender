from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class TextEncoder:
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pooling = moyenne (comme dans ta Partie 3)
        last_hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-10)

        return pooled.numpy()[0]
