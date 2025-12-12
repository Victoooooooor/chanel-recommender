import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_recommender(img_emb, txt_emb, emb_vis, emb_txt, alpha=0.5, k=10):
    if img_emb is None or txt_emb is None:
        return [], []

    sim_v = cosine_similarity([img_emb], emb_vis)[0]
    sim_t = cosine_similarity([txt_emb], emb_txt)[0]

    score = alpha * sim_v + (1 - alpha) * sim_t
    idx = np.argsort(score)[::-1][:k]

    return idx, score[idx]
