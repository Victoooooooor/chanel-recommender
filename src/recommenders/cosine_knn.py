import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_knn(query_emb, embeddings, k=10):
    sims = cosine_similarity([query_emb], embeddings)[0]
    idx = np.argsort(sims)[::-1][:k]
    return idx, sims[idx]
