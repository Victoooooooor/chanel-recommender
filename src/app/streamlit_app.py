import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder
from src.recommenders.cosine_knn import cosine_knn
from src.recommenders.hybrid_recommender import hybrid_recommender

# -----------------------
# Chargement data + modèles
# -----------------------

meta = pd.read_pickle(ROOT / "data" / "meta.pkl")
emb_vis = np.load(ROOT / "data" / "embeddings_clip.npy")
emb_txt = np.load(ROOT / "data" / "embeddings_text.npy")

img_encoder = ImageEncoder()
txt_encoder = TextEncoder()

st.title("Plateforme de recommandation Chanel")

mode = st.sidebar.selectbox(
    "Mode de recherche",
    ["Recherche par texte", "Recherche par image", "Recherche combinée"]
)

# -----------------------
# Recherche par TEXTE
# -----------------------

if mode == "Recherche par texte":
    txt = st.text_input("Décrivez un produit (en anglais ou allemand)")

    if txt:
        q_emb = txt_encoder.encode(txt)

        if q_emb is None:
            st.error("Impossible d'extraire un embedding textuel.")
            st.stop()

        idx, scores = cosine_knn(q_emb, emb_txt)

        valid = 0
        MAX_RESULTS = 10

        for i in idx:
            row = meta.iloc[i]
            img_path = ROOT / row.image_path

            if not img_path.exists():
                continue

            st.image(img_path, width=200)
            st.write(row.title_eng)
            valid += 1

            if valid >= MAX_RESULTS:
                break

        if valid == 0:
            st.warning("Aucun résultat avec image n'a été trouvé.")

# -----------------------
# Recherche par IMAGE
# -----------------------

elif mode == "Recherche par image":
    uploaded = st.file_uploader("Uploader une image", type=["jpg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=250)

        q_emb = img_encoder.encode(img)

        if q_emb is None:
            st.error("Impossible d'extraire un embedding visuel.")
            st.stop()

        idx, scores = cosine_knn(q_emb, emb_vis)

        for i in idx:
            row = meta.iloc[i]
            st.image(ROOT / row.image_path, width=200)
            st.write(row.title_eng)

# -----------------------
# Recherche COMBINÉE
# -----------------------

else:
    uploaded = st.file_uploader("Uploader une image")
    txt = st.text_input("Description textuelle")
    alpha = st.slider("Poids image vs texte", 0.0, 1.0, 0.5)

    if uploaded and txt:
        img = Image.open(uploaded).convert("RGB")

        img_emb = img_encoder.encode(img)
        txt_emb = txt_encoder.encode(txt)

        if img_emb is None or txt_emb is None:
            st.error("Impossible d'extraire les embeddings.")
            st.stop()

        idx, scores = hybrid_recommender(img_emb, txt_emb, emb_vis, emb_txt, alpha)

        for i in idx:
            row = meta.iloc[i]
            st.image(ROOT / row.image_path, width=200)
            st.write(row.title_eng)
