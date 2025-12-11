import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  
sys.path.append(str(ROOT))                

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder
from src.recommenders.cosine_knn import cosine_knn
from src.recommenders.hybrid_recommender import hybrid_recommender


# Load metadata & embeddings
meta = pd.read_pickle(ROOT / "data" / "meta.pkl")
# emb_vis = np.load(ROOT / "data" / "embeddings_vision.npy")
emb_txt = np.load(ROOT / "data" / "embeddings_text.npy")

# Load models
img_encoder = ImageEncoder()
txt_encoder = TextEncoder()

st.title("Plateforme de recommandation Chanel")

mode = st.sidebar.selectbox(
    "Mode de recherche",
    # ["Recherche par image", "Recherche par texte", "Recherche combinée"]
    ["Recherche par texte"]
)

# ----------------- Option 1 : IMAGE -----------------
# if mode == "Recherche par image":
#     uploaded = st.file_uploader("Uploader une image", type=["jpg","png"])
#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         st.image(img, width=250)

#         q_emb = img_encoder.encode(img)
#         idx, scores = cosine_knn(q_emb, emb_vis)

#         st.subheader("Résultats")
#         for i in idx:
#             row = meta.iloc[i]
#             st.image(ROOT / row.image_path, width=200)
#             st.write(row.title_eng)

# ----------------- Option 2 : TEXTE -----------------
if mode == "Recherche par texte":
    txt = st.text_input("Décrivez un produit")
    if txt:
        q_emb = txt_encoder.encode(txt)
        idx, scores = cosine_knn(q_emb, emb_txt)

        MAX_RESULTS = 10   # nombre de résultats à afficher maximum
        valid_results = 0  # compteur d'images trouvées

        for i in idx:
            row = meta.iloc[i]
            img_path = ROOT / row.image_path

            # Vérification : image présente ?
            if not img_path.exists():
                continue   # on saute cet item

            # On affiche le produit
            st.image(img_path, width=200)
            st.write(row.title_eng)

            valid_results += 1

            if valid_results >= MAX_RESULTS:
                break

        # Si aucun produit affiché :
        if valid_results == 0:
            st.warning("Aucun résultat correspondant avec une image disponible.")


else:
    st.write("Sélectionnez un mode de recherche dans la barre latérale.")

# ----------------- Option 3 : COMBINÉ -----------------
# else:
#     uploaded = st.file_uploader("Uploader une image")
#     txt = st.text_input("Description")
#     alpha = st.slider("Poids vision / texte", 0.0, 1.0, 0.5)

#     if uploaded and txt:
#         img = Image.open(uploaded).convert("RGB")
#         img_emb = img_encoder.encode(img)
#         txt_emb = txt_encoder.encode(txt)

#         idx, scores = hybrid_recommender(img_emb, txt_emb, emb_vis, emb_txt, alpha)

#         for i in idx:
#             row = meta.iloc[i]
#             st.image(ROOT / row.image_path, width=200)
#             st.write(row.title_eng)
