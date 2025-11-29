import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset


def download_image(url: str, path: str) -> bool:
    """
    Télécharge une image depuis une URL et la sauvegarde en local.

    Args:
        url (str): URL Cloudinary de l'image.
        path (str): chemin cible du fichier .jpg.

    Returns:
        bool: True si le téléchargement a réussi, False sinon.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(path)
        return True

    except Exception as e:
        print(f"[ERROR] Impossible de télécharger {url}: {e}")
        return False


def download_all_images(df: pd.DataFrame, output_dir: str = "data/images"):
    """
    Télécharge toutes les images du dataset dans un dossier local.

    Args:
        df (pd.DataFrame): DataFrame contenant au moins la colonne 'imageurl'
        output_dir (str): Dossier de sortie où sauvegarder les images
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Téléchargement de {len(df)} images dans {output_dir}/ ...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row["imageurl"]
        filename = os.path.join(output_dir, f"{idx}.jpg")

        download_image(url, filename)

    print("Téléchargement terminé.")


if __name__ == "__main__":
    # Chargement du dataset HuggingFace

    print("Chargement du dataset Chanel...")
    ds = load_dataset("DBQ/Chanel.Product.prices.Germany")
    df = ds["train"].to_pandas()

    download_all_images(df)