import os
import random
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from Qwen3_VL_Embedding.src.models.qwen3_vl_embedding import Qwen3VLEmbedder


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
CSV_PATH = "filtered_svo.csv"
IMAGE_FOLDER = "images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Qwen3-VL-Embedding...")
embedder = Qwen3VLEmbedder(
    MODEL_NAME,
    dtype=torch.float16
)
print("Model loaded.")


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def normalize(x):
    return F.normalize(x, dim=1)

def encode_text(text):
    """Encode caption into embedding."""
    with torch.no_grad():
        emb = embedder.process([{"text": text, "instruction": "Represent the given answer for binary classification"}])
        emb = torch.tensor(emb)
        emb = normalize(emb)

    return emb[0]

def encode_pair(text, image_path):
    """Encode caption and image into embedding."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    except Exception as e:
        print(f"Image error: {image_path}", e)
        return None

    inputs = [{"text": text, "image": [img], "instruction": "Represent the given image and caption for binary classification to determine whether they match or not"}]

    with torch.no_grad():
        emb = embedder.process(inputs)
        emb = torch.tensor(emb)
        emb = normalize(emb)

    return emb[0]


# ============================================================
# PAIRWISE EVALUATION (EMBEDDING VERSION)
# ============================================================

def ask_model_itm_embed(pos_path, neg_path, caption):
    try:
        pos_emb = encode_pair(caption, pos_path)
        neg_emb = encode_pair(caption, neg_path)

        yes_emb = encode_text("Yes")
        no_emb = encode_text("No")

        # cosine similarity for POS image
        sim_pos_yes = torch.dot(pos_emb, yes_emb).item()
        sim_pos_no = torch.dot(pos_emb, no_emb).item()

        prediction_pos = sim_pos_yes > sim_pos_no

        # cosine similarity for NEG image
        sim_neg_yes = torch.dot(neg_emb, yes_emb).item()
        sim_neg_no = torch.dot(neg_emb, no_emb).item()

        prediction_neg = sim_neg_no > sim_neg_yes

        return prediction_pos, prediction_neg

    except Exception as e:
        print("Error:", e)
        return None, None

    finally:
        torch.cuda.empty_cache()
        gc.collect()

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

results = []
start_idx = 0
print("Starting from scratch.")


# ============================================================
# MAIN LOOP
# ============================================================

for i, row in tqdm(
    df.iloc[start_idx:].iterrows(),
    total=len(df) - start_idx,
):

    sentence = row["sentence"]

    pos_path = os.path.join(
        IMAGE_FOLDER, f"{row['pos_image_id']}.jpg"
    )
    neg_path = os.path.join(
        IMAGE_FOLDER, f"{row['neg_image_id']}.jpg"
    )

    pred_pos, pred_neg = ask_model_itm_embed(
        pos_path,
        neg_path,
        sentence,
    )

    results.append({
        "index": i,
        "correct_pos": pred_pos == True,
        "correct_neg": pred_neg == True,
        "subj_neg": row["subj_neg"],
        "verb_neg": row["verb_neg"],
        "obj_neg": row["obj_neg"],
    })


# ============================================================
# FINAL SAVE
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(
    "svo_probes_results_pairwise_qwen3_embed.csv",
    index=False,
)

# ============================================================
# METRICS
# ============================================================

overall_acc = results_df[["correct_pos", "correct_neg"]].to_numpy().mean()

group_acc = {}
for col in ["subj_neg", "verb_neg", "obj_neg"]:
    subset = results_df[results_df[col] == True]
    group_acc[col] = (
        subset[["correct_pos", "correct_neg"]].to_numpy().mean()
        if not subset.empty else None
    )

print("\nOverall accuracy:", overall_acc)
print("Accuracy by corruption type:")
print(group_acc)