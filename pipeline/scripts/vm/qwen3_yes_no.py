"""Qwen3-VL-Embed yes/no matching decisions on the second collection, the same way as on SVO-Probes
(data/qwen3_embed/qwen3_svo_pairwise.py, which despite its name is the ITM run): the joint image+caption embedding
under the instruction "Represent the given image and caption for binary classification to determine whether they
match or not" is compared with the embeddings of "Yes" and "No" under "Represent the given answer for binary
classification"; a match when sim(Yes) > sim(No). Images resized to 1024x1024, fp16.

Usage: python scripts/26_coco_itm_qwen3.py sugarcrepe
           every SugarCrepe pair as (image, caption) -> expected yes and (image, negative caption) -> expected no;
           writes results/coco_itm_preds_Qwen3_sugarcrepe.csv and the summary into results/coco_itm.json
       python scripts/26_coco_itm_qwen3.py pool
           every pair of results/coco_pool_pairs.json (scripts/24_coco_export_inputs.py --pool);
           writes results/coco_itm_preds_Qwen3_pool.csv (scored against labels later)
The CSV is appended as the run proceeds and the run resumes from it."""
import csv
import json
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from svo_eval.paths import RESULTS, DATA
from svo_eval.coco import load_coco, SUBSETS
from svo_eval import metrics as M

sys.path.insert(0, str(DATA / "qwen3_vl_embedding"))
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "sugarcrepe"
MODEL = "Qwen3"
IMAGE_SIZE = 1024
BS = 4
INSTR_PAIR = "Represent the given image and caption for binary classification to determine whether they match or not"
INSTR_ANSWER = "Represent the given answer for binary classification"

coll = load_coco()
embedder = Qwen3VLEmbedder(str(DATA / "models" / "qwen3-vl-embedding-2b"), dtype=torch.float16)


def enc(batch):
    with torch.no_grad():
        e = embedder.process(batch)
    return torch.as_tensor(e).float().cpu().numpy()


yes_no = enc([{"text": "Yes", "instruction": INSTR_ANSWER}, {"text": "No", "instruction": INSTR_ANSWER}])
yes_vec, no_vec = yes_no[0], yes_no[1]

# jobs: (key fields..., image_id, text)
if MODE == "sugarcrepe":
    jobs = []
    for k, p in enumerate(coll.probe):
        jobs.append((k, p["subset"], p["image_id"], p["caption"], True))
        jobs.append((k, p["subset"], p["image_id"], p["negative_caption"], False))
    header = ["pair_index", "subset", "image_id", "text", "is_positive", "sim_yes", "sim_no", "pred_yes"]
    out_csv = RESULTS / f"coco_itm_preds_{MODEL}_sugarcrepe.csv"
elif MODE in ("pool", "svo_pool"):
    # pool: the COCO pool; svo_pool: the whole SVO-Probes judged pool (results/svo_pool_pairs.json), so that the
    # Qwen3 row of the SVO-Probes pool table covers the pairs retrieved by other systems as well
    pairs = json.load(open(RESULTS / ("coco_pool_pairs.json" if MODE == "pool" else "svo_pool_pairs.json"), encoding="utf-8"))
    jobs = []
    for d in ("t2i", "i2t"):
        for q, c in pairs[d]:
            img, cap = (int(c), str(q)) if d == "t2i" else (int(q), str(c))
            jobs.append((d, q, c, img, cap))
    header = ["direction", "query", "candidate", "image_id", "caption", "sim_yes", "sim_no", "pred_yes"]
    out_csv = RESULTS / (f"coco_itm_preds_{MODEL}_pool.csv" if MODE == "pool" else f"itm_pool_preds_{MODEL}_full.csv")
else:
    raise SystemExit("mode: sugarcrepe | pool | svo_pool")
if MODE == "svo_pool":
    from svo_eval.paths import IMAGES_DIR
    coll.image_path = lambda image_id: IMAGES_DIR / f"{int(image_id)}.jpg"      # SVO-Probes images

done = 0
if out_csv.exists():
    with open(out_csv, encoding="utf-8", newline="") as f:
        done = max(0, sum(1 for _ in f) - 1)
print(MODE, len(jobs), "pairs,", done, "already done")
f = open(out_csv, "a", encoding="utf-8", newline="")
w = csv.writer(f)
if done == 0:
    w.writerow(header)


def image_of(image_id):
    return Image.open(coll.image_path(image_id)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def joint_inputs(batch):
    if MODE == "sugarcrepe":
        return [{"text": text, "image": [image_of(img)], "instruction": INSTR_PAIR} for _, _, img, text, _ in batch]
    return [{"text": cap, "image": [image_of(img)], "instruction": INSTR_PAIR} for _, _, _, img, cap in batch]


for i in tqdm(range(done, len(jobs), BS), desc=f"{MODEL} ITM {MODE}"):
    batch = jobs[i:i + BS]
    try:
        emb = enc(joint_inputs(batch))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        emb = np.concatenate([enc(joint_inputs([j])) for j in batch])
    sy, sn = emb @ yes_vec, emb @ no_vec
    for j, a, b in zip(batch, sy, sn):
        w.writerow(list(j[:5]) + [f"{a:.6f}", f"{b:.6f}", int(a > b)])
    f.flush()
f.close()

if MODE == "sugarcrepe":
    import pandas as pd
    df = pd.read_csv(out_csv)
    pred = df["pred_yes"].astype(bool).values
    pos = df["is_positive"].astype(str).str.lower().isin(["true", "1"]).values
    correct = pred == pos
    res = {"overall": M.summary(correct), "tpr": M.summary(pred[pos]), "tnr": M.summary(~pred[~pos]),
           "share_predicted_match": M.summary(pred), "n_pairs": int(len(df))}
    for sub in SUBSETS:
        m = (df["subset"] == sub).values
        if m.any():
            res[sub] = M.summary(correct[m])
    path = RESULTS / "coco_itm.json"
    existing = json.load(open(path)) if path.exists() else {}
    existing.setdefault(MODEL, {})["sugarcrepe"] = res
    json.dump(existing, open(path, "w"), indent=1)
    print(MODEL, "SugarCrepe ITM %.2f  TPR %.1f  TNR %.1f  n=%d" % (res["overall"]["mean"], res["tpr"]["mean"], res["tnr"]["mean"], len(df)))
    print("saved", path)
