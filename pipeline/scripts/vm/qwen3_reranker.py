"""Qwen3-VL-Reranker-2B as a fifth matching decision, scored with the official implementation
(github.com/QwenLM/Qwen3-VL-Embedding, src/models/qwen3_vl_reranker.py: sigmoid of the yes-minus-no logit at the last
position, default instruction, images at the implementation's pixel budget). Runs on the GPU server from /srv/gpu-server/qwen/scripts
with the qwen conda environment; outputs under scripts/out, appended as the run proceeds and resumed from.

Modes:
  svo_pool    every (query, candidate) pair of inputs/svo_pool_pairs.json; caption queries are the reranker's query
              and the image its document (t2i), image queries the reverse (i2t)        -> Qwen3R_pool_svo.csv
  coco_pool   the same for inputs/coco_pool_pairs.json                                -> Qwen3R_pool_coco.csv
  svo_bench   every SVO-Probes triplet (scripts/svo_triplets.json): the caption against its positive and its
              negative image, both as (text query, image document)                      -> Qwen3R_bench_svo.csv
  sugarcrepe  every SugarCrepe item (inputs/coco_inputs.json): the image against its caption and its negative
              caption, both as (image query, text document)                             -> Qwen3R_bench_sugarcrepe.csv
    /opt/anacondas/anaconda3/envs/qwen/bin/python scripts/qwen3_reranker.py <mode>"""
import csv
import json
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm

MODE = sys.argv[1]
BASE = Path(__file__).resolve().parent                      # /srv/gpu-server/qwen/scripts
QWEN = BASE.parent                                          # /srv/gpu-server/qwen
COCO = QWEN / "inputs"
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", "/srv/gpu-server/huggingface")
sys.path.insert(0, str(BASE / "Qwen3-VL-Embedding"))
from src.models.qwen3_vl_reranker import Qwen3VLReranker   # noqa: E402

# the local snapshot path: with the hub id, transformers 4.57.0 fails on the model's empty additional_chat_templates folder
SNAPSHOTS = sorted(Path(os.environ["HF_HUB_CACHE"], "models--Qwen--Qwen3-VL-Reranker-2B", "snapshots").glob("*"))
MODEL_NAME = str(SNAPSHOTS[-1]) if SNAPSHOTS else "Qwen/Qwen3-VL-Reranker-2B"
model = Qwen3VLReranker(model_name_or_path=MODEL_NAME, torch_dtype=torch.bfloat16)
INSTRUCTION = model.default_instruction


def svo_image(image_id):
    return str(QWEN / "images" / f"{int(image_id)}.jpg")


def coco_image(image_id):
    return str(COCO / "val2017" / f"{int(image_id):012d}.jpg")


def score(query, document):
    """query / document: {"text": ...} or {"image": path}; one relevance score in [0, 1], or "" on an error."""
    try:
        with torch.no_grad():
            return model.process({"instruction": INSTRUCTION, "query": query, "documents": [document]})[0]
    except Exception as e:                                  # unreadable image, CUDA error: recorded as unanswered
        print("error", query, document, repr(e)[:200], file=sys.stderr, flush=True)
        return ""


if MODE in ("svo_pool", "coco_pool"):
    pairs = json.load(open(COCO / ("svo_pool_pairs.json" if MODE == "svo_pool" else "coco_pool_pairs.json"), encoding="utf-8"))
    img_path = svo_image if MODE == "svo_pool" else coco_image
    jobs = []
    for d in ("t2i", "i2t"):
        for q, c in pairs[d]:
            img, cap = (int(c), str(q)) if d == "t2i" else (int(q), str(c))
            jobs.append((d, q, c, img, cap))
    header = ["direction", "query", "candidate", "image_id", "caption", "score"]
    out_csv = OUT / ("Qwen3R_pool_svo.csv" if MODE == "svo_pool" else "Qwen3R_pool_coco.csv")
elif MODE == "svo_bench":
    trip = json.load(open(BASE / "svo_triplets.json", encoding="utf-8"))
    jobs = [(k, t["caption"], t["pos"], t["neg"], t["neg_type"]) for k, t in enumerate(trip)]
    header = ["triplet_index", "caption", "pos_image", "neg_image", "neg_type", "score_pos", "score_neg"]
    out_csv = OUT / "Qwen3R_bench_svo.csv"
elif MODE == "sugarcrepe":
    inp = json.load(open(COCO / "coco_inputs.json", encoding="utf-8"))
    jobs = [(k, p["subset"], p["image_id"], p["caption"], p["negative_caption"]) for k, p in enumerate(inp["sugarcrepe"])]
    header = ["pair_index", "subset", "image_id", "caption", "negative_caption", "score_pos", "score_neg"]
    out_csv = OUT / "Qwen3R_bench_sugarcrepe.csv"
else:
    raise SystemExit("mode: svo_pool | coco_pool | svo_bench | sugarcrepe")

done = 0
if out_csv.exists():
    with open(out_csv, encoding="utf-8", newline="") as f:
        done = max(0, sum(1 for _ in csv.reader(f)) - 1)
print(MODE, len(jobs), "jobs,", done, "already done", flush=True)
f = open(out_csv, "a", encoding="utf-8", newline="")
w = csv.writer(f)
if done == 0:
    w.writerow(header)

for i in tqdm(range(done, len(jobs)), desc=f"Qwen3-VL-Reranker {MODE}", mininterval=10):
    j = jobs[i]
    if MODE in ("svo_pool", "coco_pool"):
        d, q, c, img, cap = j
        if d == "t2i":
            s = score({"text": cap}, {"image": img_path(img)})
        else:
            s = score({"image": img_path(img)}, {"text": cap})
        w.writerow([d, q, c, img, cap, s])
    elif MODE == "svo_bench":
        k, cap, pos, neg, neg_type = j
        w.writerow([k, cap, pos, neg, neg_type, score({"text": cap}, {"image": svo_image(pos)}), score({"text": cap}, {"image": svo_image(neg)})])
    else:
        k, subset, img, cap, negcap = j
        w.writerow([k, subset, img, cap, negcap, score({"image": coco_image(img)}, {"text": cap}), score({"image": coco_image(img)}, {"text": negcap})])
    if i % 50 == 0:
        f.flush()
        torch.cuda.empty_cache()
f.close()
print(MODE, "DONE", len(jobs), flush=True)
