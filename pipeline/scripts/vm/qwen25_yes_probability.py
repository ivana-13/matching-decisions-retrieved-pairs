"""The zero-shot Qwen2.5-VL-7B-Instruct yes/no prompt of inputs/qwen25_coco_prompted.py, with a score instead of a
decoded answer: the probability mass of the "yes" tokens against the "no" tokens at the first generated position,
p_yes = P(yes) / (P(yes) + P(no)), with the same unsloth 4-bit load, prompt and images. The greedy answer of the earlier
runs is the token with the highest probability, so the decision at p_yes > 0.5 reproduces it up to the rare answers that
start with another token.

Modes (outputs under scripts/out, appended and resumed):
  svo_pool    inputs/svo_pool_pairs.json, images /srv/gpu-server/qwen/images/{id}.jpg   -> Qwen25p_pool_svo.csv
  coco_pool   inputs/coco_pool_pairs.json, images inputs/val2017                 -> Qwen25p_pool_coco.csv
    /opt/anacondas/anaconda3/envs/qwen/bin/python scripts/qwen25_yes_probability.py <mode>"""
import csv
import json
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoProcessor

MODE = sys.argv[1]
BASE = Path(__file__).resolve().parent
QWEN = BASE.parent
COCO = QWEN / "inputs"
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", "/srv/gpu-server/huggingface")
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=512, dtype=None, device_map="auto", load_in_4bit=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_disable()
model.eval()

ITM_PROMPT = ("You are an image caption expert. Does this image match the caption: '{caption}'? Answer yes only if "
              "everything from the caption - subject, verb and object - matches the image, otherwise answer no.")
tok = processor.tokenizer


def single_tokens(strings):
    """Token ids of the spellings that the tokenizer encodes as one token."""
    ids = set()
    for s in strings:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


YES = single_tokens(("yes", "Yes", " yes", " Yes", "YES"))
NO = single_tokens(("no", "No", " no", " No", "NO"))
print("yes tokens", YES, [tok.decode([i]) for i in YES], "| no tokens", NO, [tok.decode([i]) for i in NO], flush=True)


def image_path(image_id):
    return str(QWEN / "images" / f"{int(image_id)}.jpg") if MODE == "svo_pool" else str(COCO / "val2017" / f"{int(image_id):012d}.jpg")


def yes_probability(image_id, caption):
    """(p_yes, P(yes tokens), P(no tokens), top token) at the first generated position; None on an error."""
    try:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path(image_id)},
                                                 {"type": "text", "text": ITM_PROMPT.format(caption=caption)}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, add_generation_prompt=True,
                                               return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1)
        p_yes, p_no = float(p[YES].sum()), float(p[NO].sum())
        return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else None, p_yes, p_no, tok.decode([int(p.argmax())])
    except Exception as e:
        print("error", image_id, repr(e)[:200], file=sys.stderr, flush=True)
        return None, None, None, ""


if MODE not in ("svo_pool", "coco_pool"):
    raise SystemExit("mode: svo_pool | coco_pool")
pairs = json.load(open(COCO / ("svo_pool_pairs.json" if MODE == "svo_pool" else "coco_pool_pairs.json"), encoding="utf-8"))
jobs = []
for d in ("t2i", "i2t"):
    for q, c in pairs[d]:
        img, cap = (int(c), str(q)) if d == "t2i" else (int(q), str(c))
        jobs.append((d, q, c, img, cap))
header = ["direction", "query", "candidate", "image_id", "caption", "p_yes", "mass_yes", "mass_no", "top_token"]
out_csv = OUT / ("Qwen25p_pool_svo.csv" if MODE == "svo_pool" else "Qwen25p_pool_coco.csv")

done = 0
if out_csv.exists():
    with open(out_csv, encoding="utf-8", newline="") as f:
        done = max(0, sum(1 for _ in csv.reader(f)) - 1)
print(MODE, len(jobs), "jobs,", done, "already done", flush=True)
f = open(out_csv, "a", encoding="utf-8", newline="")
w = csv.writer(f)
if done == 0:
    w.writerow(header)
for i in tqdm(range(done, len(jobs)), desc=f"Qwen2.5 p_yes {MODE}", mininterval=10):
    d, q, c, img, cap = jobs[i]
    p_yes, m_yes, m_no, top = yes_probability(img, cap)
    w.writerow([d, q, c, img, cap, "" if p_yes is None else f"{p_yes:.6f}", "" if m_yes is None else f"{m_yes:.6f}",
                "" if m_no is None else f"{m_no:.6f}", top])
    if i % 50 == 0:
        f.flush()
        torch.cuda.empty_cache()
f.close()
print(MODE, "DONE", len(jobs), flush=True)
