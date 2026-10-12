"""Zero-shot Qwen2.5-VL-7B-Instruct (no adapter) prompted decisions on the second collection, on the GPU server, the same way as
on SVO-Probes (/srv/gpu-server/qwen/qwen_eval.py, qwen_eval_pairwaise.py, qwen_eval_new_set.py): unsloth 4-bit load,
greedy decoding of at most 10 new tokens, images passed at their native size through the chat template.

Modes (one per run; outputs under inputs/out, appended as the run proceeds and resumed from):
  sugarcrepe_itm      every SugarCrepe pair as (image, caption) and (image, negative caption) with the SVO-Probes
                      yes/no prompt                                -> Qwen25_itm_sugarcrepe.csv
  sugarcrepe_pairwise one image, the caption and the negative caption in random order (seeded per pair), asked which
                      caption matches better ('A' or 'B'); the SVO-Probes prompt showed two images and one caption,
                      here the negative is a caption, so the roles are swapped   -> Qwen25_pairwise_sugarcrepe.csv
  pool_itm            every (query, candidate) pair of coco_pool_pairs.json with the yes/no prompt
                                                                   -> Qwen25_itm_pool.csv
Answer parsing: 'yes'/'no' as in the SVO-Probes scripts; for A/B the first letter of the answer (the SVO-Probes
script tested `'a' in answer`, which misreads answers such as 'b matches'), anything else is recorded as unanswered.
Run from /srv/gpu-server/qwen with the qwen conda environment:
    /opt/anacondas/anaconda3/envs/qwen/bin/python inputs/qwen25_coco_prompted.py <mode>"""
import csv
import json
import random
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoProcessor

MODE = sys.argv[1]
BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=512, dtype=None, device_map="auto", load_in_4bit=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_disable()
model.eval()

ITM_PROMPT = ("You are an image caption expert. Does this image match the caption: '{caption}'? Answer yes only if "
              "everything from the caption - subject, verb and object - matches the image, otherwise answer no.")
PAIRWISE_PROMPT = ("You are an image caption expert.\n"
                   "An image is given above. Two captions are given below: the first caption is CAPTION_A, "
                   "the second caption is CAPTION_B.\n\n"
                   "CAPTION_A: '{a}'\nCAPTION_B: '{b}'\n\n"
                   "Question: Which caption matches the image better? "
                   "Answer with just 'A' if CAPTION_A matches better, or just 'B' if CAPTION_B matches better.")


def image_path(image_id):
    return str(BASE / "val2017" / f"{int(image_id):012d}.jpg")


def generate(messages):
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, add_generation_prompt=True,
                                           return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].lower().strip()


def ask_itm(image_id, caption):
    try:
        answer = generate([{"role": "user", "content": [{"type": "image", "image": image_path(image_id)},
                                                        {"type": "text", "text": ITM_PROMPT.format(caption=caption)}]}])
    except Exception as e:                       # unreadable image, CUDA error: recorded as unanswered
        print("error", image_id, repr(e)[:200], file=sys.stderr)
        return None, ""
    if "yes" in answer:
        return "yes", answer
    if "no" in answer:
        return "no", answer
    return None, answer


def ask_pairwise(image_id, caption_a, caption_b):
    try:
        answer = generate([{"role": "user", "content": [{"type": "image", "image": image_path(image_id)},
                                                        {"type": "text", "text": PAIRWISE_PROMPT.format(a=caption_a, b=caption_b)}]}])
    except Exception as e:
        print("error", image_id, repr(e)[:200], file=sys.stderr)
        return None, ""
    letters = [ch for ch in answer if ch.isalpha()]
    first = letters[0] if letters else ""
    return ({"a": "A", "b": "B"}.get(first), answer)


inp = json.load(open(BASE / "coco_inputs.json", encoding="utf-8"))
if MODE == "sugarcrepe_itm":
    jobs = []
    for k, p in enumerate(inp["sugarcrepe"]):
        jobs.append((k, p["subset"], p["image_id"], p["caption"], True))
        jobs.append((k, p["subset"], p["image_id"], p["negative_caption"], False))
    header = ["pair_index", "subset", "image_id", "text", "is_positive", "model_pred", "raw_answer"]
    out_csv = OUT / "Qwen25_itm_sugarcrepe.csv"
elif MODE == "sugarcrepe_pairwise":
    jobs = [(k, p["subset"], p["image_id"], p["caption"], p["negative_caption"]) for k, p in enumerate(inp["sugarcrepe"])]
    header = ["pair_index", "subset", "image_id", "caption", "negative_caption", "positive_is", "model_pred", "correct", "raw_answer"]
    out_csv = OUT / "Qwen25_pairwise_sugarcrepe.csv"
elif MODE in ("pool_itm", "svo_pool_itm"):
    # pool_itm: the COCO pool; svo_pool_itm: the whole SVO-Probes judged pool (svo_pool_pairs.json, images in
    # /srv/gpu-server/qwen/images/{id}.jpg), so that the zero-shot row of the SVO-Probes pool table covers the pairs
    # retrieved by other systems as well (the June run answered only the model's own top-10 lists)
    pairs = json.load(open(BASE / ("coco_pool_pairs.json" if MODE == "pool_itm" else "svo_pool_pairs.json"), encoding="utf-8"))
    jobs = []
    for d in ("t2i", "i2t"):
        for q, c in pairs[d]:
            img, cap = (int(c), str(q)) if d == "t2i" else (int(q), str(c))
            jobs.append((d, q, c, img, cap))
    header = ["direction", "query", "candidate", "image_id", "caption", "model_pred", "raw_answer"]
    out_csv = OUT / ("Qwen25_itm_pool.csv" if MODE == "pool_itm" else "Qwen25_itm_svo_pool.csv")
    if MODE == "svo_pool_itm":
        image_path = lambda image_id: str(BASE.parent / "images" / f"{int(image_id)}.jpg")   # noqa: E731
else:
    raise SystemExit("mode: sugarcrepe_itm | sugarcrepe_pairwise | pool_itm | svo_pool_itm")

done = 0
if out_csv.exists():
    with open(out_csv, encoding="utf-8", newline="") as f:
        done = max(0, sum(1 for _ in csv.reader(f)) - 1)
print(MODE, len(jobs), "jobs,", done, "already done", flush=True)
f = open(out_csv, "a", encoding="utf-8", newline="")
w = csv.writer(f)
if done == 0:
    w.writerow(header)

for i in tqdm(range(done, len(jobs)), desc=f"Qwen2.5 {MODE}", mininterval=10):
    j = jobs[i]
    if MODE == "sugarcrepe_itm":
        pred, raw = ask_itm(j[2], j[3])
        w.writerow([j[0], j[1], j[2], j[3], j[4], pred if pred else "", raw])
    elif MODE == "sugarcrepe_pairwise":
        k, subset, image_id, caption, negative = j
        rng = random.Random(k)
        positive_is = "A" if rng.random() < 0.5 else "B"
        a, b = (caption, negative) if positive_is == "A" else (negative, caption)
        pred, raw = ask_pairwise(image_id, a, b)
        correct = "" if pred is None else str(pred == positive_is)
        w.writerow([k, subset, image_id, caption, negative, positive_is, pred if pred else "", correct, raw])
    else:
        d, q, c, img, cap = j
        pred, raw = ask_itm(img, cap)
        w.writerow([d, q, c, img, cap, pred if pred else "", raw])
    if i % 50 == 0:
        f.flush()
        torch.cuda.empty_cache()
f.close()
print(MODE, "DONE", len(jobs), flush=True)
