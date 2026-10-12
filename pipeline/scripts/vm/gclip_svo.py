"""Grounding CLIP scores (Vongala, Srivastava and Kosecka, CVPRW 2025, "Compositional Image-Text Matching and
Retrieval by Grounding Entities") for SVO-Probes pairs, re-implemented from the official repository
(github.com/madhukarreddyvongala/GroundingCLIP, ITM/gclip.py and GroundingDINO/run_gdino_gclip.py) with the
Hugging Face versions of the same two models: Grounding DINO Swin-T OGC (IDEA-Research/grounding-dino-tiny) and
CLIP ViT-L/14 (openai/clip-vit-large-patch14).

Per (caption, image) pair: the subject and the object of the caption's triplet are grounded with the prompt
"{subject}. {object}." (box threshold 0.35, text threshold 0.25). The image blacked out outside the subject box is the
subject image, likewise for the object; the relation image overlays both regions; a missing region falls back to
the whole image (as in the original). Each sub-image embedding, weighted by its CLIP similarity with its phrase
(subject word, object word, verb), is added to the whole-image embedding, and the score is the cosine similarity of
the sum with the caption embedding (unnormalised CLIP projections, as in the original).
Differences from the original: boxes are computed for every (image, caption) pair (the original grounds only the
positive image of each triplet once); when several boxes carry the same phrase the highest-scoring one is kept.
Usage: python gclip_svo.py --inputs gclip_inputs.json --images /srv/gpu-server/qwen/images --out out --part pool
       (--part bench for the benchmark pairs; --limit N for a smoke test; resumes from an existing output file)"""
import argparse
import csv
import json
import os
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, GroundingDinoForObjectDetection, CLIPModel, CLIPProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--part", choices=["pool", "bench"], required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-tiny")
ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
args = ap.parse_args()
BOX_T, TEXT_T = 0.35, 0.25
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, f"gclip_{args.part}.csv")

data = json.load(open(args.inputs, encoding="utf-8"))[args.part]
if args.limit:
    data = data[: args.limit]
done = set()
if os.path.exists(out_path):
    with open(out_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r.get("direction", ""), r.get("query", ""), r.get("candidate", ""), r["caption"], r["image_id"]))
print(f"{len(data)} pairs, {len(done)} already scored", flush=True)

gd_proc = AutoProcessor.from_pretrained(args.gdino)
gd = GroundingDinoForObjectDetection.from_pretrained(args.gdino).to(device).eval()
cl_proc = CLIPProcessor.from_pretrained(args.clip)
cl = CLIPModel.from_pretrained(args.clip).to(device).eval()
text_cache, image_cache = {}, {}


@torch.no_grad()
def text_embed(s):
    if s not in text_cache:
        t = cl_proc(text=[s], return_tensors="pt", padding=True, truncation=True).to(device)
        text_cache[s] = cl.get_text_features(**t)[0].float()
    return text_cache[s]


@torch.no_grad()
def image_embeds(images):
    px = cl_proc(images=images, return_tensors="pt")["pixel_values"].to(device)
    return cl.get_image_features(pixel_values=px).float()


@torch.no_grad()
def ground(image, subject, obj):
    """{phrase: (x0, y0, x1, y1)} for the phrases that Grounding DINO returns, best box per phrase."""
    prompt = f"{subject}. {obj}."
    inp = gd_proc(images=image, text=prompt, return_tensors="pt").to(device)
    out = gd(**inp)
    res = gd_proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=BOX_T, text_threshold=TEXT_T,
                                                         target_sizes=[image.size[::-1]])[0]
    labels = res.get("text_labels", res.get("labels"))
    best = {}
    for box, score, phrase in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
        phrase = str(phrase).strip()
        if phrase and (phrase not in best or score > best[phrase][0]):
            best[phrase] = (score, box)
    return {p: b for p, (s, b) in best.items()}


def black_outside(image, box):
    out = Image.new("RGB", image.size, "black")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])], fill=255)
    out.paste(image, mask=mask)
    return out


def overlay(a, b):
    base = Image.new("RGB", a.size, (0, 0, 0))
    for im in (a, b):
        m = im.convert("L").point(lambda x: 255 if x > 0 else 0, mode="1")
        base.paste(im, (0, 0), mask=m)
    return base


def cos(a, b):
    return float((a / a.norm()) @ (b / b.norm()))


def whole_embed(image_id, image):
    if image_id not in image_cache:
        image_cache[image_id] = image_embeds([image])[0]
        if len(image_cache) > 4000:
            image_cache.clear()
    return image_cache[image_id]


def score_pair(image_id, caption, subject, verb, obj):
    image = Image.open(os.path.join(args.images, f"{image_id}.jpg")).convert("RGB")
    boxes = ground(image, subject, obj)
    sub_img = black_outside(image, boxes[subject]) if subject in boxes else image
    obj_img = black_outside(image, boxes[obj]) if obj in boxes else image
    rel_img = overlay(obj_img, sub_img) if (subject in boxes and obj in boxes) else image
    e_whole = whole_embed(image_id, image)
    e_sub, e_obj, e_rel = image_embeds([sub_img, obj_img, rel_img])
    total = e_whole.clone()
    for phrase, e in ((subject, e_sub), (obj, e_obj), (verb, e_rel)):
        if phrase:
            total = total + cos(text_embed(phrase), e) * e
    return cos(total, text_embed(caption)), subject in boxes, obj in boxes


fields = (["direction", "query", "candidate"] if args.part == "pool" else []) + ["caption", "image_id", "score", "subject_found", "object_found"]
new_file = not os.path.exists(out_path)
t0, n = time.time(), 0
with open(out_path, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()
    for i, p in enumerate(data):
        key = (p.get("direction", ""), p.get("query", ""), p.get("candidate", ""), p["caption"], str(p["image_id"]))
        if key in done:
            continue
        try:
            s, fs, fo = score_pair(p["image_id"], p["caption"], p["subject"], p["verb"], p["object"])
            row = {"caption": p["caption"], "image_id": p["image_id"], "score": f"{s:.6f}", "subject_found": int(fs), "object_found": int(fo)}
        except Exception as e:  # unreadable image etc.: no score
            row = {"caption": p["caption"], "image_id": p["image_id"], "score": "", "subject_found": "", "object_found": ""}
            print(f"pair {i}: {type(e).__name__}: {e}", flush=True)
        if args.part == "pool":
            row.update({"direction": p["direction"], "query": p["query"], "candidate": p["candidate"]})
        w.writerow(row)
        n += 1
        if n % 200 == 0:
            f.flush()
            el = time.time() - t0
            print(f"{n} scored, {el / n:.3f} s/pair, ETA {(len(data) - i - 1) * el / n / 60:.1f} min", flush=True)
print(f"done: {n} new rows in {out_path} ({(time.time() - t0) / 60:.1f} min)", flush=True)
