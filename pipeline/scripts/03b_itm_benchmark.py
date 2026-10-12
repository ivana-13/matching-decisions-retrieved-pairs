"""ITM accuracy on the benchmark pairs (each triplet's positive and its negative pair), one implementation per system.

BLIP-2 and FLAVA: the matching heads of svo_eval.itm_heads, run here over every unique (caption, image) pair.
Qwen2.5 (zero-shot Instruct, yes/no prompt) and Qwen3-VL-Embedding (Yes/No similarity): VM prediction files,
excluding pairs the run left unanswered. Per-pair probabilities of the heads are saved next to the summary.
Usage: python scripts/03b_itm_benchmark.py [BLIP2 FLAVA Qwen25 Qwen3]
"""
import json
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from svo_eval.paths import RESULTS
from svo_eval.collection import load_collection
from svo_eval.itm_heads import MatchingHead, THRESHOLD
from svo_eval.qwen_csv import load_qwen25_itm, load_qwen3_itm
from svo_eval import metrics as M

TYPES = ["subject", "verb", "object"]


def summarize(correct, types, is_pos, n_unanswered, n_rows):
    correct, types, is_pos = np.asarray(correct, bool), np.asarray(types, bool), np.asarray(is_pos, bool)
    res = {"overall": M.summary(correct)}
    for k, t in enumerate(TYPES):
        res[t] = M.summary(correct[types[:, k]])
    res["tpr"] = M.summary(correct[is_pos])
    res["tnr"] = M.summary(correct[~is_pos])
    res["share_predicted_match"] = M.summary(np.where(is_pos, correct, ~correct))
    res["n_pairs"] = int(len(correct))
    res["n_unanswered"] = int(n_unanswered)
    res["n_source_rows"] = int(n_rows)
    return res


which = sys.argv[1:] or ["BLIP2", "FLAVA", "Qwen25", "Qwen3"]
out = {}
coll = load_collection()
df = coll.df
pairs = []                                   # (caption, image, is_positive, type vector)
for r in df.itertuples(index=False):
    tv = (bool(r.subj_neg), bool(r.verb_neg), bool(r.obj_neg))
    pairs.append((r.sentence, int(r.pos_image_id), True, tv))
    pairs.append((r.sentence, int(r.neg_image_id), False, tv))
uniq = sorted({(c, i) for c, i, _, _ in pairs}, key=lambda x: (x[1], x[0]))   # image-sorted: consecutive pairs share images
print(f"{len(pairs)} benchmark pairs, {len(uniq)} unique (caption, image) pairs")

for model in [m for m in ["BLIP", "BLIP2", "FLAVA"] if m in which]:
    head = MatchingHead(model)
    probs = head.match_probs([(i, c) for c, i in uniq], progress=lambda it, total: tqdm(it, total=total, desc=model))
    pd.DataFrame({"caption": [c for c, _ in uniq], "image_id": [i for _, i in uniq], "p_match": probs}).to_csv(
        RESULTS / f"itm_benchmark_preds_{model}.csv", index=False)
    lookup = dict(zip(uniq, probs))
    correct, types, is_pos, skipped = [], [], [], 0
    for c, i, pos, tv in pairs:
        p = lookup[(c, i)]
        if p is None:
            skipped += 1
            continue
        correct.append((p > THRESHOLD) == pos)
        types.append(tv)
        is_pos.append(pos)
    out[model] = summarize(correct, types, is_pos, skipped, len(pairs))
    print(model, "ITM %.2f  n=%d  skipped=%d" % (out[model]["overall"]["mean"], out[model]["n_pairs"], skipped))
    del head
    import torch
    torch.cuda.empty_cache()

for model, loader in [("Qwen25", load_qwen25_itm), ("Qwen3", load_qwen3_itm)]:
    if model not in which:
        continue
    r = loader()
    out[model] = summarize(r["correct"], r["types"], r["is_positive"], r["n_unanswered"], 2 * r["n_rows"])
    print(model, "ITM %.2f  n=%d  unanswered=%d" % (out[model]["overall"]["mean"], out[model]["n_pairs"], r["n_unanswered"]))

path = RESULTS / "itm_benchmark.json"
existing = json.load(open(path)) if path.exists() else {}
existing.update(out)
json.dump(existing, open(path, "w"), indent=1)
print("saved", path)
