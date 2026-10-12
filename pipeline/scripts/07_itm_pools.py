"""ITM decisions on the fixed judged pool against the human labels.

BLIP-2 and FLAVA: the matching heads of svo_eval.itm_heads (the implementation that also scores the benchmark in
scripts/03b_itm_benchmark.py), run over every judged pair. Qwen2.5 and Qwen3: the yes/no predictions recorded on
the VMs for the pairs those systems retrieved, mapped onto the pool. Per-pair head probabilities are saved.
Usage: python scripts/07_itm_pools.py [BLIP2 FLAVA Qwen25 Qwen3]
"""
import json
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from svo_eval.paths import RESULTS, DATA, DIRECTIONS
from svo_eval import pools, metrics as M
from svo_eval.itm_heads import MatchingHead, THRESHOLD

fp = pools.build_fixed_pool()
out = {}


def summarize(model, preds):
    """preds: dict[direction] -> list of (entry, pred_yes: bool)"""
    out[model] = {}
    for d, rows in preds.items():
        y = np.array([e["correct"] for e, _ in rows], bool)
        p = np.array([pr for _, pr in rows], bool)
        ok = (y == p)
        selfm = np.array([model in e["retrieved_by"] for e, _ in rows])
        res = {"pool_all": M.summary(ok),
               "pool_self": M.summary(ok[selfm]) if selfm.any() else None,
               "pool_other": M.summary(ok[~selfm]) if (~selfm).any() else None,
               "positives_recall": M.summary(p[y]) if y.any() else None,
               "negatives_recall": M.summary(~p[~y]) if (~y).any() else None,
               "share_predicted_yes": M.summary(p), "n_pairs": int(len(rows)), "by_error_type": {}}
        for t in ["subject", "verb", "object"]:
            mask = np.array([(not e["correct"]) and e["error_type"] == t for e, _ in rows])
            res["by_error_type"][t] = M.summary(~p[mask]) if mask.any() else None
        out[model][d] = res
        print(model, d, "pool_all %.1f self %s other %s n=%d" % (
            res["pool_all"]["mean"], round(res["pool_self"]["mean"], 1) if res["pool_self"] else None,
            round(res["pool_other"]["mean"], 1) if res["pool_other"] else None, len(rows)))


which = sys.argv[1:] or ["BLIP2", "FLAVA", "Qwen25", "Qwen3"]
for model in [m for m in ["BLIP", "BLIP2", "FLAVA"] if m in which]:
    head = MatchingHead(model)
    preds, records = {}, []
    for d in DIRECTIONS:
        entries, pairs = [], []
        for q, cands in fp[d].items():
            for cand, e in cands.items():
                img, cap = (cand, q) if d == "t2i" else (q, cand)
                entries.append((q, cand, e))
                pairs.append((int(img), str(cap)))
        probs = head.match_probs(pairs, progress=lambda it, total: tqdm(it, total=total, desc=f"{model} {d}"))
        rows = []
        for (q, cand, e), (img, cap), p in zip(entries, pairs, probs):
            records.append({"direction": d, "query": q, "candidate": cand, "image_id": img, "caption": cap,
                            "human_correct": e["correct"], "p_match": p})
            if p is not None:
                rows.append((e, p > THRESHOLD))
        preds[d] = rows
    pd.DataFrame(records).to_csv(RESULTS / f"itm_pool_preds_{model}.csv", index=False)
    summarize(model, preds)
    del head
    import torch
    torch.cuda.empty_cache()

# Qwen2.5 (zero-shot prompt) and Qwen3 (yes/no construction): the full-pool prediction files, if present, cover every
# judged pair (self- and other-mined); otherwise the June VM files, which answered only the model's own top-10 lists.
FULL = {"Qwen25": (RESULTS / "Qwen25_itm_svo_pool.csv", "model_pred"), "Qwen3": (RESULTS / "itm_pool_preds_Qwen3_full.csv", "pred_yes")}
OWN = {"Qwen25": ("image_matching_results.csv", "text_matching_results.csv"),
       "Qwen3": ("qwen3_embed_image_matching_results.csv", "qwen3_embed_text_matching_results.csv")}
for model in [m for m in ["Qwen25", "Qwen3"] if m in which]:
    preds = {}
    path, col = FULL[model]
    if path.exists():
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for d in DIRECTIONS:
            rows, unanswered = [], 0
            for r in df[df["direction"] == d].itertuples(index=False):
                q, cand = (str(r.query), int(r.candidate)) if d == "t2i" else (int(r.query), str(r.candidate))
                e = fp[d].get(q, {}).get(cand)
                if e is None:
                    continue
                v = str(getattr(r, col)).strip().lower()
                pred = (v in ("1", "true")) if col == "pred_yes" and v != "" else ({"yes": True, "no": False}.get(v) if col == "model_pred" else None)
                if pred is None:
                    unanswered += 1
                    continue
                rows.append((e, pred))
            preds[d] = rows
            print(model, d, "full-pool predictions:", len(rows), "pairs,", unanswered, "unanswered")
    else:
        for d, f in zip(DIRECTIONS, OWN[model]):
            df = pd.read_csv(DATA / "qwen_itm" / f)
            rows = []
            for r in df.itertuples(index=False):
                q, cand = (r.caption, int(r.image_id)) if d == "t2i" else (int(r.image_id), r.caption)
                e = fp[d].get(q, {}).get(cand)
                if e is not None and isinstance(r.model_pred, str):
                    rows.append((e, r.model_pred.strip().lower() == "yes"))
            preds[d] = rows
    summarize(model, preds)

path = RESULTS / "itm_pools.json"
existing = json.load(open(path)) if path.exists() else {}
existing.update(out)
json.dump(existing, open(path, "w"), indent=1)
print("saved", path)
