"""Analyses of the paper on binary matching decisions (stored decisions only, no model is run here).

Decisions: the matching heads of BLIP-2 and FLAVA, the zero-shot Qwen2.5-VL prompt (answer only, and with the
yes-probability from scripts/vm/qwen25_yes_probability.py when results/Qwen25p_pool_*.csv exist), the Qwen3-VL-Embed
yes/no construction, and Qwen3-VL-Reranker-2B (scripts/vm/qwen3_reranker.py, when results/Qwen3R_pool_*.csv exist).
On the judged pools of SVO-Probes (human labels) and COCO (o3 labels):
  separation  AUC of the decision's score over the pool pairs, TPR / TNR of the decision itself, and the AUC inside
              each top-10 list (lists with both labels), next to the AUC of the first stage's own order;
  errors      which error types the decision accepts (SVO-Probes, the expert's error type of each incorrect pair);
  rerank      judged Success@1 of each embedding model's top-10 list before and after the list is re-ordered by the
              decision ("score": by the decision's score; "binary": accepted items first; ties keep the original
              order), the same with the benchmark's strict labels, and with only the top k re-ordered.
A list is judged by its own labels on SVO-Probes (every list was labelled separately) and by the pair's label on COCO.
Output: results/short_matching.json"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from svo_eval.paths import RESULTS, MODELS, DUAL_ENCODERS, DIRECTIONS
from svo_eval import metrics as M
from svo_eval.pools import load_pool, is_correct, error_type
from svo_eval.collection import load_collection
from svo_eval.coco import load_coco
from svo_eval.embeddings import Scorer
from svo_eval.qwen_csv import _align_to_collection

FILES = {
    "svo": {"BLIP": "itm_pool_preds_BLIP.csv", "BLIP2": "itm_pool_preds_BLIP2.csv", "FLAVA": "itm_pool_preds_FLAVA.csv",
            "Qwen25": "Qwen25_itm_svo_pool.csv", "Qwen3": "itm_pool_preds_Qwen3_full.csv", "Qwen25p": "Qwen25p_pool_svo.csv",
            "Qwen3R": "Qwen3R_pool_svo.csv", "GCLIP": "GCLIP_pool_svo.csv"},
    "coco": {"BLIP2": "coco_itm_preds_BLIP2_pool.csv", "FLAVA": "coco_itm_preds_FLAVA_pool.csv", "Qwen25": "Qwen25_itm_pool.csv",
             "Qwen3": "coco_itm_preds_Qwen3_pool.csv", "Qwen25p": "Qwen25p_pool_coco.csv", "Qwen3R": "Qwen3R_pool_coco.csv"},
}
ALL_DECISIONS = ["BLIP", "BLIP2", "FLAVA", "Qwen25", "Qwen25p", "Qwen3", "Qwen3R", "GCLIP"]
SCORE_ONLY = {"GCLIP"}                        # Grounding CLIP (scripts/vm/gclip_svo.py): a similarity without a threshold
# a decision is analysed on a benchmark when its stored pool decisions exist there (the BLIP head and GCLIP: SVO-Probes only)
DECISIONS_BY_BENCH = {b: [k for k in ALL_DECISIONS if k in FILES[b] and (RESULTS / FILES[b][k]).exists()] for b in FILES}
DECISIONS = DECISIONS_BY_BENCH["svo"]
OWN_LIST = {"BLIP": None, "BLIP2": "BLIP2", "FLAVA": "FLAVA", "Qwen25": "Qwen25", "Qwen25p": "Qwen25", "Qwen3": "Qwen3", "Qwen3R": "Qwen3", "GCLIP": "CLIP"}
DEPTHS = (2, 3, 5)
N_BOOT = 1000
N_FOLDS = 10                                  # cross-validation over queries (threshold tuning, interpolation weight)
COLS = {"svo": load_collection(), "coco": load_coco() if DECISIONS_BY_BENCH["coco"] else None}   # no COCO files: no COCO
# the judged lists of the paper: the fine-tuned Qwen2.5-VL embedder is left out (its adaptation has no room in the
# paper); the SVO-Probes pairs are the union of these lists, each pair labelled by the first list in this order
LISTS = [m for m in MODELS if m != "Qwen25"]
# the LLM assessor's labels are an extra of the stored lists; the analyses that use them run only where they exist
HAS_O3 = any(j["o3"] is not None for m in LISTS for d in DIRECTIONS for items in load_pool(m, d).values() for j in items)


def load_decision(bench, dec):
    """{(direction, query, candidate): (score, accepted)}; pairs without an answer are left out."""
    df = pd.read_csv(RESULTS / FILES[bench][dec], dtype=str, keep_default_na=False)
    out = {}
    for r in df.to_dict("records"):
        key = (r["direction"], r["query"], r["candidate"])
        if dec in ("BLIP", "BLIP2", "FLAVA"):
            if r["p_match"] == "":
                continue
            p = float(r["p_match"])
            out[key] = (p, p > 0.5)
        elif dec == "Qwen3":
            if r["sim_yes"] == "" or r["sim_no"] == "":
                continue
            s = float(r["sim_yes"]) - float(r["sim_no"])
            out[key] = (s, s > 0)
        elif dec == "Qwen25":
            v = r["model_pred"].strip().lower()
            if v not in ("yes", "no"):
                continue
            out[key] = (1.0 if v == "yes" else 0.0, v == "yes")
        elif dec == "GCLIP":                                      # score only: no default decision
            if r["score"] == "":
                continue
            out[key] = (float(r["score"]), None)
        else:                                                   # Qwen25p: p_yes; Qwen3R: sigmoid score
            col = "p_yes" if dec == "Qwen25p" else "score"
            if r[col] == "":
                continue
            p = float(r[col])
            out[key] = (p, p > 0.5)
    return out


def load_pairs(bench, coco_judge="o3"):
    """pairs[d][q][c] = {"label": bool or None, "etype": str or None, "ranks": {model: rank}} with string keys."""
    pairs = {}
    if bench == "svo":
        for d in DIRECTIONS:
            pairs[d] = {}
            for m in LISTS:
                for q, items in load_pool(m, d).items():
                    bucket = pairs[d].setdefault(str(q), {})
                    for j in items:
                        ok = bool(is_correct(j))
                        e = bucket.setdefault(str(j["cand"]), {"label": ok, "etype": error_type(j) if not ok else None, "ranks": {}, "labels": []})
                        e["ranks"][m] = j["rank"]
                        e["labels"].append(ok)                    # one label per list that holds the pair
    else:
        cands = json.load(open(RESULTS / "coco_pool_candidates.json", encoding="utf-8"))
        labels = json.load(open(RESULTS / f"coco_pool_labels_{coco_judge}.json", encoding="utf-8"))
        for d in DIRECTIONS:
            pairs[d] = {}
            for q, cs in cands[d].items():
                if coco_judge != "o3" and q not in labels.get(d, {}):
                    continue
                lab = labels.get(d, {}).get(q, {})
                pairs[d][q] = {c: {"label": (lab[c] == "correct") if lab.get(c) else None,
                                   "etype": (lab[c].replace(" incorrect", "") if lab.get(c) and lab[c] != "correct" else None),
                                   "ranks": v["retrieved_by"]} for c, v in cs.items()}
    return pairs


def o3_pairs_svo():
    """The SVO-Probes pairs of the paper's lists with the assessor o3's label and error type (first list in LISTS order
    that has an o3 label for the pair); pairs without an o3 label get None."""
    pairs = {}
    for d in DIRECTIONS:
        pairs[d] = {}
        for m in LISTS:
            for q, items in load_pool(m, d).items():
                bucket = pairs[d].setdefault(str(q), {})
                for j in items:
                    c = str(j["cand"])
                    e = bucket.setdefault(c, {"label": None, "etype": None, "ranks": {}})
                    e["ranks"][m] = j["rank"]
                    if e["label"] is None and j["o3"] is not None:
                        e["label"] = j["o3"] == "correct"
                        e["etype"] = j["o3"].replace(" incorrect", "") if j["o3"] != "correct" else None
    return pairs


def gold(bench, d, q):
    """The benchmark's own relevant items of a pool query, as strings."""
    col = COLS[bench]
    return {str(x) for x in col.gold(d, q if d == "t2i" else int(q))}


def auc(y, s):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def separation(pairs, dec, d):
    rows, missing = [], 0                      # (query, label, score, accepted)
    for q, cs in pairs[d].items():
        for c, v in cs.items():
            if v["label"] is None:
                continue
            x = dec.get((d, q, c))
            if x is None:
                missing += 1
                continue
            rows.append((q, v["label"], x[0], x[1]))
    qs = sorted({r[0] for r in rows})
    qi = {q: i for i, q in enumerate(qs)}
    g = np.array([qi[r[0]] for r in rows]); y = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows])
    has_decision = all(r[3] is not None for r in rows)
    a = np.array([bool(r[3]) for r in rows]) if has_decision else None
    rng = np.random.default_rng(0)
    idx_by_q = [np.where(g == i)[0] for i in range(len(qs))]
    boots = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx_by_q[i] for i in rng.integers(0, len(qs), len(qs))])
        boots.append(auc(y[take], s[take]))
    nan = float("nan")
    is_prob = has_decision and s.min() >= 0 and s.max() <= 1 and len(np.unique(s)) > 2   # probabilities: calibration error
    return {"auc": auc(y, s), "lo": float(np.nanquantile(boots, 0.025)), "hi": float(np.nanquantile(boots, 0.975)),
            "ece": expected_calibration_error(s, y) if is_prob else nan,
            "tpr": 100 * float(a[y].mean()) if has_decision else nan, "tnr": 100 * float((~a[~y]).mean()) if has_decision else nan,
            "share_yes": 100 * float(a.mean()) if has_decision else nan,
            "acc": 100 * float((a == y).mean()) if has_decision else nan, "share_incorrect": 100 * float((~y).mean()),
            "n": int(len(rows)), "n_missing": int(missing)}


def error_types(pairs, dec):
    out = {}
    for scope in DIRECTIONS + ["both"]:
        ds = DIRECTIONS if scope == "both" else [scope]
        pos, neg = [], {"subject": [], "verb": [], "object": []}
        for d in ds:
            for q, cs in pairs[d].items():
                for c, v in cs.items():
                    x = dec.get((d, q, c))
                    if x is None or v["label"] is None:
                        continue
                    if v["label"]:
                        pos.append(x)
                    elif v["etype"] in neg:
                        neg[v["etype"]].append(x)
        out[scope] = {}
        for t, xs in neg.items():
            y = np.array([True] * len(pos) + [False] * len(xs)); s = np.array([x[0] for x in pos] + [x[0] for x in xs])
            accept = 100 * float(np.mean([bool(x[1]) for x in xs])) if xs and all(x[1] is not None for x in xs) else float("nan")
            out[scope][t] = {"accept": accept, "n": len(xs), "auc": auc(y, s)}
        out[scope]["n_correct"] = len(pos)
    return out


def own_labels(bench, pairs, m, d, label_fn=None):
    """{query: [(rank, candidate, correct)]} for the top-10 list of model m, sorted by rank.
    label_fn(judgment) replaces the paper's label rule (majority vote, else the expert) on SVO-Probes; a query whose
    list has an item without a label under that rule (None) is left out."""
    out = {}
    if bench == "svo":
        for q, items in load_pool(m, d).items():
            labs = [is_correct(j) if label_fn is None else label_fn(j) for j in items]
            if any(x is None for x in labs):
                continue
            out[str(q)] = sorted((j["rank"], str(j["cand"]), bool(x)) for j, x in zip(items, labs))
    else:
        for q, cs in pairs[d].items():
            lst = sorted((v["ranks"][m], c, bool(v["label"])) for c, v in cs.items() if m in v["ranks"])
            if lst:
                out[q] = lst
    return out


def reciprocal_rank(labels):
    for i, ok in enumerate(labels):
        if ok:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(labels):
    """nDCG@10 with binary gains; the ideal order puts the judged-correct items first."""
    g = np.array([float(x) for x in labels[:10]])
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    ideal = float((np.sort(g)[::-1] * disc).sum())
    return float((g * disc).sum() / ideal) if ideal > 0 else 0.0


def expected_calibration_error(p, y, n_bins=10):
    """ECE of probabilities p against binary labels y, equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def rerank(bench, pairs, decs, with_extras=True, models=MODELS, directions=DIRECTIONS, label_fn=None, keep=None):
    """keep: {direction: set of queries} restricts every list to those queries."""
    res, per_query, strict_q, depth_q, within = {}, {}, {}, {}, {k: [] for k in list(decs) + ["first_stage"]}
    mrr_q = {}                                     # (name, mode, m, d) -> {query: change in judged reciprocal rank}
    ndcg_q = {}                                    # the same for judged nDCG@10 (binary gains)
    for m in models:
        res[m] = {}
        for d in directions:
            lists = own_labels(bench, pairs, m, d, label_fn)
            if keep is not None:
                lists = {q: v for q, v in lists.items() if q in keep[d]}
            qs = sorted(lists)
            orig = np.array([lists[q][0][2] for q in qs], float)
            golds = {q: gold(bench, d, q) for q in qs} if with_extras else {}
            strict0 = np.array([lists[q][0][1] in golds[q] for q in qs], float) if with_extras else None
            cell = {"n": len(qs), "orig": 100 * float(orig.mean()),
                    "mrr_orig": 100 * float(np.mean([reciprocal_rank([x[2] for x in lists[q]]) for q in qs])),
                    "share": 100 * float(np.mean([np.mean([x[2] for x in lists[q]]) for q in qs])),
                    "oracle": 100 * float(np.mean([any(x[2] for x in lists[q]) for q in qs])),
                    "p3_orig": 100 * float(np.mean([np.mean([x[2] for x in lists[q][:3]]) for q in qs]))}
            if with_extras:
                cell["strict_orig"] = 100 * float(strict0.mean())
                for q in qs:
                    y = np.array([x[2] for x in lists[q]])
                    if 0 < y.sum() < len(y):
                        within["first_stage"].append(auc(y, -np.array([x[0] for x in lists[q]], float)))
            for name, dec in decs.items():
                cell[name] = {}
                for mode in ("score", "binary"):
                    after, p3, moved, strict1, rr_after, nd_after = [], [], 0, [], [], []
                    for q in qs:
                        keyed = []
                        for rank, c, ok in lists[q]:
                            x = dec.get((d, q, c))
                            # score-only decisions have no default filter: the binary mode leaves the list unchanged
                            val = -np.inf if x is None else (x[0] if mode == "score" else (0.0 if x[1] is None else float(x[1])))
                            keyed.append((-val, rank, c, ok))
                        keyed.sort()
                        after.append(keyed[0][3]); p3.append(np.mean([k[3] for k in keyed[:3]])); moved += keyed[0][1] != 1
                        rr_after.append(reciprocal_rank([k[3] for k in keyed]) - reciprocal_rank([x[2] for x in lists[q]]))
                        nd_after.append(ndcg([k[3] for k in keyed]) - ndcg([x[2] for x in lists[q]]))
                        if with_extras:
                            strict1.append(keyed[0][2] in golds[q])
                            if mode == "score":
                                y = np.array([x[3] for x in sorted(keyed, key=lambda k: k[1])])
                                vals = np.array([-k[0] for k in sorted(keyed, key=lambda k: k[1])])
                                if 0 < y.sum() < len(y) and np.isfinite(vals).all():
                                    within[name].append(auc(y, vals))
                                for kk in DEPTHS:
                                    head = sorted(keyed, key=lambda k: k[1])[:kk]
                                    top = min(head)                     # the best-scored item among the first kk
                                    depth_q.setdefault((name, kk, m, d), {})[q] = float(top[3]) - float(lists[q][0][2])
                    after = np.array(after, float)
                    diff = after - orig
                    mean, lo, hi = M.bootstrap_ci(diff)
                    cell[name][mode] = {"after": 100 * float(after.mean()), "diff": 100 * mean, "lo": 100 * lo, "hi": 100 * hi,
                                        "p": M.randomisation_p(diff), "p3_after": 100 * float(np.mean(p3)),
                                        "top_changed": 100 * moved / len(qs),
                                        "better": int((diff > 0).sum()), "worse": int((diff < 0).sum())}
                    per_query[(name, mode, m, d)] = dict(zip(qs, diff))
                    mrr_q[(name, mode, m, d)] = dict(zip(qs, rr_after))
                    ndcg_q[(name, mode, m, d)] = dict(zip(qs, nd_after))
                    cell[name][mode]["mrr_diff"] = 100 * float(np.mean(rr_after))
                    cell[name][mode]["ndcg_diff"] = 100 * float(np.mean(nd_after))
                    if with_extras:
                        sdiff = np.array(strict1, float) - strict0
                        cell[name][mode]["strict_after"] = 100 * float(np.mean(strict1))
                        cell[name][mode]["strict_diff"] = 100 * float(sdiff.mean())
                        strict_q[(name, mode, m, d)] = dict(zip(qs, sdiff))
            res[m][d] = cell
    # Holm over the list-and-direction cells of each decision and mode (twelve in the paper's setting)
    for name in decs:
        for mode in ("score", "binary"):
            adj = M.holm({(m, d): res[m][d][name][mode]["p"] for m in models for d in directions})
            for (m, d), p in adj.items():
                res[m][d][name][mode]["p_holm"] = p
    # one test per decision and direction on the per-query gain averaged over the lists (all six, and without the own family)
    mean, ps = {}, {}

    def averaged(store, name, mode, d, ms):
        qs = sorted(set.intersection(*[set(store[(name, mode, m, d)]) for m in ms]))
        v = np.array([np.mean([store[(name, mode, m, d)][q] for m in ms]) for q in qs])
        mu, lo, hi = M.bootstrap_ci(v)
        return {"diff": 100 * mu, "lo": 100 * lo, "hi": 100 * hi, "p": M.randomisation_p(v), "n": len(qs)}

    for name in decs:
        mean[name] = {}
        for d in directions:
            mean[name][d] = {}
            for mode in ("score", "binary"):
                cell = {"all": averaged(per_query, name, mode, d, models),
                        "other": averaged(per_query, name, mode, d, [m for m in models if m != OWN_LIST[name]]),
                        "own": res[OWN_LIST[name]][d][name][mode]["diff"] if OWN_LIST[name] in models else None,
                        "mrr": averaged(mrr_q, name, mode, d, models), "ndcg": averaged(ndcg_q, name, mode, d, models)}
                if with_extras:
                    cell["strict"] = averaged(strict_q, name, mode, d, models)
                    if mode == "score":
                        cell["depth"] = {kk: averaged(depth_q, name, kk, d, models) for kk in DEPTHS}
                mean[name][d][mode] = cell
                ps[(name, d, mode)] = cell["all"]["p"]
    for mode in ("score", "binary"):
        adj = M.holm({k: p for k, p in ps.items() if k[2] == mode})
        for (name, d, _), p in adj.items():
            mean[name][d][mode]["all"]["p_holm"] = p
        if with_extras:
            adj = M.holm({k: mean[k[0]][k[1]][mode]["strict"]["p"] for k in ps if k[2] == mode})
            for (name, d, _), p in adj.items():
                mean[name][d][mode]["strict"]["p_holm"] = p
    # the two matching heads against each other: per-query difference of their re-ordered Success@1, averaged over the lists
    contrast = {}
    for d in directions:
        qs = sorted(set.intersection(*[set(per_query[(k, "score", m, d)]) for k in ("BLIP2", "FLAVA") for m in models]))
        v = np.array([np.mean([per_query[("BLIP2", "score", m, d)][q] - per_query[("FLAVA", "score", m, d)][q] for m in models]) for q in qs])
        mu, lo, hi = M.bootstrap_ci(v)
        contrast[d] = {"diff": 100 * mu, "lo": 100 * lo, "hi": 100 * hi, "p": M.randomisation_p(v), "n": len(qs)}
    mean["BLIP2_minus_FLAVA"] = contrast
    if with_extras:
        mean["within_list_auc"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in within.items() if v}
    return res, mean, per_query


def learning_curve(per_query, decs, models, sizes=(25, 50, 75, 100), n_draws=500):
    """How many judged queries are needed: in random subsets of n queries per direction, the share of draws in which
    the difference between the BLIP-2 and FLAVA heads is significant (paired randomisation test, 10,000 flips,
    p < 0.05), the half-width of the 95% bootstrap interval of the prompt's mean gain, and Spearman's rho between
    the decisions' mean gains in the subset and in the full sample (over the thresholded decisions)."""
    from scipy.stats import spearmanr
    shown = [k for k in decs if k not in SCORE_ONLY and not (k == "Qwen25" and "Qwen25p" in decs)]
    prompt = "Qwen25p" if "Qwen25p" in decs else "Qwen25"
    out = {}
    for d in DIRECTIONS:
        qs = sorted(set.intersection(*[set(per_query[(k, "score", m, d)]) for k in shown for m in models]))
        gains = {k: np.array([np.mean([per_query[(k, "score", m, d)][q] for m in models]) for q in qs]) for k in shown}
        contrast = gains["BLIP2"] - gains["FLAVA"]
        full_order = [gains[k].mean() for k in shown]
        rng = np.random.default_rng(0)
        out[d] = {}
        for n in sizes:
            sig, widths, rhos = 0, [], []
            for _ in range(n_draws):
                idx = rng.choice(len(qs), size=n, replace=n > len(qs))
                sig += M.randomisation_p(contrast[idx], n_perm=10000) < 0.05
                _, lo, hi = M.bootstrap_ci(gains[prompt][idx], n_boot=500)
                widths.append(100 * (hi - lo) / 2)
                rhos.append(spearmanr(full_order, [gains[k][idx].mean() for k in shown])[0])
            out[d][str(n)] = {"contrast_significant_share": 100 * sig / n_draws, "prompt_ci_halfwidth": float(np.mean(widths)),
                              "order_rho_mean": float(np.nanmean(rhos)), "order_rho_ge_0.9_share": 100 * float(np.mean(np.array(rhos) >= 0.9))}
    return out


def holm_cells(table, decs, key=None):
    """Holm over the decision-and-direction cells of a {decision: {direction: {... "p"}}} table (key: a sub-level)."""
    cell = (lambda k, d: table[k][d][key]) if key else (lambda k, d: table[k][d])
    adj = M.holm({(k, d): cell(k, d)["p"] for k in decs for d in DIRECTIONS})
    for (k, d), p in adj.items():
        cell(k, d)["p_holm"] = p


def aggregate(diffs):
    v = np.array(list(diffs.values()), float)
    mu, lo, hi = M.bootstrap_ci(v)
    return {"diff": 100 * mu, "lo": 100 * lo, "hi": 100 * hi, "p": M.randomisation_p(v), "n": len(v)}


def list_arrays(pairs, dec, m, d):
    """Per list of model m: queries, decision scores (nq, 10; missing -> -inf), correctness (nq, 10), in rank order."""
    lists = own_labels("svo", pairs, m, d)
    qs = sorted(lists)
    S = np.full((len(qs), 10), -np.inf)
    C = np.zeros((len(qs), 10), bool)
    for i, q in enumerate(qs):
        for j, (rank, c, ok) in enumerate(lists[q][:10]):
            x = dec.get((d, q, c))
            S[i, j] = x[0] if x is not None else -np.inf
            C[i, j] = ok
    return qs, S, C


def top_after_filter(S, C, taus):
    """Correctness (T, nq) of the first item once the items with score > tau move to the front in their original order."""
    acc = S[None, :, :] > np.asarray(taus)[:, None, None]            # (T, nq, 10)
    first = np.argmax(acc, axis=2)                                   # first accepted item, 0 when none is accepted
    return C[np.arange(C.shape[0])[None, :], first]


def tuned_filter(pairs, decs):
    """The default decision as a filter, with the threshold chosen on the judged pairs of the other queries (N_FOLDS-fold
    cross-validation over queries): "balanced" maximises balanced accuracy on the training pairs, "s1" maximises the
    judged Success@1 of the six filtered training lists. Mean change over the six lists per decision and direction."""
    out = {}
    for name, dec in decs.items():
        out[name] = {}
        for d in DIRECTIONS:
            arrays = {m: list_arrays(pairs, dec, m, d) for m in LISTS}
            qs = sorted(set.intersection(*[set(a[0]) for a in arrays.values()]))
            idx = {m: {q: i for i, q in enumerate(arrays[m][0])} for m in LISTS}
            orig = {q: np.mean([arrays[m][2][idx[m][q], 0] for m in LISTS]) for q in qs}
            rng = np.random.default_rng(0)
            folds = np.array_split(rng.permutation(qs), N_FOLDS)
            pair_s, pair_y, pair_q = [], [], []
            for q, cs in pairs[d].items():
                for c, v in cs.items():
                    x = dec.get((d, q, c))
                    if v["label"] is None or x is None:
                        continue
                    pair_s.append(x[0]); pair_y.append(v["label"]); pair_q.append(q)
            pair_s, pair_y, pair_q = np.array(pair_s), np.array(pair_y, bool), np.array(pair_q)
            u = np.unique(pair_s)
            cands = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
            if len(cands) > 300:
                cands = np.quantile(pair_s, np.linspace(0.002, 0.998, 300))
            res = {}
            for obj in ("balanced", "s1"):
                diffs, taus = {}, []
                for fold in folds:
                    test = set(fold)
                    tr = ~np.isin(pair_q, list(test))
                    if obj == "balanced":
                        s, y = pair_s[tr], pair_y[tr]
                        vals = np.array([((s[y] > t).mean() + (s[~y] <= t).mean()) / 2 for t in cands])
                    else:
                        vals = np.zeros(len(cands))
                        for m in LISTS:
                            keep = np.array([q not in test for q in arrays[m][0]])
                            vals += top_after_filter(arrays[m][1][keep], arrays[m][2][keep], cands).mean(axis=1)
                    tau = float(cands[int(np.argmax(vals))])
                    taus.append(tau)
                    for q in fold:
                        after = np.mean([top_after_filter(arrays[m][1][[idx[m][q]]], arrays[m][2][[idx[m][q]]], [tau])[0, 0] for m in LISTS])
                        diffs[q] = float(after - orig[q])
                cell = aggregate(diffs)
                cell.update({"tau_mean": float(np.mean(taus)), "tau_min": float(min(taus)), "tau_max": float(max(taus)),
                             "tpr_at_mean_tau": 100 * float((pair_s[pair_y] > np.mean(taus)).mean()),
                             "tnr_at_mean_tau": 100 * float((pair_s[~pair_y] <= np.mean(taus)).mean())})
                res[obj] = cell
            out[name][d] = res
    for obj in ("balanced", "s1"):
        holm_cells(out, decs, obj)
    return out


def benchmark_matches(name):
    """{(caption, image_id): decision answers "match"} on the benchmark pairs, for the probe outcome per query."""
    out = {}
    if name in ("BLIP", "BLIP2", "FLAVA"):
        df = pd.read_csv(RESULTS / f"itm_benchmark_preds_{name}.csv", dtype=str, keep_default_na=False)
        for r in df.itertuples(index=False):
            if r.p_match != "":
                out[(r.caption, int(r.image_id))] = float(r.p_match) > 0.5
    elif name in ("Qwen25", "Qwen25p"):
        df, _, _ = _align_to_collection(pd.read_csv(RESULTS.parent / "data" / "qwen_itm" / "svo_probes_results.csv"))
        for r in df.itertuples(index=False):
            for col, img in (("pos_pred", r.pos_image_id), ("neg_pred", r.neg_image_id)):
                v = str(getattr(r, col)).strip().lower()
                if v in ("yes", "no"):
                    out[(r.sentence, int(img))] = v == "yes"
    elif name == "Qwen3":
        # rows are the 33,685 triplets of the collection in file order ("index" = row of filtered_svo.csv)
        df = pd.read_csv(RESULTS.parent / "data" / "qwen3_embed" / "svo_probes_results_pairwise_qwen3_embed.csv")
        coll = COLS["svo"].df
        assert len(df) == len(coll) and (df[["subj_neg", "verb_neg", "obj_neg"]].values.astype(bool)
                                         == coll[["subj_neg", "verb_neg", "obj_neg"]].values.astype(bool)).all(), "Qwen3 file misaligned"
        for i, cp, cn in zip(df["index"].values, df["correct_pos"].values, df["correct_neg"].values):
            row = coll.iloc[int(i)]
            for v, img, flip in ((cp, row.pos_image_id, False), (cn, row.neg_image_id, True)):
                v = str(v).strip().lower()
                if v in ("true", "false"):
                    out[(row.sentence, int(img))] = (v == "true") != flip
    elif name == "Qwen3R":
        df = pd.read_csv(RESULTS / "Qwen3R_bench_svo.csv", dtype=str, keep_default_na=False)
        for r in df.itertuples(index=False):
            if r.score_pos != "" and r.score_neg != "":
                out[(r.caption, int(r.pos_image))] = float(r.score_pos) > 0.5
                out[(r.caption, int(r.neg_image))] = float(r.score_neg) > 0.5
    elif name == "GCLIP":
        return None                                               # no threshold: the probe is the pairwise outcome (below)
    return out


def gclip_bench_scores():
    """{(caption, image_id): Grounding CLIP score} on the benchmark pairs, when the run exists."""
    path = RESULTS / "GCLIP_bench_svo.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {(r.caption, int(r.image_id)): float(r.score) for r in df.itertuples(index=False) if r.score != ""}


def gclip_bench_accuracy():
    """Pairwise accuracy of the Grounding CLIP score on the benchmark triplets (positive scored above the negative)."""
    sc = gclip_bench_scores()
    if not sc:
        return None
    df = COLS["svo"].df
    ok, by = [], {"subject": [], "verb": [], "object": []}
    for r in df.itertuples(index=False):
        a, b = sc.get((r.sentence, int(r.pos_image_id))), sc.get((r.sentence, int(r.neg_image_id)))
        if a is None or b is None:
            continue
        ok.append(a > b)
        by["subject" if r.subj_neg else ("verb" if r.verb_neg else "object")].append(a > b)
    return {"pairwise": 100 * float(np.mean(ok)), "n_items": len(ok), "by": {t: {"pairwise": 100 * float(np.mean(v)), "n": len(v)} for t, v in by.items()}}


def probe_crosstab(pairs, decs, per_query):
    """Per query: does the decision answer every benchmark pair of the query correctly at its default threshold (the
    query's probe), and how much does the score re-ordering change the judged Success@1 of the six lists for that
    query? Mean gain of the passing and the failing queries, their difference and a permutation test on it."""
    df = COLS["svo"].df
    by_cap = {s: g for s, g in df.groupby("sentence")}
    by_img = {int(i): g for i, g in df.groupby("pos_image_id")}
    out = {}
    rng = np.random.default_rng(0)
    for name, dec in decs.items():
        matches = benchmark_matches(name)
        scores = gclip_bench_scores() if matches is None else None
        if matches is None and not scores:
            out[name] = {d: {"n_pass": 0, "n_fail": 0, "n": 0} for d in DIRECTIONS}
            continue
        out[name] = {}
        for d in DIRECTIONS:
            qs = sorted(set.intersection(*[set(per_query[(name, "score", m, d)]) for m in LISTS]))
            passed, gains = {}, {}
            for q in qs:
                rows = by_cap.get(q) if d == "t2i" else by_img.get(int(q))
                if rows is None or len(rows) == 0:
                    continue
                verdicts = []
                for r in rows.itertuples(index=False):
                    if matches is not None:
                        a, b = matches.get((r.sentence, int(r.pos_image_id))), matches.get((r.sentence, int(r.neg_image_id)))
                        verdicts.append(None if a is None or b is None else (a and not b))
                    else:                                         # score only: the probe is the pairwise choice
                        a, b = scores.get((r.sentence, int(r.pos_image_id))), scores.get((r.sentence, int(r.neg_image_id)))
                        verdicts.append(None if a is None or b is None else (a > b))
                if any(v is None for v in verdicts):
                    continue
                passed[q] = all(verdicts)
                gains[q] = float(np.mean([per_query[(name, "score", m, d)][q] for m in LISTS]))
            qq = sorted(passed)
            p = np.array([passed[q] for q in qq]); g = np.array([gains[q] for q in qq])
            if p.sum() == 0 or (~p).sum() == 0:
                out[name][d] = {"n_pass": int(p.sum()), "n_fail": int((~p).sum()), "n": len(qq)}
                continue
            obs = g[p].mean() - g[~p].mean()
            null = np.array([g[perm].mean() - g[~perm].mean() for perm in (rng.permutation(p) for _ in range(10000))])
            out[name][d] = {"n_pass": int(p.sum()), "n_fail": int((~p).sum()), "n": len(qq),
                            "gain_pass": 100 * float(g[p].mean()), "gain_fail": 100 * float(g[~p].mean()),
                            "diff": 100 * float(obs), "p": float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)),
                            "helped_pass": int((g[p] > 0).sum()), "hurt_pass": int((g[p] < 0).sum()),
                            "helped_fail": int((g[~p] > 0).sum()), "hurt_fail": int((g[~p] < 0).sum())}
    return out


def zscore_rows(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0.0)


def interpolation(pairs, decs, grid=np.round(np.linspace(0, 1, 11), 1)):
    """Re-ordering by (1 - w) * z(first-stage similarity) + w * z(decision score), both standardised inside the list;
    w chosen on the other queries (N_FOLDS-fold cross-validation, maximising the judged Success@1 of the six training
    lists). Mean change over the six lists per decision and direction at the chosen w, and the full curve over w."""
    sim_file = RESULTS / "list_similarities.json"       # the similarity rows of the judged lists, for a copy without embeddings
    try:
        scorers = {m: Scorer(m) for m in LISTS}
    except FileNotFoundError:
        scorers, stored = None, json.load(open(sim_file))
    first = {}
    for m in LISTS:
        for d in DIRECTIONS:
            lists = own_labels("svo", pairs, m, d)
            for q, items in lists.items():
                if scorers is None:
                    first[(m, d, q)] = np.array(stored[m][d][str(q)])
                    continue
                row = []
                for _, c, _ in items[:10]:
                    try:
                        row.append(float(scorers[m].score(d, q if d == "t2i" else int(q), [int(c) if d == "t2i" else c])[0]))
                    except (KeyError, ValueError):
                        row.append(np.nan)
                row = np.array(row)
                if np.isnan(row).any():                           # candidates outside the embedded collection: below the rest
                    row[np.isnan(row)] = (np.nanmin(row) if not np.isnan(row).all() else 0.0) - 1e-6
                # the judged list is the stored order; the recomputed scores reproduce it up to ties and rare swaps, so
                # the score values are assigned in the stored rank order (w = 0 is then exactly the judged order)
                first[(m, d, q)] = np.sort(row)[::-1]
    if scorers is not None:
        json.dump({m: {d: {str(q): [float(x) for x in v] for (mm, dd, q), v in first.items() if mm == m and dd == d}
                       for d in DIRECTIONS} for m in LISTS}, open(sim_file, "w"), indent=0)
    out = {}
    for name, dec in decs.items():
        out[name] = {}
        for d in DIRECTIONS:
            arrays = {m: list_arrays(pairs, dec, m, d) for m in LISTS}
            qs = sorted(set.intersection(*[set(a[0]) for a in arrays.values()]))
            s1 = {}                                                # s1[m] -> (len(grid), nq) correctness of the top item
            for m in LISTS:
                mq, S, C = arrays[m]
                keep = [mq.index(q) for q in qs]
                S, C = S[keep], C[keep]
                S = np.where(np.isfinite(S), S, np.where(np.isfinite(S), S, np.inf).min(axis=1, keepdims=True) - 1e-6)
                F = np.stack([first[(m, d, q)] for q in qs])
                Z1, Z2 = zscore_rows(F), zscore_rows(S)
                tie = -1e-9 * np.arange(10)[None, :]
                tops = np.stack([np.argmax((1 - w) * Z1 + w * Z2 + tie, axis=1) for w in grid])
                s1[m] = C[np.arange(len(qs))[None, :], tops]
            mean_s1 = np.mean([s1[m] for m in LISTS], axis=0)    # (len(grid), nq)
            orig = mean_s1[0]                                     # w = 0: the first-stage order
            rng = np.random.default_rng(0)
            folds = np.array_split(rng.permutation(len(qs)), N_FOLDS)
            diffs, ws = {}, []
            for fold in folds:
                tr = np.ones(len(qs), bool); tr[fold] = False
                w_i = int(np.argmax(mean_s1[:, tr].mean(axis=1)))
                ws.append(float(grid[w_i]))
                for i in fold:
                    diffs[qs[i]] = float(mean_s1[w_i, i] - orig[i])
            cell = aggregate(diffs)
            curve = {str(w): 100 * float((mean_s1[i] - orig).mean()) for i, w in enumerate(grid)}
            cell.update({"w_mean": float(np.mean(ws)), "w_min": float(min(ws)), "w_max": float(max(ws)), "curve": curve,
                         "best_w": str(grid[int(np.argmax([curve[str(w)] for w in grid]))]),
                         "score_only": 100 * float((mean_s1[-1] - orig).mean())})
            out[name][d] = cell
    holm_cells(out, decs)
    return out


def bench_itm_from_scores(path, pos_col="score_pos", neg_col="score_neg", by=None):
    """ITM accuracy (threshold 0.5), TPR, TNR and pairwise accuracy from stored scores of the positive and negative pair."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[(df[pos_col] != "") & (df[neg_col] != "")]
    sp, sn = df[pos_col].astype(float).values, df[neg_col].astype(float).values
    out = {"overall": 100 * float(np.concatenate([sp > 0.5, sn <= 0.5]).mean()), "tpr": 100 * float((sp > 0.5).mean()),
           "tnr": 100 * float((sn <= 0.5).mean()), "pairwise": 100 * float((sp > sn).mean()), "n_items": int(len(df))}
    if by:
        out["by"] = {g: {"overall": 100 * float(np.concatenate([sp[idx] > 0.5, sn[idx] <= 0.5]).mean()),
                         "pairwise": 100 * float((sp[idx] > sn[idx]).mean()), "n": int(idx.sum())}
                     for g in sorted(df[by].unique()) for idx in [df[by].values == g]}
    return out


out = {"decisions": DECISIONS, "decisions_by_bench": DECISIONS_BY_BENCH, "bench_acc": {}}
bench_itm = json.load(open(RESULTS / "itm_benchmark.json"))
coco_itm = json.load(open(RESULTS / "coco_itm.json")) if (RESULTS / "coco_itm.json").exists() else {}
for k in ALL_DECISIONS:
    base = "Qwen25" if k == "Qwen25p" else k
    if k == "Qwen3R":
        continue
    if k in DECISIONS_BY_BENCH["svo"] and base in bench_itm:
        out["bench_acc"].setdefault("svo", {})[k] = {x: bench_itm[base][x]["mean"] for x in ("overall", "tpr", "tnr")}
    if k in DECISIONS_BY_BENCH["coco"] and base in coco_itm:
        out["bench_acc"].setdefault("coco", {})[k] = {x: coco_itm[base]["sugarcrepe"][x]["mean"] for x in ("overall", "tpr", "tnr")}
for bench, fname, by in (("svo", "Qwen3R_bench_svo.csv", "neg_type"), ("coco", "Qwen3R_bench_sugarcrepe.csv", "subset")):
    if "Qwen3R" in DECISIONS_BY_BENCH[bench] and (RESULTS / fname).exists():
        out["bench_acc"][bench]["Qwen3R"] = bench_itm_from_scores(RESULTS / fname, by=by)
if "GCLIP" in DECISIONS_BY_BENCH["svo"]:
    g = gclip_bench_accuracy()
    if g:
        out["bench_acc"]["svo"]["GCLIP"] = {"overall": float("nan"), "tpr": float("nan"), "tnr": float("nan"), **g}

for bench in [b for b in ("svo", "coco") if DECISIONS_BY_BENCH[b]]:   # COCO only when its stored decisions exist
    decs_b = DECISIONS_BY_BENCH[bench]
    pairs = load_pairs(bench)
    decs = {k: load_decision(bench, k) for k in decs_b}
    sep = {k: {d: separation(pairs, decs[k], d) for d in DIRECTIONS} for k in decs_b}
    res, mean, pq = rerank(bench, pairs, decs, models=LISTS)       # the same five first stages on both benchmarks
    out[bench] = {"decisions": decs_b, "separation": sep, "rerank": res, "rerank_mean": mean}
    if bench == "svo":
        out["lists"] = LISTS
        # how incomplete the benchmark labels are: judged-correct pool pairs that are benchmark positives; the pool's
        # share of correct pairs (the benchmark's is 50%); pairs that occur in several lists with conflicting labels
        out[bench]["gold_share"], out[bench]["pool"] = {}, {}
        for d in DIRECTIONS:
            n_corr, n_gold, n_pairs, n_multi, n_conflict = 0, 0, 0, 0, 0
            for q, cs in pairs[d].items():
                g = gold(bench, d, q)
                for c, v in cs.items():
                    n_pairs += 1
                    if v["label"]:
                        n_corr += 1
                        n_gold += c in g
                    if len(v["labels"]) > 1:
                        n_multi += 1
                        n_conflict += len(set(v["labels"])) > 1
            out[bench]["gold_share"][d] = {"judged_correct": n_corr, "benchmark_positives": n_gold, "share": 100 * n_gold / n_corr}
            out[bench]["pool"][d] = {"pairs": n_pairs, "share_correct": 100 * n_corr / n_pairs, "in_several_lists": n_multi,
                                     "conflicting_labels": n_conflict, "conflict_share": 100 * n_conflict / max(n_multi, 1)}
        # Spearman between ITM accuracy and the retrieval measures across the decisions of the paper
        from scipy.stats import spearmanr
        shown = [k for k in decs_b if not (k == "Qwen25" and "Qwen25p" in decs_b) and k in out["bench_acc"]["svo"]
                 and k not in SCORE_ONLY]                          # Spearman over the decisions with an ITM accuracy
        acc = [out["bench_acc"]["svo"][k]["overall"] for k in shown]
        out[bench]["spearman"] = {"decisions": shown,
                                  "list_auc": float(spearmanr(acc, [mean["within_list_auc"][k]["mean"] for k in shown])[0])}
        for d in DIRECTIONS:
            out[bench]["spearman"][d] = {"pool_auc": float(spearmanr(acc, [sep[k][d]["auc"] for k in shown])[0]),
                                         "gain": float(spearmanr(acc, [mean[k][d]["score"]["all"]["diff"] for k in shown])[0])}
        out[bench]["errors"] = {k: error_types(pairs, decs[k]) for k in decs_b}
        if HAS_O3:
            o3p = o3_pairs_svo()
            out[bench]["errors_o3"] = {k: error_types(o3p, decs[k]) for k in decs_b}
        out[bench]["tuned"] = tuned_filter(pairs, decs)
        try:                                                  # needs the per-pair benchmark answers of every decision
            out[bench]["probe"] = probe_crosstab(pairs, decs, pq)
        except FileNotFoundError as e:                        # not part of the released copy
            print("probe cross-tab skipped:", e)
        out[bench]["interp"] = interpolation(pairs, decs)
        out[bench]["learning_curve"] = learning_curve(pq, decs_b, LISTS)
    print(f"\n=== {bench}: separation (pool AUC [CI], TPR, TNR, accuracy, n) and within-list AUC")
    for k in decs_b:
        print(" ", k, " | ".join(f"{d}: {sep[k][d]['auc']:.2f} [{sep[k][d]['lo']:.2f}, {sep[k][d]['hi']:.2f}] tpr {sep[k][d]['tpr']:.0f} tnr {sep[k][d]['tnr']:.0f} acc {sep[k][d]['acc']:.1f} n {sep[k][d]['n']}" for d in DIRECTIONS),
              "| list AUC %.2f" % mean["within_list_auc"][k]["mean"])
    print("  first stage list AUC %.2f" % mean["within_list_auc"]["first_stage"]["mean"])
    print(f"=== {bench}: judged S@1 of each list, original -> re-ordered by score (binary), Holm-corrected p of the score variant")
    for d in DIRECTIONS:
        for m in LISTS:
            c = res[m][d]
            print(f"  {d} {m:8s} orig {c['orig']:5.1f} strict {c['strict_orig']:5.1f} share {c['share']:5.1f} oracle {c['oracle']:5.1f} | " + " | ".join(
                f"{k}: {c[k]['score']['after']:5.1f} ({c[k]['binary']['after']:5.1f}) p={c[k]['score']['p_holm']:.3f}" for k in decs_b))
    print(f"=== {bench}: mean gain over the six lists (score; binary; strict labels; depth 2/3/5), and without the own list")
    for k in decs_b:
        for d in DIRECTIONS:
            c = mean[k][d]
            own = c['score']['own']
            print(f"  {k:7s} {d}: {c['score']['all']['diff']:+.1f} [{c['score']['all']['lo']:+.1f}, {c['score']['all']['hi']:+.1f}] p={c['score']['all']['p_holm']:.4f}"
                  f" other {c['score']['other']['diff']:+.1f} own {('%+.1f' % own) if own is not None else '-'} | binary {c['binary']['all']['diff']:+.1f} p={c['binary']['all']['p_holm']:.4f}"
                  f" | strict {c['score']['strict']['diff']:+.1f} p={c['score']['strict']['p_holm']:.3f}"
                  f" | depth " + " ".join(f"{kk}:{c['score']['depth'][kk]['diff']:+.1f}" for kk in DEPTHS))
    print("  BLIP-2 head minus FLAVA head:", " | ".join(
        f"{d}: {mean['BLIP2_minus_FLAVA'][d]['diff']:+.1f} [{mean['BLIP2_minus_FLAVA'][d]['lo']:+.1f}, {mean['BLIP2_minus_FLAVA'][d]['hi']:+.1f}] p={mean['BLIP2_minus_FLAVA'][d]['p']:.4f}" for d in DIRECTIONS))
    if bench == "svo":
        print("=== svo: judged MRR and nDCG@10 change (score re-ordering, mean over the lists); ECE on the pool; benchmark positives among judged-correct pairs; Spearman with ITM accuracy")
        for k in decs_b:
            print(f"  {k:7s} " + " | ".join(f"{d}: MRR {mean[k][d]['score']['mrr']['diff']:+.1f} nDCG {mean[k][d]['score']['ndcg']['diff']:+.1f} ECE {sep[k][d]['ece']:.3f}" for d in DIRECTIONS))
        print("  learning curve:", json.dumps(out[bench]["learning_curve"]))
        print("  gold share:", {d: round(v["share"], 1) for d, v in out[bench]["gold_share"].items()}, "| pool:", out[bench]["pool"], "| spearman:", out[bench]["spearman"])
        print("=== svo: filter with a tuned threshold (balanced accuracy / Success@1 objective, cross-validated), mean change")
        for k in decs_b:
            for d in DIRECTIONS:
                t = out[bench]["tuned"][k][d]
                print(f"  {k:7s} {d}: balanced {t['balanced']['diff']:+.1f} [{t['balanced']['lo']:+.1f}, {t['balanced']['hi']:+.1f}] p={t['balanced']['p_holm']:.4f} tau {t['balanced']['tau_mean']:.3f} tpr/tnr {t['balanced']['tpr_at_mean_tau']:.0f}/{t['balanced']['tnr_at_mean_tau']:.0f}"
                      f" | s1 {t['s1']['diff']:+.1f} p={t['s1']['p_holm']:.4f} tau {t['s1']['tau_mean']:.3f}")
        print("=== svo: probe outcome per query (all benchmark pairs of the query right at the default threshold) vs the gain of the score re-ordering")
        for k in (decs_b if "probe" in out[bench] else []):
            for d in DIRECTIONS:
                t = out[bench]["probe"][k][d]
                if "diff" in t:
                    print(f"  {k:7s} {d}: pass n={t['n_pass']} gain {t['gain_pass']:+.1f} | fail n={t['n_fail']} gain {t['gain_fail']:+.1f} | diff {t['diff']:+.1f} p={t['p']:.3f}")
                else:
                    print(f"  {k:7s} {d}: pass n={t['n_pass']} fail n={t['n_fail']} (no contrast)")
        print("=== svo: interpolation with the first-stage score (w chosen by cross-validation), mean change; curve over w")
        for k in decs_b:
            for d in DIRECTIONS:
                t = out[bench]["interp"][k][d]
                print(f"  {k:7s} {d}: {t['diff']:+.1f} [{t['lo']:+.1f}, {t['hi']:+.1f}] p={t['p_holm']:.4f} w {t['w_min']:.1f}-{t['w_max']:.1f} (best w {t['best_w']}, score only {t['score_only']:+.1f}) | "
                      + " ".join(f"{w}:{v:+.1f}" for w, v in t['curve'].items()))

# SVO-Probes label check: the re-ordering result under other label rules. "o3": the LLM assessor's label of every list
# entry (all twelve lists; a query whose list has an entry without an o3 label is left out); "annotator_k": the k-th
# annotator alone, "unanimous" / "any": all three or any of them, on the four image-query lists with three annotators
# (with "majority" on the same lists for reference). Mean change of the score re-ordering per decision and direction,
# the BLIP-2 minus FLAVA contrast, and the per-list cells.
LABEL_RULES = {
    "o3": (lambda j: None if j["o3"] is None else j["o3"] == "correct", LISTS, DIRECTIONS),
    "human_on_o3_queries": (None, LISTS, DIRECTIONS),                       # filled below: the paper's rule on the o3 queries
    "majority": (lambda j: is_correct(j) if len(j["votes"]) == 3 else None, DUAL_ENCODERS, ["i2t"]),
    "annotator_1": (lambda j: j["votes"][0] == 1 if len(j["votes"]) == 3 else None, DUAL_ENCODERS, ["i2t"]),
    "annotator_2": (lambda j: j["votes"][1] == 1 if len(j["votes"]) == 3 else None, DUAL_ENCODERS, ["i2t"]),
    "annotator_3": (lambda j: j["votes"][2] == 1 if len(j["votes"]) == 3 else None, DUAL_ENCODERS, ["i2t"]),
    "unanimous": (lambda j: all(v == 1 for v in j["votes"]) if len(j["votes"]) == 3 else None, DUAL_ENCODERS, ["i2t"]),
    "any": (lambda j: any(v == 1 for v in j["votes"]) if len(j["votes"]) == 3 else None, DUAL_ENCODERS, ["i2t"]),
}
if not HAS_O3:
    LABEL_RULES = {r: v for r, v in LABEL_RULES.items() if r not in ("o3", "human_on_o3_queries")}
svo_pairs = load_pairs("svo")
svo_decs = {k: load_decision("svo", k) for k in DECISIONS}
# the paper's own labels on the queries that have a complete o3 label in every list of the direction, for reference
o3_complete = {d: set.intersection(*[set(own_labels("svo", svo_pairs, m, d, LABEL_RULES["o3"][0])) for m in LISTS]) for d in DIRECTIONS} if HAS_O3 else None
label_check = {}
for rule, (fn, ms, ds) in LABEL_RULES.items():
    keep = o3_complete if rule == "human_on_o3_queries" else None
    res_r, mean_r, _ = rerank("svo", svo_pairs, svo_decs, with_extras=False, models=ms, directions=ds, label_fn=fn, keep=keep)
    label_check[rule] = {
        "models": ms, "directions": ds,
        "mean": {k: {d: mean_r[k][d]["score"]["all"] for d in ds} for k in DECISIONS},
        "mean_binary": {k: {d: mean_r[k][d]["binary"]["all"] for d in ds} for k in DECISIONS},
        "contrast": mean_r["BLIP2_minus_FLAVA"],
        "cells": {m: {d: {"n": res_r[m][d]["n"], "orig": res_r[m][d]["orig"],
                          **{k: {"after": res_r[m][d][k]["score"]["after"], "diff": res_r[m][d][k]["score"]["diff"],
                                 "p_holm": res_r[m][d][k]["score"]["p_holm"]} for k in DECISIONS}}
                      for d in ds} for m in ms},
    }
out["svo"]["label_check"] = label_check
print("\n=== SVO-Probes label check: mean change of the score re-ordering (Holm p) under other label rules")
for rule, lc in label_check.items():
    for d in lc["directions"]:
        print(f"  {rule:20s} {d}: " + " | ".join(f"{k}: {lc['mean'][k][d]['diff']:+.1f} [{lc['mean'][k][d]['lo']:+.1f}, {lc['mean'][k][d]['hi']:+.1f}] p={lc['mean'][k][d]['p_holm']:.3f} n={lc['mean'][k][d]['n']}" for k in DECISIONS)
              + f" | BLIP2-FLAVA {lc['contrast'][d]['diff']:+.1f} p={lc['contrast'][d]['p']:.4f}")

# COCO check with the expert's labels: the queries the expert labelled, judged by the expert and by o3
check = {}
for judge in (("human", "o3") if DECISIONS_BY_BENCH["coco"] else ()):
    pairs = load_pairs("coco", "human")
    if judge == "o3":
        full = load_pairs("coco", "o3")
        pairs = {d: {q: full[d][q] for q in pairs[d]} for d in DIRECTIONS}
    decs = {k: load_decision("coco", k) for k in DECISIONS_BY_BENCH["coco"]}
    _, mean, _ = rerank("coco", pairs, decs, with_extras=False, models=LISTS)
    check[judge] = {k: {d: {mode: mean[k][d][mode]["all"] for mode in ("score", "binary")} for d in DIRECTIONS} for k in decs}
    check[judge]["n_queries"] = {d: len(pairs[d]) for d in DIRECTIONS}
if DECISIONS_BY_BENCH["coco"]:
    out["coco"]["expert_check"] = check
print("\n=== COCO, queries labelled by the expert: mean gain (score) under the expert's and under o3's labels")
for k in DECISIONS_BY_BENCH["coco"]:
    print(" ", k, " | ".join(f"{d}: expert {check['human'][k][d]['score']['diff']:+.1f}, o3 {check['o3'][k][d]['score']['diff']:+.1f}" for d in DIRECTIONS))

for key, label in (("errors", "human"), ("errors_o3", "o3")):
    if key not in out["svo"]:
        continue
    print(f"\n=== SVO-Probes ({label} labels): accepted share of incorrect pairs by error type (both directions), and AUC against correct pairs")
    for k in DECISIONS:
        e = out["svo"][key][k]["both"]
        print(" ", k, " | ".join(f"{t}: {e[t]['accept']:.1f}% (n={e[t]['n']}, AUC {e[t]['auc']:.2f})" for t in ("subject", "verb", "object")))
print("\n=== benchmark ITM accuracy:", {b: {k: round(v["overall"], 1) for k, v in out["bench_acc"][b].items()} for b in out["bench_acc"]})

json.dump(out, open(RESULTS / "short_matching.json", "w"), indent=1)
print("\nsaved", RESULTS / "short_matching.json")
