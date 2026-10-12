"""Check that the numbers quoted in the paper on matching decisions (SVO-Probes, human labels) match
results/short_matching.json. Each expected string is rebuilt from the data and must occur in the section files;
thresholds stated in the prose are asserted directly. Ranges in the prose are rounded half up (10.5 -> 11).
The paper uses human labels only: the o3 label check and the COCO analyses of scripts/50 are not quoted."""
import json
import sys
import numpy as np
from svo_eval.paths import RESULTS, ECIR, MODELS, DIRECTIONS
from svo_eval.metrics import holm

A = json.load(open(RESULTS / "short_matching.json"))
text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((ECIR / "paper" / "sections").glob("*.tex")))
S = A["svo"]
LISTS = A.get("lists", MODELS)                           # the judged lists of the paper
PROMPT = "Qwen25p" if "Qwen25p" in A["decisions"] else "Qwen25"
HAS_GCLIP = "GCLIP" in A["decisions"]
DEC = ["BLIP", "BLIP2", "FLAVA", PROMPT, "Qwen3", "Qwen3R"] + (["GCLIP"] if HAS_GCLIP else [])   # the decisions of the paper
THRESHOLDED = [k for k in DEC if k != "GCLIP"]               # decisions with a default threshold (ITM accuracy, filter, TPR/TNR)
QWEN = [PROMPT, "Qwen3", "Qwen3R"]
UP = ["BLIP", "BLIP2", PROMPT, "Qwen3", "Qwen3R"] + (["GCLIP"] if HAS_GCLIP else [])             # those that improve the lists
UP_T = [k for k in UP if k != "GCLIP"]                       # improving decisions with a default threshold
bench = A["bench_acc"]["svo"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]


def half_up(x):
    return int(abs(x) + 0.5 + 1e-9)


def pair(vals, nd=1, sep=" and "):
    return sep.join(f"{abs(v):.{nd}f}" for v in vals)


def rng(vals):
    return f"{min(half_up(v) for v in vals)} to {max(half_up(v) for v in vals)}"


def gain(k, mode="score", key="all"):
    return [S["rerank_mean"][k][d][mode][key]["diff"] for d in DIRECTIONS]


def auc(k):
    return [S["separation"][k][d]["auc"] for d in DIRECTIONS]


def lauc(k):
    return S["rerank_mean"]["within_list_auc"][k]["mean"]


def orig(m):
    return [half_up(S["rerank"][m][d]["orig"]) for d in DIRECTIONS]


def p_mean(k, d, mode="score", key="all"):
    return S["rerank_mean"][k][d][mode][key]["p_holm"]


def tpr(k, d):
    return S["separation"][k][d]["tpr"]


def tnr(k, d):
    return S["separation"][k][d]["tnr"]


con = S["rerank_mean"]["BLIP2_minus_FLAVA"]
err = lambda k, t: S["errors"][k]["both"][t]
strict_changes = [gain(k, "score", "strict")[i] for k in DEC for i in (0, 1)]
depth3_share = [S["rerank_mean"][k][d]["score"]["depth"]["3"]["diff"] / S["rerank_mean"][k][d]["score"]["all"]["diff"] for k in UP for d in DIRECTIONS]
LC = S["label_check"]                                        # the per-annotator check on the triple-annotated lists (human)
ann = [LC[f"annotator_{i}"]["mean"] for i in (1, 2, 3)]
maj = LC["majority"]["mean"]
tuned = lambda k: [S["tuned"][k][d]["s1"]["diff"] for d in DIRECTIONS]
interp = lambda k: [S["interp"][k][d]["diff"] for d in DIRECTIONS]
interp_maxdev = max(abs(interp(k)[i] - gain(k)[i]) for k in UP for i in (0, 1))
mrr = lambda k: [S["rerank_mean"][k][d]["score"]["mrr"]["diff"] for d in DIRECTIONS]
mrr_ratios = [mrr(k)[i] / gain(k)[i] for k in DEC for i in (0, 1)]
ndcg_ratios = [S["rerank_mean"][k][d]["score"]["ndcg"]["diff"] / gain(k)[i] for k in DEC for i, d in enumerate(DIRECTIONS)]
rho_gain_both = [S["spearman"][d]["gain"] for d in DIRECTIONS]
rho_auc_both = [S["spearman"][d]["pool_auc"] for d in DIRECTIONS]
rho_gain, rho_auc = rho_gain_both[0], rho_auc_both[0]
tnr_drop = [bench[k]["tnr"] - tnr(k, d) for k in THRESHOLDED for d in DIRECTIONS]
ece = lambda k: max(S["separation"][k][d]["ece"] for d in DIRECTIONS)                  # the larger of the two directions
ece_lo = lambda k: min(S["separation"][k][d]["ece"] for d in DIRECTIONS)
ece_heads = [S["separation"][k][d]["ece"] for k in ("BLIP", "BLIP2", "FLAVA") for d in DIRECTIONS]
verb_margin = [err(k, "verb")["accept"] - max(err(k, "subject")["accept"], err(k, "object")["accept"]) for k in ("BLIP", "BLIP2", "FLAVA", PROMPT, "Qwen3R")]
LCV = S["learning_curve"]
flava_interp_min = min(interp("FLAVA"))
flava_interp_text = ("removes the loss of the FLAVA head" if flava_interp_min > -0.5
                     else f"reduces the loss of the FLAVA head to at most {half_up(abs(flava_interp_min))} point")

expected = {
    "ITM accuracy of the heads": f"{bench['BLIP2']['overall']:.1f}\\% and {bench['FLAVA']['overall']:.1f}\\%",
    "ITM accuracy around 63%": ("both BLIP-2 and FLAVA have an ITM accuracy around 63\\%" if all(62 <= bench[k]["overall"] <= 64 for k in ("BLIP2", "FLAVA")) else "ACCURACY NOT AROUND 63"),
    "within-list AUC": f"the FLAVA head does not: its AUC inside the lists is below the {lauc('first_stage'):.2f} of the original order",
    "TNR falls on the retrieved pairs": f"by {rng(tnr_drop)} points of TNR",
    "BLIP-2 gain": f"the BLIP-2 head by {half_up(min(gain('BLIP2')))} or more",
    "FLAVA loss": f"The FLAVA head lowers it by {half_up(min(abs(v) for v in gain('FLAVA')))} and {half_up(max(abs(v) for v in gain('FLAVA')))} points",
    "head contrast": f"their difference of {rng([con[d]['diff'] for d in DIRECTIONS])} points is ($p<0.001$)",
    "BLIP head: gain": (lambda a, b: f"the BLIP head by {a}," if a == b else f"the BLIP head by {a} to {b},")(half_up(min(gain('BLIP'))), half_up(max(gain('BLIP')))),
    "Qwen gains": f"the three Qwen-based decisions by {half_up(min(gain(k)[i] for k in QWEN for i in (0, 1)))} points or more",
    "GCLIP gain": (f"Grounding CLIP by {half_up(min(gain('GCLIP')))} or more." if HAS_GCLIP else ""),
    "spearman": f"(Spearman's $\\rho$ {rho_gain:.2f})",
    "spearman: accuracy against pool AUC": f"as by the AUC of their score over the judged pairs (Spearman's $\\rho$ {rho_auc:.2f})",
    "spacing": f"while the {half_up(bench['BLIP']['overall'] - bench['BLIP2']['overall'])} points between the BLIP and BLIP-2 heads buy {rng([gain('BLIP')[i] - gain('BLIP2')[i] for i in (0, 1)])}",
    "learning curve": f"it is found in {half_up(min(LCV[d]['50']['contrast_significant_share'] for d in DIRECTIONS))}\\% of random subsets of 50 queries",
    "judged MRR and nDCG": "Judged MRR and nDCG@10 move the same way",
    "default filter exception": "every improving decision keeps only part of its gain;",
    "Qwen3 default": "the lenient \\qemb{} decision keeps about a third" if all(0.25 <= gain('Qwen3', 'binary')[i] / gain('Qwen3')[i] <= 0.4 for i in (0, 1)) else "QWEN3 FILTER NOT ABOUT A THIRD",
    "tuned threshold": "helps only the \\qemb{} decision, and the tuned filters of the three Qwen-based decisions still stay below their scores",
    "interpolation": f"changes the improving decisions by less than {int(np.floor(interp_maxdev)) + 1} points and {flava_interp_text}",
    "gold share": f"they are {half_up(S['gold_share']['t2i']['share'])}\\% of the judged-correct candidates",
    "strict changes": f"move it by at most {half_up(max(abs(v) for v in strict_changes))} points in either direction, and none of these changes is significant but one loss of the FLAVA head",
    "annotators": "Each single annotator of the triple-annotated lists gives the same picture as the majority vote",
    "RQ3: verb errors margin": f"by {rng(verb_margin)} points",
    "abstract: heads": f"raises the judged Success@1 by {rng(gain('BLIP2'))} points while re-ordering by the FLAVA head lowers it by {rng(gain('FLAVA'))}.",
    "abstract: ITM accuracy": f"({bench['BLIP2']['overall']:.1f}\\% and {bench['FLAVA']['overall']:.1f}\\%)",
    "cost": "the heads cost a 0.2--1.1B-parameter forward pass, the two Qwen3 decisions a 2B one and the prompt a 7B one",
}
checks = {
    "BLIP-2 gain: the image-query gain is significant after correction, the caption-query gain is not": p_mean("BLIP2", "i2t") < 0.05 <= p_mean("BLIP2", "t2i"),
    "neither head's effect is significant in both directions after correction": all(max(p_mean(k, d) for d in DIRECTIONS) >= 0.05 for k in ("BLIP2", "FLAVA")),
    "head contrast: p<0.001 in both directions": all(con[d]["p"] < 0.001 for d in DIRECTIONS),
    "Qwen decisions: p<0.001": all(p_mean(k, d) < 0.001 for k in QWEN for d in DIRECTIONS),
    "BLIP head: corrected p<0.01 in both directions": all(p_mean("BLIP", d) < 0.01 for d in DIRECTIONS),
    "the two heads are within one point of ITM accuracy": abs(bench["BLIP2"]["overall"] - bench["FLAVA"]["overall"]) <= 1.05,
    "the prompt has the highest pool AUC": all(auc(PROMPT)[i] >= max(auc(k)[i] for k in DEC) for i in (0, 1)),
    "prompt and re-ranker are balanced (TPR and TNR within 20 points)":
        all(abs(tpr(k, d) - tnr(k, d)) < 20 for k in (PROMPT, "Qwen3R") for d in DIRECTIONS),
    "TNR falls on the retrieved pairs for every thresholded decision": all(v > 0 for v in tnr_drop),
    "BLIP-2 head is strict: benchmark TPR below 50 and TNR above 90, lower TPR on the retrieved pairs": bench["BLIP2"]["tpr"] < 50 and bench["BLIP2"]["tnr"] > 90 and all(tpr("BLIP2", d) < bench["BLIP2"]["tpr"] for d in DIRECTIONS),
    "BLIP head leans the same way: TNR above TPR on the benchmark and on the retrieved pairs": bench["BLIP"]["tnr"] > bench["BLIP"]["tpr"] and all(tnr("BLIP", d) > tpr("BLIP", d) for d in DIRECTIONS),
    "Qwen3 decision is lenient: accepts at least 90% of the benchmark positives and 40-60% of the incorrect retrieved pairs": bench["Qwen3"]["tpr"] >= 90 and all(40 <= 100 - tnr("Qwen3", d) <= 60 for d in DIRECTIONS),
    "judged pairs are about as balanced as the benchmark (45-55% correct)": all(45 <= S["pool"][d]["share_correct"] <= 55 for d in DIRECTIONS),
    "the FLAVA head is the only decision with a within-list AUC below the first-stage order": lauc("FLAVA") < lauc("first_stage") and all(lauc(k) > lauc("first_stage") for k in DEC if k != "FLAVA"),
    "the BLIP-2 head has a clearly higher pool AUC than the FLAVA head (by at least 0.05 in both directions)": all(auc("BLIP2")[i] - auc("FLAVA")[i] >= 0.05 for i in (0, 1)),
    "every decision but the FLAVA head improves the lists on average in both directions": all(gain(k)[i] > 0 for k in DEC if k != "FLAVA" for i in (0, 1)) and all(v < 0 for v in gain("FLAVA")),
    "GCLIP gain is not significant in either direction after Holm": all(p_mean("GCLIP", d) >= 0.05 for d in DIRECTIONS),
    "learning curve: the share at 50 queries is the same in both directions": len({half_up(LCV[d]["50"]["contrast_significant_share"]) for d in DIRECTIONS}) == 1,
    "the gold share rounds to the same value in both directions": len({half_up(S["gold_share"][d]["share"]) for d in DIRECTIONS}) == 1,
    "MRR and nDCG move in the same direction as Success@1 for every decision": min(mrr_ratios) > 0 and min(ndcg_ratios) > 0,
    "the largest cell of each direction lies on a CLIP list": all(max(((S["rerank"][m][d][k]["score"]["after"] - S["rerank"][m][d]["orig"], m) for m in LISTS for k in DEC))[1] == "CLIP" for d in DIRECTIONS),
    "the BLIP-2 head does not improve the BLIP-2 Q-Former's own lists": all(S["rerank"]["BLIP2"][d]["BLIP2"]["score"]["after"] - S["rerank"]["BLIP2"][d]["orig"] <= 0 for d in DIRECTIONS),
    "tuning the threshold helps only the Qwen3 decision (others gain under 3 points over their default filter)":
        all(tuned("Qwen3")[i] - gain("Qwen3", "binary")[i] > 4 for i in (0, 1)) and all(tuned(k)[i] - gain(k, "binary")[i] < 3 for k in DEC if k != "Qwen3" for i in (0, 1)),
    "the lenient Qwen3 filter lets about half of the incorrect retrieved pairs through (40-60%)": all(40 <= 100 - tnr("Qwen3", d) <= 60 for d in DIRECTIONS),
    "strict Success@1: exactly one significant change, a loss of the FLAVA head": sum(p_mean(k, d, "score", "strict") < 0.05 for k in DEC for d in DIRECTIONS) == 1 and any(p_mean("FLAVA", d, "score", "strict") < 0.05 and gain("FLAVA", "score", "strict")[i] < 0 for i, d in enumerate(DIRECTIONS)),
    "ITM accuracy orders the thresholded decisions as the pool AUC does (rho 1.00 in both directions)": all(round(r, 2) == 1.0 for r in rho_auc_both),
    "CLIP lists are the weakest": all(orig("CLIP")[i] == min(orig(m)[i] for m in LISTS) for i in (0, 1)),
    "no decision improves the Qwen3 image-query list significantly; it is the strongest list (90)":
        all(not (S["rerank"]["Qwen3"]["i2t"][k]["score"]["p_holm"] < 0.05 and S["rerank"]["Qwen3"]["i2t"][k]["score"]["diff"] > 0) for k in DEC)
        and half_up(S["rerank"]["Qwen3"]["i2t"]["orig"]) == 90 == max(orig(m)[i] for m in LISTS for i in (0, 1)),
    "strict labels: nothing significant except one FLAVA loss":
        [(k, d) for k in DEC for d in DIRECTIONS if p_mean(k, d, "score", "strict") < 0.05] == [("FLAVA", "i2t")],
    "default thresholds discard up to two thirds of the gain":
        0.60 <= max(1 - gain(k, "binary")[i] / gain(k)[i] for k in UP_T for i in (0, 1)) <= 0.70,
    "default filter: exactly one of the ten improving cells keeps more than the score": sum(gain(k, "binary")[i] > gain(k)[i] for k in UP_T for i in (0, 1)) == 1,
    "re-ordering the top 2, 3 or 5 never changes the sign":
        all((S["rerank_mean"][k][d]["score"]["depth"][kk]["diff"] > 0) == (gain(k)[i] > 0) for k in DEC for i, d in enumerate(DIRECTIONS) for kk in ("2", "3", "5")),
    "five decisions accept verb errors most; Qwen3 accepts object errors most":
        all(err(k, "verb")["accept"] > max(err(k, "subject")["accept"], err(k, "object")["accept"]) for k in ("BLIP", "BLIP2", "FLAVA", PROMPT, "Qwen3R"))
        and err("Qwen3", "object")["accept"] > max(err("Qwen3", "subject")["accept"], err("Qwen3", "verb")["accept"]),
    "verb errors: lowest AUC against correct pairs for the five decisions": all(err(k, "verb")["auc"] < min(err(k, t)["auc"] for t in ("subject", "object")) for k in ("BLIP", "BLIP2", "FLAVA", PROMPT, "Qwen3R")),
    "prompt matches the re-ranker (within 5 points or above in both directions)": all(gain(PROMPT)[i] >= gain("Qwen3R")[i] - 5 for i in (0, 1)),
    "annotators: every annotator alone keeps the sign of every decision's effect (image queries)":
        all((a[k]["i2t"]["diff"] > 0) == (maj[k]["i2t"]["diff"] > 0) for a in ann for k in DEC),
    "annotators: head contrast p<0.001 under each annotator": all(LC[f"annotator_{i}"]["contrast"]["i2t"]["p"] < 0.001 for i in (1, 2, 3)),
    "annotators: the three-vote lists are the four dual encoders' image-query lists":
        all(LC[f"annotator_{i}"]["models"] == ["CLIP", "BLIP2", "FLAVA", "SigLIP2"] and LC[f"annotator_{i}"]["directions"] == ["i2t"] for i in (1, 2, 3)),
    "tuned threshold: the tuned filters of the Qwen-based decisions stay below their scores": all(tuned(k)[i] < gain(k)[i] for k in QWEN for i in (0, 1)),
    "tuned threshold: the Qwen3 filter improves on its default": all(tuned("Qwen3")[i] > gain("Qwen3", "binary")[i] for i in (0, 1)),
    "interpolation: the improving decisions change by less than 3 points": interp_maxdev < 3,
    "spearman: the same value in both directions (2 decimals)": round(rho_gain_both[0], 2) == round(rho_gain_both[1], 2) and round(rho_auc_both[0], 2) == round(rho_auc_both[1], 2),
    "judged MRR: same sign as Success@1 for every decision and direction": all((mrr(k)[i] > 0) == (gain(k)[i] > 0) for k in DEC for i in (0, 1)),
    "nDCG: same sign as Success@1 for every decision and direction": all((S["rerank_mean"][k][d]["score"]["ndcg"]["diff"] > 0) == (gain(k)[i] > 0) for k in DEC for i, d in enumerate(DIRECTIONS)),
    "gold share: both directions round to the same percentage": half_up(S["gold_share"]["t2i"]["share"]) == half_up(S["gold_share"]["i2t"]["share"]),
    "GCLIP: not significant after correction, largest gain on CLIP's image-query list":
        (not HAS_GCLIP) or (all(p_mean("GCLIP", d) >= 0.05 for d in DIRECTIONS)
                            and max(((S["rerank"][m][d]["GCLIP"]["score"]["diff"], m, d) for m in LISTS for d in DIRECTIONS))[1:] == ("CLIP", "i2t")),
    "lists: five, without the fine-tuned Qwen2.5-VL embedder": LISTS == ["CLIP", "BLIP2", "FLAVA", "SigLIP2", "Qwen3"],
    "learning curve: the contrast is significant in every draw of 100 queries": all(LCV[d]["100"]["contrast_significant_share"] == 100 for d in DIRECTIONS),
    "learning curve: the share of significant draws at 50 queries rounds to the same value in both directions":
        len({half_up(LCV[d]["50"]["contrast_significant_share"]) for d in DIRECTIONS}) == 1,
    "no LLM assessor and no COCO pool in the paper (the BLIP head's COCO checkpoint is a model name)": not any(w in text for w in ("o3", "COCO pool", "assessor", "SugarCrepe image", "Cohen")),
}
bad = 0
for name, s in expected.items():
    ok = s in text
    bad += not ok
    print(("ok      " if ok else "MISSING ") + f"{name}: {s}")
for name, ok in checks.items():
    bad += not ok
    print(("ok      " if ok else "FAILED  ") + name)
print(f"\n{len(expected) + len(checks) - bad} of {len(expected) + len(checks)} checks pass")
sys.exit(1 if bad else 0)
