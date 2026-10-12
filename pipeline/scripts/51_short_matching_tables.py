"""Tables of the paper on matching decisions (SVO-Probes, human labels), from results/short_matching.json
(scripts/50). Written to paper/tables: tab_match_decisions.tex and tab_match_rerank.tex."""
import json
from svo_eval.paths import RESULTS, ECIR, MODELS, DIRECTIONS, DISPLAY

OUT = ECIR / "paper" / "tables"
A = json.load(open(RESULTS / "short_matching.json"))
# the prompt appears once: with its yes-probability when that run exists, otherwise with its answer only
DECISIONS = [k for k in A["decisions"] if not (k == "Qwen25" and "Qwen25p" in A["decisions"])]
NAME = {"BLIP": "BLIP head", "BLIP2": "BLIP-2 head", "FLAVA": "FLAVA head", "Qwen25": "Qwen2.5-VL prompt", "Qwen25p": "Qwen2.5-VL prompt",
        "Qwen3": "\\qemb{} yes/no", "Qwen3R": "Qwen3-VL-Reranker", "GCLIP": "Grounding CLIP$^{\\dagger}$"}
SHORT = {"BLIP": "BLIP", "BLIP2": "BLIP-2", "FLAVA": "FLAVA", "Qwen25": "Prompt", "Qwen25p": "Prompt", "Qwen3": "Yes/no", "Qwen3R": "Reranker",
         "GCLIP": "GCLIP"}
SCORE_ONLY = {"GCLIP"}                 # no default decision: ITM accuracy is replaced by the pairwise accuracy (dagger), filter cells by --
S = A["svo"]
nan = lambda x: x != x


def stars(p):
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")


def half_up(x):
    return int(x + 0.5 + 1e-9)


def signed(x, p=None):
    s = f"{x:+.1f}".replace("+", "$+$").replace("-", "$-$")
    return s + (stars(p) if p is not None else "")


# Table 1: the decisions on the benchmark's balanced pairs and on the judged pool
rows = []
for k in DECISIONS:
    sep = S["separation"][k]
    acc = A["bench_acc"]["svo"].get(k)
    if k in SCORE_ONLY:                      # pairwise accuracy (dagger), no threshold: no TPR/TNR, no default decision
        cells = [NAME[k], "--" if acc is None else f"{acc['pairwise']:.1f}", "--", "--"]
    else:
        cells = [NAME[k], f"{acc['overall']:.1f}", f"{half_up(acc['tpr'])}", f"{half_up(acc['tnr'])}"]
    cells += [f"{sep[d]['auc']:.2f}" for d in DIRECTIONS]
    cells.append(f"{S['rerank_mean']['within_list_auc'][k]['mean']:.2f}")
    if k in SCORE_ONLY:
        cells += ["--", "--", "--", "--", "--"]
    else:
        cells.append("/".join(f"{half_up(sep[d]['tpr'])}" for d in DIRECTIONS))
        cells.append("/".join(f"{half_up(sep[d]['tnr'])}" for d in DIRECTIONS))
        cells += [f"{S['errors'][k]['both'][t]['accept']:.0f}" for t in ("subject", "verb", "object")]
    rows.append(" & ".join(cells) + " \\\\")
first = " & ".join(["Original order", "", "", "", "", "", f"{S['rerank_mean']['within_list_auc']['first_stage']['mean']:.2f}", "", "", "", "", ""]) + " \\\\"
tab1 = "\n".join([
    "\\begin{tabular}{l ccc cc c cc ccc}", "\\toprule",
    "& \\multicolumn{3}{c}{Benchmark (\\%)} & \\multicolumn{2}{c}{Pool AUC} & List & \\multicolumn{2}{c}{Pool (\\%)} & \\multicolumn{3}{c}{Accepted errors (\\%)}\\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}\\cmidrule(lr){10-12}",
    "Decision & Acc. & TPR & TNR & \\ttoi{} & \\itot{} & AUC & TPR & TNR & Subj. & Verb & Obj.\\\\", "\\midrule",
    *rows, "\\midrule", first, "\\bottomrule", "\\end{tabular}", ""])
(OUT / "tab_match_decisions.tex").write_text(tab1, encoding="utf-8")

# Table 2: mean change of the judged Success@1 over the six lists per decision (rows), direction and variant
# (columns); the per-list values are in the heatmap figure (scripts/55)
Mn = S["rerank_mean"]
VARIANTS = [  # (column header, cell getter, available for score-only decisions)
    ("Score", lambda k, d: Mn[k][d]["score"]["all"], True),
    ("Filter", lambda k, d: Mn[k][d]["binary"]["all"], False),                   # the default decision as a filter
    ("Tuned", lambda k, d: S["tuned"][k][d]["s1"], True),                       # threshold cross-validated on the judged lists
    ("+Sim.", lambda k, d: S["interp"][k][d], True),                              # interpolated with the embedding similarity
    ("Strict", lambda k, d: Mn[k][d]["score"]["strict"], True),                 # score re-ordering, strict Success@1 (benchmark positives only)
]                                                                               # no LLM-assessor labels in this paper
nv = len(VARIANTS)
lines = ["\\begin{tabular}{l " + "c" * nv + " c " + "c" * nv + "}", "\\toprule",
         f"& \\multicolumn{{{nv}}}{{c}}{{\\ttoi{{}} (caption queries)}} & & \\multicolumn{{{nv}}}{{c}}{{\\itot{{}} (image queries)}}\\\\",
         f"\\cmidrule(lr){{2-{1 + nv}}}\\cmidrule(lr){{{3 + nv}-{2 + 2 * nv}}}",
         "Decision & " + " & ".join(v[0] for v in VARIANTS) + " & & " + " & ".join(v[0] for v in VARIANTS) + "\\\\", "\\midrule"]
for k in DECISIONS:
    cells = [NAME[k]]
    for d in DIRECTIONS:
        for _, get, for_score_only in VARIANTS:
            cells.append("--" if (k in SCORE_ONLY and not for_score_only) else signed(get(k, d)["diff"], get(k, d)["p_holm"]))
        if d == DIRECTIONS[0]:
            cells.append("")
    lines.append(" & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]
(OUT / "tab_match_rerank.tex").write_text("\n".join(lines), encoding="utf-8")
print(tab1)
print("\n".join(lines))
