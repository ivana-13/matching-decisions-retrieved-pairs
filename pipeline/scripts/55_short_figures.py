"""Figures of the paper on matching decisions (paper/figures):

fig_example.pdf  one caption query of SVO-Probes: its benchmark pair (positive image, handcrafted negative) and
                 selected candidates of one judged top-10 list, with the human label of every image and the match
                 probability of the BLIP-2 and FLAVA heads; the item each head puts first (over the ten) is marked.
fig_rerank.pdf   change in judged Success@1 of every list when it is re-ordered by the score of each decision,
                 caption queries and image queries side by side (results/short_matching.json, scripts/50).
Usage: python scripts/55_short_figures.py"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageOps
from svo_eval.paths import RESULTS, ECIR, MODELS, DIRECTIONS, DISPLAY, IMAGES_DIR
from svo_eval.pools import load_pool, is_correct, error_type
from svo_eval.collection import load_collection

OUT = ECIR / "paper" / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.5})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GOOD, BAD = "#0ca30c", "#d03b3b"                 # status colours, always with a text label
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"   # diverging pair with a neutral midpoint
SHORT = {"BLIP": "BLIP", "BLIP2": "BLIP-2", "FLAVA": "FLAVA", "Qwen25": "Prompt", "Qwen25p": "Prompt", "Qwen3": "Yes/no", "Qwen3R": "Reranker",
         "GCLIP": "GCLIP"}
HEADS = [("BLIP2", "BLIP-2 head"), ("FLAVA", "FLAVA head")]
FIG_W = 4.8                                       # inches: the LNCS text width (12.2 cm)

# ---------------------------------------------------------------- example
EXAMPLE = {"model": "FLAVA", "query": "The man sits on the sofa.", "neg_image": 6684, "ranks": [1, 2, 3, 4, 5, 9]}


def head_probs(name):
    pool = pd.read_csv(RESULTS / f"itm_pool_preds_{name}.csv", dtype=str, keep_default_na=False)
    bench = pd.read_csv(RESULTS / f"itm_benchmark_preds_{name}.csv", dtype=str, keep_default_na=False)
    p = {("pool", r.query, str(r.candidate)): float(r.p_match) for r in pool.itertuples() if r.direction == "t2i" and r.p_match != ""}
    p.update({("bench", r.caption, str(r.image_id)): float(r.p_match) for r in bench.itertuples() if r.p_match != ""})
    return p


def thumb(image_id, w=300, h=200):
    im = Image.open(IMAGES_DIR / f"{int(image_id)}.jpg").convert("RGB")
    return ImageOps.fit(im, (w, h), Image.LANCZOS)


def example_figure():
    coll = load_collection()
    q, m, neg = EXAMPLE["query"], EXAMPLE["model"], EXAMPLE["neg_image"]
    pos = sorted(coll.gold_images(q))[0]
    items = {j["rank"]: j for j in load_pool(m, "t2i")[q]}
    probs = {k: head_probs(k) for k, _ in HEADS}
    first = {k: max(items.values(), key=lambda j: (probs[k].get(("pool", q, str(j["cand"])), -1), -j["rank"]))["rank"] for k, _ in HEADS}
    neg_type = coll.cap2negtype[q][int(neg)]
    cols = [("bench", pos, "positive", True, None, None), ("bench", neg, "negative", False, None, None)]
    prev = None
    for r in EXAMPLE["ranks"]:
        if prev is not None and r > prev + 1:
            cols.append(("gap", None, None, None, None, None))
        j = items[r]
        cols.append(("pool", j["cand"], f"rank {r}", is_correct(j), error_type(j), r))
        prev = r
    widths = [0.28 if c[0] == "gap" else 1 for c in cols]
    fig, axes = plt.subplots(1, len(cols), figsize=(FIG_W, 0.94),
                             gridspec_kw={"wspace": 0.10, "left": 0.115, "right": 0.99, "top": 0.73, "bottom": 0.405, "width_ratios": widths})
    for ax, (kind, img, label, ok, et, rank) in zip(axes, cols):
        ax.set_xticks([]); ax.set_yticks([])
        if kind == "gap":
            ax.axis("off")
            ax.text(0.5, 0.5, "…", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=MUTED)
            continue
        ax.imshow(thumb(img))
        for s in ax.spines.values():
            s.set_edgecolor(GOOD if ok else BAD); s.set_linewidth(1.2)
        verdict = "correct" if ok else (f"{et} error" if et else "incorrect")
        ax.set_title(label, fontsize=6.2, color=INK2, pad=1.5)
        ax.text(0.5, -0.08, ("✓ " if ok else "✗ ") + verdict, transform=ax.transAxes, ha="center", va="top", fontsize=5.6,
                color=GOOD if ok else BAD, fontweight="bold")
        for r_i, (k, _) in enumerate(HEADS):
            p = probs[k].get(("bench", q, str(img)) if kind == "bench" else ("pool", q, str(img)))
            txt = "--" if p is None else f"{p:.2f}"
            is_first = kind == "pool" and first[k] == rank
            ax.text(0.5, -0.36 - 0.27 * r_i, txt + (" ◀" if is_first else ""), transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.3, color=INK, fontweight="bold" if is_first else "normal")
    # row labels aligned with the probability rows, group headers and the divider
    p0 = axes[0].get_position()
    for r_i, (_, name) in enumerate(HEADS):
        y = p0.y0 + (-0.36 - 0.27 * r_i) * p0.height
        fig.text(0.005, y, name, fontsize=6.0, color=INK2, ha="left", va="top")
    fig.text(0.005, p0.y0 - 0.08 * p0.height, "human label", fontsize=6.0, color=INK2, ha="left", va="top")
    x_split = axes[1].get_position().x1 + (axes[2].get_position().x0 - axes[1].get_position().x1) / 2
    fig.add_artist(plt.Line2D([x_split, x_split], [0.07, 0.97], color=MUTED, lw=0.5, ls=(0, (2, 2))))
    xl = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    fig.text(xl, 0.99, "Benchmark pair", fontsize=6.4, color=INK, ha="center", va="top")
    fig.text(axes[2].get_position().x0, 0.99, f"Six of the ten images retrieved by {DISPLAY[m]} for the same caption", fontsize=6.4, color=INK, ha="left", va="top")
    fig.savefig(OUT / "fig_example.pdf")
    plt.close(fig)
    print("fig_example.pdf:", q, "| first by head over the ten:", first)


# ---------------------------------------------------------------- re-ranking heatmap
def rerank_figure():
    A = json.load(open(RESULTS / "short_matching.json"))
    S = A["svo"]
    decs = [k for k in A["decisions"] if not (k == "Qwen25" and "Qwen25p" in A["decisions"])]
    LISTS = A.get("lists", MODELS)
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    # per-list cells plus a last row with the mean over the lists (the Score columns of Table 2)
    vals = {d: np.array([[S["rerank"][m][d][k]["score"]["after"] - S["rerank"][m][d]["orig"] for k in decs] for m in LISTS]
                        + [[S["rerank_mean"][k][d]["score"]["all"]["diff"] for k in decs]]) for d in DIRECTIONS}
    pval = {d: [[S["rerank"][m][d][k]["score"]["p_holm"] for k in decs] for m in LISTS]
               + [[S["rerank_mean"][k][d]["score"]["all"]["p_holm"] for k in decs]] for d in DIRECTIONS}
    n_rows = len(LISTS) + 1
    vmax = 5 * np.ceil(max(np.abs(v).max() for v in vals.values()) / 5)
    # no colour bar: every cell carries its number; significance after Holm is shown in bold, so nothing is colour-alone
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.32), gridspec_kw={"wspace": 0.12, "left": 0.215, "right": 0.995, "top": 0.615, "bottom": 0.03,
                                                                        "width_ratios": [1, 1]})
    for i_ax, (ax, d, title) in enumerate(zip(axes, DIRECTIONS, ("Caption queries (T→I)", "Image queries (I→T)"))):
        v = vals[d]
        ax.imshow(v, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(decs))); ax.set_xticklabels([SHORT[k] for k in decs], fontsize=6, rotation=35, ha="left", rotation_mode="anchor")
        ax.xaxis.tick_top()
        ax.set_yticks(range(n_rows))
        if i_ax == 0:
            ax.set_yticklabels([f"{DISPLAY[m]}  {S['rerank'][m][d]['orig']:.0f}" for m in LISTS] + ["Mean"], fontsize=6)
        else:
            ax.set_yticklabels([f"{S['rerank'][m][d]['orig']:.0f}" for m in LISTS] + [""], fontsize=6)
        ax.tick_params(length=0, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(n_rows):
            for j, k in enumerate(decs):
                dark = abs(v[i, j]) > 0.55 * vmax
                ax.text(j, i, f"{v[i, j]:+.0f}", ha="center", va="center", fontsize=6.2, color="white" if dark else INK,
                        fontweight="bold" if pval[d][i][j] < 0.05 else "normal")
        for x in range(1, len(decs)):
            ax.axvline(x - 0.5, color="white", lw=1.2)
        for y in range(1, n_rows):
            ax.axhline(y - 0.5, color="white", lw=3.5 if y == len(LISTS) else 1.2)
    fig.text(0.005, 0.985, "Rows: top-10 lists of five embedding models (number: original Success@1, %); columns: the re-ordering matching decision.",
             fontsize=5.5, color=INK2, ha="left", va="top")
    for ax, title in zip(axes, ("Caption queries (T→I)", "Image queries (I→T)")):   # panel titles drawn by hand: set_title would fall outside the figure
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.925, title, fontsize=6.4, color=INK, ha="center", va="top")
    fig.savefig(OUT / "fig_rerank.pdf")
    plt.close(fig)
    print("fig_rerank.pdf: decisions", decs, "vmax", vmax)


if IMAGES_DIR.is_dir():
    example_figure()
else:
    print("fig_example.pdf skipped: the SVO-Probes images are not in", IMAGES_DIR)
rerank_figure()
