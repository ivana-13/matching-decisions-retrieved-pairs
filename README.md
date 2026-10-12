# Supplementary repository

Judged retrieval lists, stored matching decisions and analysis code for an anonymous submission on image–text
matching decisions as re-rankers of retrieved candidates.

The paper uses SVO-Probes in two roles: as the image–text matching (ITM) benchmark and as a retrieval collection.
For 100 caption queries and 100 image queries, the top-10 lists of five embedding models (CLIP ViT-L/14, the BLIP-2
Q-Former, FLAVA, SigLIP2 and Qwen3-VL-Embed) were labelled by human annotators. Seven matching decisions (the BLIP,
BLIP-2 and FLAVA matching heads, the Qwen2.5-VL prompt, the Qwen3-VL-Embed yes/no construction, Qwen3-VL-Reranker and
Grounding CLIP) are scored on these judged pairs and used as re-rankers and as filters of the lists.

## What is here

```
svo_probes/     filtered_svo.csv (the benchmark file) and the ten judged top-10 lists (five models, two directions)
pipeline/
  svo_eval/     analysis package (collection, judged lists, metrics, matching heads)
  scripts/      50: every analysis of the paper; 51: the tables; 52: every number phrase of the paper checked
                against the result file; 55: the figures; 03b and 07: how the matching heads were run;
                vm/: the scripts that produced the other decisions on the GPU server
  results/      the stored decisions of every pair, the benchmark summaries, the similarity rows of the judged
                lists, and short_matching.json, the file behind every number of the paper
  paper/        the sections, tables and figures of the paper (tables and figures are generated)
docs/
  annotation_instructions.md   who labelled what, the instructions the annotators received, how labels are stored
  matching_decisions.md        checkpoints, prompts and thresholds of the seven decisions; measures; statistics
```

### Judged lists

`svo_probes/<MODEL>_selected_samples_images_retrieval_evaluation.json` holds the caption queries (text-to-image) and
`..._text_retrieval_evaluation.json` the image queries (image-to-text) of one model: for each of the 100 queries, the
model's ten candidates in its original order (`<MODEL> retrieved images` / `<MODEL> retrieved captions`; the CLIP
image-query file carries the key `FLAVA retrieved captions`, a naming accident that `svo_eval/paths.py` resolves) and
the label of every candidate (`human evaluation`: `correct`, `subject incorrect`, `verb incorrect` or
`object incorrect`). The image-query lists of CLIP, BLIP-2, FLAVA and SigLIP2 were labelled by three annotators;
there `human evaluation 2` and `human evaluation 3` hold the votes of the second and third annotator (1 yes, -1 no,
0 cannot tell) and the label is the majority vote. Image ids are the benchmark's `pos_image_id` / `neg_image_id`;
images are not redistributed. The ten lists hold 10,000 entries, 3,318 distinct caption–image pairs for the caption
queries and 3,383 for the image queries; a pair retrieved by several models takes the label of the first list in
the order CLIP, BLIP-2, FLAVA, SigLIP2, Qwen3-VL-Embed.

### Stored decisions

One row per judged pair (`direction`, `query`, `candidate`, `image_id`, `caption`) in `pipeline/results/`:

| File | Decision | Score column | Default decision |
|---|---|---|---|
| `itm_pool_preds_BLIP.csv` | BLIP head | `p_match` | match if > 0.5 |
| `itm_pool_preds_BLIP2.csv` | BLIP-2 head | `p_match` | match if > 0.5 |
| `itm_pool_preds_FLAVA.csv` | FLAVA head | `p_match` | match if > 0.5 |
| `Qwen25p_pool_svo.csv` | Qwen2.5-VL prompt | `p_yes` | match if > 0.5 |
| `Qwen25_itm_svo_pool.csv` | Qwen2.5-VL prompt, decoded answer only | `model_pred` (yes / no) | the answer |
| `itm_pool_preds_Qwen3_full.csv` | Qwen3-VL-Embed yes/no | `sim_yes` − `sim_no` | match if > 0 |
| `Qwen3R_pool_svo.csv` | Qwen3-VL-Reranker | `score` | match if > 0.5 |
| `GCLIP_pool_svo.csv` | Grounding CLIP | `score` | none (score only) |

On the benchmark: `itm_benchmark_preds_{BLIP,BLIP2,FLAVA}.csv` (every unique caption–image pair, `p_match`),
`Qwen3R_bench_svo.csv` (per triplet, `score_pos` and `score_neg`), `GCLIP_bench_svo.csv` (per pair, `score`), and
`itm_benchmark.json` with the ITM accuracy, TPR and TNR of the BLIP, BLIP-2, FLAVA, Qwen2.5-VL and Qwen3-VL-Embed
decisions. `list_similarities.json` holds, for every judged list, the similarity of its ten candidates with the query
under the embedding model that retrieved them (the embeddings themselves are not included); the interpolation
variant of the paper reads it.

## Reproducing the paper's numbers without any model

Run from `pipeline/` with that folder on `PYTHONPATH`:

```
cd pipeline
pip install -r requirements.txt
python scripts/50_short_matching_analyses.py          # every analysis -> results/short_matching.json (a few minutes)
python scripts/51_short_matching_tables.py            # Tables 1 and 2 -> paper/tables/
python scripts/55_short_figures.py                    # Fig. 2 -> paper/figures/ (Fig. 1 needs the images)
python scripts/52_check_short_matching_numbers.py     # every number phrase of paper/sections/ against the result file
```

`scripts/50` also contains analyses that are not in the paper (re-ordering at depth 2, 3 and 5; the benchmark outcome
of a query against its re-ranking gain; the label check with single annotators) and the hooks for a second judged
pool of the companion study, whose files are not part of this repository.

## Recomputing the decisions

1. Obtain the SVO-Probes images (the URLs are in the benchmark file) and save them as `svo_probes/images/<image_id>.jpg`.
2. Matching heads: `scripts/03b_itm_benchmark.py` (benchmark pairs) and `scripts/07_itm_pools.py` (judged pairs) with
   `svo_eval/itm_heads.py`; BLIP and BLIP-2 need LAVIS in a separate environment (`scripts/README_lavis.md`).
3. The other decisions ran on a GPU server with the scripts in `scripts/vm/` (their paths are those of that server):
   `qwen25_prompt.py` (the decoded answer of the Qwen2.5-VL prompt, mode `svo_pool_itm`) and
   `qwen25_yes_probability.py` (its yes-probability), `qwen3_yes_no.py` (the Qwen3-VL-Embed construction on the judged
   pairs, mode `svo_pool`) and `qwen3_yes_no_benchmark.py` (the same on the benchmark), `qwen3_reranker.py`
   (Qwen3-VL-Reranker on the judged pairs and on the benchmark), `gclip_export.py` and `gclip_svo.py` (Grounding CLIP).
   Some of these scripts also carry the modes of the companion study's second collection, which this paper does not use.

`docs/matching_decisions.md` gives the checkpoints, prompts and thresholds.

## Data sources and terms

SVO-Probes (Hendricks and Nematzadeh, 2021) remains under the terms of its providers; `svo_probes/filtered_svo.csv`
is a row subset of the benchmark file. The licence of the labels and code in this repository will be stated with the
camera-ready version.
