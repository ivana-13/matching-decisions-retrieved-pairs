# Matching decisions, measures and statistics

## The seven decisions

A matching decision is a model that scores an image–caption pair together with the threshold that turns the score
into a match / no-match answer. The same code scores the benchmark pairs (`scripts/03b_itm_benchmark.py`) and the
judged pairs (`scripts/07_itm_pools.py`, `scripts/vm/`), so the two numbers of a decision are comparable.

- **BLIP head**: LAVIS `blip_image_text_matching` / `base` (ViT-B, 384 px, the COCO retrieval checkpoint whose ITM
  head re-ranks in the published BLIP setting), `match_head="itm"`, softmax over the two ITM logits; the score is the
  match probability, the threshold 0.5.
- **BLIP-2 head**: LAVIS `blip2_image_text_matching` / `pretrain`, the ITM logits averaged over the query tokens,
  softmax over the two logits; score and threshold as for BLIP.
- **FLAVA head**: `facebook/flava-full`, the ITM head of `FlavaForPreTraining`, fp32, softmax over the two logits;
  score and threshold as for BLIP. For the three heads, images that cannot be read or are shorter than 32 px on a
  side get no decision (10 benchmark pairs, 2 judged pairs) and are excluded.
- **Qwen2.5-VL prompt**: Qwen2.5-VL-7B-Instruct, 4-bit, the prompt "You are an image caption expert. Does this image
  match the caption: '[caption]'? Answer yes only if everything from the caption - subject, verb and object - matches
  the image, otherwise answer no." The score is the probability mass of the "yes" tokens against the "no" tokens at
  the first generated position, p_yes = P(yes) / (P(yes) + P(no)); the threshold 0.5 reproduces the greedy answer up
  to the rare answers that start with another token. `Qwen25_itm_svo_pool.csv` holds the decoded answers of an earlier
  run with the same prompt and `Qwen25p_pool_svo.csv` the probabilities.
- **Qwen3-VL-Embed yes/no**: the joint image–caption embedding under the instruction "Represent the given image and
  caption for binary classification to determine whether they match or not" is compared with the embeddings of "Yes"
  and "No" under "Represent the given answer for binary classification"; the score is the similarity with "Yes"
  minus the similarity with "No", the threshold 0.
- **Qwen3-VL-Reranker**: Qwen3-VL-Reranker-2B with the official implementation and its default instruction; the
  score is the sigmoid of the yes-minus-no logit at the last position, the threshold 0.5. A caption query is the
  re-ranker's query and the image its document; an image query the reverse. On the benchmark every triplet is scored
  as (caption, positive image) and (caption, negative image).
- **Grounding CLIP** (Vongala, Srivastava and Kosecka, CVPRW 2025), re-implemented from the official repository with
  Grounding DINO Swin-T (`IDEA-Research/grounding-dino-tiny`) and CLIP ViT-L/14 (`openai/clip-vit-large-patch14`):
  the subject and the object of the caption's triplet are grounded with the prompt "{subject}. {object}." (box
  threshold 0.35, text threshold 0.25); the image blacked out outside the subject box, likewise for the object, and
  the overlay of both regions are embedded, each weighted by its CLIP similarity with its phrase (subject, object,
  verb), added to the whole-image embedding, and the score is the cosine of the sum with the caption embedding. A
  missing region falls back to the whole image. The score has no threshold, so its benchmark number is the pairwise
  accuracy (the positive image scored above the negative) over the 33,672 scorable triplets.

Per pair, the heads cost a 0.2–1.1B-parameter forward pass, the two Qwen3 decisions a 2B one and the prompt a 7B one.

## Benchmark numbers

- **ITM accuracy**: the share of the benchmark's pairs (each triplet's positive and its negative pair, 67,370 pairs)
  that the decision answers correctly at its default threshold; TPR on the positive pairs, TNR on the negative pairs.
- **Pairwise accuracy** (Grounding CLIP only): the share of triplets in which the positive image scores above the
  negative.

## Measures on the judged lists

- **Pool AUC**: area under the ROC curve of the decision's score over all distinct judged pairs of a direction, each
  with its human label; **TPR** and **TNR**: the share of judged-correct pairs the default decision accepts and of
  judged-incorrect pairs it rejects.
- **Within-list AUC**: the AUC of the score inside one top-10 list, over the lists that contain both correct and
  incorrect candidates (860 of the 1,000 lists), averaged; for the original order the ranks play the role of the
  score.
- **Judged Success@1**: the first candidate of a list is judged correct, whether or not it is the benchmark positive;
  **strict Success@1**: the first candidate is a benchmark positive of the query. Both are means over the 100 queries
  of a direction, reported in percent; changes are in points.
- **Re-ranker** (Score): the ten candidates re-ordered by the decision's score, ties kept in the original order.
- **Filter**: the candidates the default decision accepts moved in front of those it rejects, the original order
  kept inside each group.
- **Tuned filter**: the same filter with the threshold that maximises the judged Success@1 of the five lists of the
  training queries, chosen in ten random folds over the queries (seed 0) and applied to the held-out queries.
- **+Sim.**: re-ordering by (1 − w) · z(similarity) + w · z(score), where z standardises inside the list, the similarity
  is that of the embedding model that produced the list (`results/list_similarities.json`), and w in {0, 0.1, ..., 1}
  is tuned in the same folds as the threshold.
- **Accepted errors**: among the judged-incorrect pairs with a subject, verb or object error, the share the default
  decision accepts, per error type, both directions pooled.
- Judged MRR and nDCG@10 are computed with the same labels and reported in `short_matching.json`.

## Statistics

- Every change of the judged Success@1 is tested with a two-sided paired randomisation (sign-flip) test on the
  per-query differences, 100,000 permutations, seed 0, with Holm's correction over the ten list-and-direction cells
  of a decision for the per-list changes (Fig. 2) and over the fourteen decision-and-direction cells for the mean
  changes (Table 2).
- The contrast between the BLIP-2 and FLAVA heads is the per-query difference of their mean gains over the five
  lists, tested the same way.
- Intervals in `short_matching.json` are 95% percentile intervals from 1,000 bootstrap resamples over queries.
- The learning curve draws 500 random subsets of 25, 50, 75 and 100 queries per direction and reports the share of
  draws in which the contrast is significant (10,000 sign flips, p < 0.05).
- Spearman's rank correlation between ITM accuracy and the pool AUC, the within-list AUC and the mean gain is taken
  over the six thresholded decisions.
