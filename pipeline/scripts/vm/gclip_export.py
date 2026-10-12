"""Export the image--caption pairs that the Grounding CLIP score (Vongala et al., CVPRW 2025) is computed on:
the judged SVO-Probes pool (both directions) and the benchmark's unique (caption, image) pairs, each with the
caption's subject, verb and object from the benchmark's triplet (the method parses captions with an LLM; SVO-Probes
provides the triplets, so no parsing is needed). Output: results/gclip_inputs.json, copied to the VM.
Usage: python scripts/vm/gclip_export.py"""
import json
from svo_eval.paths import RESULTS, MODELS, DIRECTIONS
from svo_eval.collection import load_collection

coll = load_collection()
fixed = json.load(open(RESULTS / "fixed_pool.json", encoding="utf-8"))


def triplet(caption):
    t = coll.cap2trip[caption]
    return {"subject": t[0], "verb": t[1], "object": t[2]}


pool = []
for d in DIRECTIONS:
    for q, cands in fixed[d].items():
        for c in cands:
            cap, img = (q, c) if d == "t2i" else (c, q)
            pool.append({"direction": d, "query": q, "candidate": c, "caption": cap, "image_id": int(img), **triplet(cap)})
bench_pairs = sorted({(r.sentence, int(r.pos_image_id)) for r in coll.df.itertuples()} | {(r.sentence, int(r.neg_image_id)) for r in coll.df.itertuples()},
                     key=lambda x: (x[1], x[0]))
bench = [{"caption": c, "image_id": i, **triplet(c)} for c, i in bench_pairs]
images = sorted({p["image_id"] for p in pool} | {p["image_id"] for p in bench})
out = {"pool": pool, "bench": bench, "n_images": len(images)}
json.dump(out, open(RESULTS / "gclip_inputs.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"pool pairs {len(pool)}, benchmark pairs {len(bench)}, images {len(images)}; pool images {len({p['image_id'] for p in pool})}")
