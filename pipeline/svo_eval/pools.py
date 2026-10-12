import json
from functools import lru_cache
import numpy as np
from .paths import MODELS, DUAL_ENCODERS, DIRECTIONS, pool_file, pool_key, RESULTS
from .metrics import fleiss_kappa

LABELS = ["correct", "subject incorrect", "verb incorrect", "object incorrect"]


def _vote(x):
    return 1 if str(x).strip().lower() == "correct" else -1


@lru_cache(maxsize=None)
def load_pool(model, direction):
    """Judged top-10 list of one model in one direction: {query: [judgment, ...]}."""
    data = json.load(open(pool_file(model, direction), encoding="utf-8"))
    llm = {}
    llm_file = pool_file(model, direction, judged_by_llm=True)
    if llm_file.exists():                                  # the LLM assessor's labels, where that file exists
        llm = json.load(open(llm_file, encoding="utf-8"))
    key = pool_key(model, direction)
    out = {}
    for q, v in data.items():
        qq = q if direction == "t2i" else int(q)
        cands = v[key]
        human = v["human evaluation"]
        votes2 = v.get("human evaluation 2")
        votes3 = v.get("human evaluation 3")
        gpt = v.get("GPT evaluation") or [None] * len(cands)
        o3 = (llm.get(q, {}).get("o3 evaluation") if llm else None) or [None] * len(cands)
        items = []
        for r, cand in enumerate(cands):
            votes = [_vote(human[r])]
            if votes2 is not None and votes3 is not None:
                votes += [int(votes2[r]), int(votes3[r])]
            items.append({"cand": int(cand) if direction == "t2i" else str(cand), "rank": r + 1,
                          "expert": str(human[r]).strip().lower(), "votes": votes,
                          "gpt4o": (str(gpt[r]).strip().lower() if gpt[r] not in (None, "") else None),
                          "o3": (str(o3[r]).strip().lower() if o3[r] not in (None, "") else None)})
        out[qq] = items
    return out


def is_correct(j):
    v = j["votes"]
    if len(v) == 3:
        return sum(1 for x in v if x == 1) >= 2
    return v[0] == 1


def error_type(j):
    e = j["expert"]
    return e.replace(" incorrect", "") if e in LABELS[1:] else None


def query_ids(direction):
    return list(load_pool("CLIP", direction))


OUT_OF_COLLECTION = {"t2i": 0, "i2t": 0}


@lru_cache(maxsize=1)
def build_fixed_pool():
    """Union of all judged candidates per query: {direction: {query: {cand: entry}}}.
    Judged candidates outside the 13,285-image / 10,978-caption collection (5 in the Qwen3 T->I pool,
    mined over the unfiltered image set) are dropped and counted in OUT_OF_COLLECTION."""
    from .collection import load_collection
    coll = load_collection()
    caps, imgs = set(coll.captions), set(coll.images)
    fp = {}
    for d in DIRECTIONS:
        fp[d] = {}
        for m in MODELS:
            for q, items in load_pool(m, d).items():
                bucket = fp[d].setdefault(q, {})
                for j in items:
                    if (d == "t2i" and j["cand"] not in imgs) or (d == "i2t" and j["cand"] not in caps):
                        OUT_OF_COLLECTION[d] += 1
                        continue
                    e = bucket.setdefault(j["cand"], {"correct": None, "error_type": None,
                                                       "retrieved_by": {}, "labels": {}})
                    e["retrieved_by"][m] = j["rank"]
                    c = is_correct(j)
                    if e["correct"] is None:
                        e["correct"] = c
                        e["error_type"] = error_type(j) if not c else None
                    elif e["correct"] != c:
                        e["labels"]["disagreement"] = True   # same pair judged in two files; keep first, flag
                    e["labels"][m] = {"expert": j["expert"], "votes": j["votes"], "gpt4o": j["gpt4o"], "o3": j["o3"]}
    return fp


def fleiss_kappa_i2t():
    rows = []
    for m in DUAL_ENCODERS:
        for items in load_pool(m, "i2t").values():
            for j in items:
                v = j["votes"]
                rows.append([v.count(-1), v.count(0), v.count(1)])
    return fleiss_kappa(np.array(rows))


def save_fixed_pool():
    fp = build_fixed_pool()
    ser = {d: {str(q): {str(c): e for c, e in v.items()} for q, v in fp[d].items()} for d in fp}
    json.dump(ser, open(RESULTS / "fixed_pool.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return fp
