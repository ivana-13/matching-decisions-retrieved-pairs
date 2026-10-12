"""Embeddings and scoring.

Scoring reproduces the runs that produced the annotated top-10 lists. The original notebook ranked CLIP, FLAVA and
SigLIP2 candidates by the raw dot product of the stored (unnormalised) embeddings; with that rule the stored files in
the repository root reproduce the judged lists exactly, whereas cosine similarity reorders them. BLIP-2 (LAVIS) and
Qwen3-VL-Embedding embeddings are unit-norm, and the Qwen2.5-VL-FT run normalised before ranking, so those three use
cosine similarity, which reproduces their lists too.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .paths import ROOT, DATA

NORMALISE = {"CLIP": False, "FLAVA": False, "SigLIP2": False, "BLIP2": True, "Qwen25": True, "Qwen3": True}


@dataclass
class Embeddings:
    cap: np.ndarray
    img: np.ndarray
    captions: np.ndarray
    image_ids: np.ndarray
    cap_index: dict
    img_index: dict


def _norm(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def _location(model):
    if model == "Qwen25":
        d = DATA / "qwen25_ft"
        return {"cap": d / "caption_embeddings.npy", "img": d / "image_embeddings.npy",
                "captions": d / "captions.npy", "image_ids": d / "image_id.npy"}
    if model == "Qwen3":
        d = DATA / "qwen3_embed"
        return {"cap": d / "caption_embeddings.npy", "img": d / "image_embeddings.npy",
                "captions": d / "captions.npy", "image_ids": d / "image_id.npy"}
    # CLIP, BLIP2, FLAVA, SigLIP2: the stored files of the original runs in the repository root
    return {"cap": ROOT / f"{model}_caption_embeddings.npy", "img": ROOT / f"{model}_image_embeddings.npy",
            "captions": ROOT / f"{model}_captions.npy", "image_ids": ROOT / f"{model}_image_id.npy"}


@lru_cache(maxsize=None)
def load_embeddings(model) -> Embeddings:
    loc = _location(model)
    cap = np.asarray(np.load(loc["cap"]), dtype=np.float32)
    img = np.asarray(np.load(loc["img"]), dtype=np.float32)
    if NORMALISE[model]:
        cap, img = _norm(cap), _norm(img)
    captions = np.load(loc["captions"], allow_pickle=True).astype(str)
    image_ids = np.load(loc["image_ids"], allow_pickle=True).astype(int)
    assert cap.shape[0] == len(captions) and img.shape[0] == len(image_ids), model
    return Embeddings(cap, img, captions, image_ids,
                      {c: i for i, c in enumerate(captions)}, {int(i): k for k, i in enumerate(image_ids)})


class Scorer:
    """Scores are dot products of the loaded embeddings: cosine where NORMALISE is set, raw otherwise."""

    def __init__(self, model):
        self.model = model
        self.e = load_embeddings(model)

    def all_scores(self, direction, query):
        e = self.e
        if direction == "t2i":
            return e.img @ e.cap[e.cap_index[query]]
        return e.cap @ e.img[e.img_index[int(query)]]

    def score_t2i(self, cap, image_ids):
        e = self.e
        return e.img[[e.img_index[int(i)] for i in image_ids]] @ e.cap[e.cap_index[cap]]

    def score_i2t(self, img, captions):
        e = self.e
        return e.cap[[e.cap_index[c] for c in captions]] @ e.img[e.img_index[int(img)]]

    def score(self, direction, query, cands):
        return self.score_t2i(query, cands) if direction == "t2i" else self.score_i2t(query, cands)

    def index_of(self, direction, cand):
        return self.e.img_index[int(cand)] if direction == "t2i" else self.e.cap_index[cand]

    def topk(self, query, direction, k):
        s = self.all_scores(direction, query)
        idx = np.argpartition(-s, k)[:k]
        idx = idx[np.argsort(-s[idx])]
        if direction == "t2i":
            return [int(self.e.image_ids[i]) for i in idx]
        return [str(self.e.captions[i]) for i in idx]

    def rank_of(self, query, direction, cand):
        s = self.all_scores(direction, query)
        j = self.index_of(direction, cand)
        return int((s > s[j]).sum()) + 1

    def candidates(self, direction):
        return self.e.image_ids if direction == "t2i" else self.e.captions
