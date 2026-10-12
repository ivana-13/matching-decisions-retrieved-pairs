from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd
from .paths import COLLECTION_CSV


@dataclass
class Collection:
    df: pd.DataFrame
    captions: list
    images: list
    cap_index: dict
    img_index: dict
    cap2imgs: dict
    img2caps: dict
    cap2trip: dict
    img2trips: dict
    cap2negimgs: dict
    img2negcaps: dict
    cap2negtype: dict = field(default_factory=dict)

    def gold_images(self, cap):
        return self.cap2imgs.get(cap, set())

    def gold_captions(self, img):
        return self.img2caps.get(int(img), set())

    def evaluable_queries(self, direction):
        if direction == "t2i":
            return list(self.captions)
        return [i for i in self.images if self.img2caps.get(i)]

    def gold(self, direction, query):
        return self.cap2imgs[query] if direction == "t2i" else self.img2caps[int(query)]


def _negtype(subj, verb):
    if subj:
        return "subject"
    if verb:
        return "verb"
    return "object"


@lru_cache(maxsize=1)
def load_collection() -> Collection:
    df = pd.read_csv(COLLECTION_CSV)
    df["pos_image_id"] = df["pos_image_id"].astype(int)
    df["neg_image_id"] = df["neg_image_id"].astype(int)
    captions, cap_index = [], {}
    cap2imgs, img2caps, cap2trip, img2trips = {}, {}, {}, {}
    cap2negimgs, cap2negtype = {}, {}
    for r in df.itertuples(index=False):
        s, p, n = r.sentence, int(r.pos_image_id), int(r.neg_image_id)
        if s not in cap_index:
            cap_index[s] = len(captions)
            captions.append(s)
        cap2imgs.setdefault(s, set()).add(p)
        img2caps.setdefault(p, set()).add(s)
        trip = tuple(x.strip() for x in str(r.pos_triplet).split(","))
        cap2trip[s] = trip
        img2trips.setdefault(p, set()).add(trip)
        cap2negimgs.setdefault(s, set()).add(n)
        cap2negtype.setdefault(s, {})[n] = _negtype(bool(r.subj_neg), bool(r.verb_neg))
    images = sorted(set(df["pos_image_id"]) | set(df["neg_image_id"]))
    img_index = {i: k for k, i in enumerate(images)}
    img2negcaps = {}
    for r in df.itertuples(index=False):
        p, n = int(r.pos_image_id), int(r.neg_image_id)
        for c in img2caps.get(n, set()):
            if p not in cap2imgs.get(c, set()):
                img2negcaps.setdefault(p, set()).add(c)
    return Collection(df, captions, images, cap_index, img_index, cap2imgs, img2caps,
                      cap2trip, img2trips, cap2negimgs, img2negcaps, cap2negtype)
