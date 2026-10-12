"""COCO val2017 as a second retrieval collection, with SugarCrepe as the handcrafted probe.

Images: the 5,000 val2017 images. Captions: the val2017 caption annotations (about five per image; identical strings
are merged). Strict relevance: a caption is relevant to the image(s) it annotates. Probe: SugarCrepe pairs
(image, caption, negative caption) in seven subsets; the negative is a textual edit of the caption, so the probe is
image-to-text only.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from .paths import DATA

COCO_DIR = DATA / "coco"
IMAGES = COCO_DIR / "val2017"
CAPTIONS_JSON = COCO_DIR / "annotations" / "captions_val2017.json"
SUGARCREPE_DIR = COCO_DIR / "sugarcrepe"
SUBSETS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]


@dataclass
class CocoCollection:
    captions: list          # unique caption strings
    images: list            # sorted image ids
    cap_index: dict
    img_index: dict
    cap2imgs: dict          # caption -> set of image ids
    img2caps: dict          # image id -> set of captions
    probe: list             # dicts: image_id, caption, negative_caption, subset

    def image_path(self, image_id):
        return IMAGES / f"{int(image_id):012d}.jpg"

    def evaluable_queries(self, direction):
        return list(self.captions) if direction == "t2i" else [i for i in self.images if self.img2caps.get(i)]

    def gold(self, direction, query):
        return self.cap2imgs[query] if direction == "t2i" else self.img2caps[int(query)]


def _clean(s):
    return " ".join(str(s).strip().split())


@lru_cache(maxsize=1)
def load_coco() -> CocoCollection:
    ann = json.load(open(CAPTIONS_JSON, encoding="utf-8"))
    images = sorted(im["id"] for im in ann["images"])
    cap2imgs, img2caps = {}, {}
    for a in ann["annotations"]:
        c = _clean(a["caption"])
        if not c:
            continue
        cap2imgs.setdefault(c, set()).add(int(a["image_id"]))
        img2caps.setdefault(int(a["image_id"]), set()).add(c)
    captions = sorted(cap2imgs)
    probe = []
    for sub in SUBSETS:
        p = SUGARCREPE_DIR / f"{sub}.json"
        if not p.exists():
            continue
        for e in json.load(open(p, encoding="utf-8")).values():
            probe.append({"image_id": int(Path(e["filename"]).stem), "caption": _clean(e["caption"]),
                          "negative_caption": _clean(e["negative_caption"]), "subset": sub})
    return CocoCollection(captions, images, {c: i for i, c in enumerate(captions)}, {i: k for k, i in enumerate(images)},
                          cap2imgs, img2caps, probe)
