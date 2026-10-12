from pathlib import Path

ECIR = Path(__file__).resolve().parents[1]          # the folder that holds svo_eval/, scripts/, results/
# ROOT holds the SVO-Probes files (filtered_svo.csv, images/, embeddings) and the judged pool files: the folder
# "svo_probes" next to this one when it exists (layout of the released repository), else the parent folder.
ROOT = ECIR.parent / "svo_probes" if (ECIR.parent / "svo_probes").is_dir() else ECIR.parent
DATA = ECIR / "data"
RESULTS = ECIR / "results"
EMB = RESULTS / "emb"
FIGURES = ECIR / "figures"
for _p in (RESULTS, EMB, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

COLLECTION_CSV = ROOT / "filtered_svo.csv"
IMAGES_DIR = ROOT / "images"

MODELS = ["CLIP", "BLIP2", "FLAVA", "SigLIP2", "Qwen25", "Qwen3"]
DUAL_ENCODERS = ["CLIP", "BLIP2", "FLAVA", "SigLIP2"]
DISPLAY = {"CLIP": "CLIP", "BLIP2": "BLIP-2", "FLAVA": "FLAVA", "SigLIP2": "SigLIP2",
           "Qwen25": "Qwen2.5-VL-FT", "Qwen3": "Qwen3-VL-Embed"}
DIRECTIONS = ["t2i", "i2t"]      # t2i: caption query -> images; i2t: image query -> captions
DIRECTION_LABEL = {"t2i": "T$\\rightarrow$I", "i2t": "I$\\rightarrow$T"}

_LIST_FILES = {
    "CLIP":    {"t2i": ROOT / "CLIP_image_retrieval_results.json",    "i2t": ROOT / "CLIP_text_retrieval_results.json"},
    "BLIP2":   {"t2i": ROOT / "BLIP2_image_retrieval_results.json",   "i2t": ROOT / "BLIP2_text_retrieval_results.json"},
    "FLAVA":   {"t2i": ROOT / "FLAVA_image_retrieval_results.json",   "i2t": ROOT / "FLAVA_text_retrieval_results.json"},
    "SigLIP2": {"t2i": ROOT / "SigLIP2_image_retrieval_results.json", "i2t": ROOT / "SigLIP2_text_retrieval_results.json"},
    "Qwen25":  {"t2i": DATA / "qwen25_ft" / "InfoNCE_image_retrieval_results.json",
                "i2t": DATA / "qwen25_ft" / "InfoNCE_text_retrieval_results.json"},
    "Qwen3":   {"t2i": DATA / "qwen3_embed" / "qwen3_embed_image_retrieval_results.json",
                "i2t": DATA / "qwen3_embed" / "qwen_3_embed_text_retrieval_results.json"},
}
_POOL_PREFIX = {"CLIP": "CLIP", "BLIP2": "BLIP2", "FLAVA": "FLAVA", "SigLIP2": "SigLIP2",
                "Qwen25": "Qwen2.5", "Qwen3": "Qwen3"}


def list_file(model: str, direction: str) -> Path:
    return _LIST_FILES[model][direction]


def pool_file(model: str, direction: str, judged_by_llm: bool = False) -> Path:
    kind = "images" if direction == "t2i" else "text"
    suffix = "_o3_experiment" if judged_by_llm else ""
    return ROOT / f"{_POOL_PREFIX[model]}_selected_samples_{kind}_retrieval_evaluation{suffix}.json"


def pool_key(model: str, direction: str) -> str:
    """Name of the list field inside a pool file. The CLIP i2t file was saved with a FLAVA key."""
    what = "images" if direction == "t2i" else "captions"
    prefix = _POOL_PREFIX[model]
    if model == "CLIP" and direction == "i2t":
        prefix = "FLAVA"
    return f"{prefix} retrieved {what}"
