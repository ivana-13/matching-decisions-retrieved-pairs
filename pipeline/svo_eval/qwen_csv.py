"""Readers for the Qwen prediction files produced on the VMs.

Qwen2.5: zero-shot Qwen2.5-VL-7B-Instruct (no adapter) answered an A/B prompt (pairwise) and a yes/no prompt (ITM)
over the 36,841 rows of the unfiltered benchmark file `svo_probes.csv` on the GPU server (copied as
data/qwen25_ft/svo_probes_vm.csv). Each result row carries the row index of that file, so the answers are aligned
to the 33,685 triplets of our collection by (sentence, positive image, negative image). Rows outside the collection
are dropped; rows inside it for which the run produced no answer are excluded and counted, never treated as correct
or as wrong. Qwen3-VL-Embedding decisions cover the 33,685 filtered triplets directly.
"""
import numpy as np
import pandas as pd
from .paths import DATA, COLLECTION_CSV

TYPE_COLS = ["subj_neg", "verb_neg", "obj_neg"]
_KEY = ["sentence", "pos_image_id", "neg_image_id"]


def _answered_bool(series):
    """True/False (bool or text) -> 1/0; empty or unrecognised -> NaN."""
    return series.map(lambda x: {"true": 1.0, "false": 0.0}.get(str(x).strip().lower(), np.nan))


def _yes_no(series, expected):
    s = series.astype(object)
    ok = s.notna().values
    ans = s.fillna("").astype(str).str.strip().str.lower()
    return (ans == expected).values, ok


def _align_to_collection(df, vm_csv=None):
    """Attach the VM file's triplet key to each result row and keep the rows of the local collection."""
    vm = pd.read_csv(vm_csv or DATA / "qwen25_ft" / "svo_probes_vm.csv")
    local = pd.read_csv(COLLECTION_CSV)
    vm["pos_image_id"] = vm["pos_image_id"].astype(int)
    vm["neg_image_id"] = vm["neg_image_id"].astype(int)
    local["pos_image_id"] = local["pos_image_id"].astype(int)
    local["neg_image_id"] = local["neg_image_id"].astype(int)
    keyed = df.merge(vm[_KEY].reset_index().rename(columns={"index": "vm_row"}), left_on="index", right_on="vm_row", how="left")
    in_local = keyed.merge(local[_KEY].drop_duplicates(), on=_KEY, how="inner")
    return in_local, int(len(df)), int(len(local))


def load_qwen25_pairwise(path=None, vm_csv=None):
    df = pd.read_csv(path or DATA / "qwen_itm" / "svo_probes_results_pairwise.csv")
    df, n_rows, n_local = _align_to_collection(df, vm_csv)
    c = _answered_bool(df["correct"])
    ok = c.notna().values
    types = df[TYPE_COLS].values.astype(bool)
    return {"correct": c.values[ok].astype(bool), "types": types[ok], "n_rows": n_rows, "n_in_collection": int(len(df)),
            "n_collection": n_local, "n_unanswered": int((~ok).sum())}


def load_qwen25_itm(path=None, vm_csv=None):
    df = pd.read_csv(path or DATA / "qwen_itm" / "svo_probes_results.csv")
    df, n_rows, n_local = _align_to_collection(df, vm_csv)
    pos_corr, pos_ok = _yes_no(df["pos_pred"], "yes")
    neg_corr, neg_ok = _yes_no(df["neg_pred"], "no")
    types = df[TYPE_COLS].values.astype(bool)
    return {"correct": np.concatenate([pos_corr[pos_ok], neg_corr[neg_ok]]),
            "types": np.concatenate([types[pos_ok], types[neg_ok]]),
            "is_positive": np.concatenate([np.ones(int(pos_ok.sum()), bool), np.zeros(int(neg_ok.sum()), bool)]),
            "n_rows": n_rows, "n_in_collection": int(len(df)), "n_collection": n_local,
            "n_unanswered": int((~pos_ok).sum() + (~neg_ok).sum())}


def load_qwen3_itm(path=None):
    df = pd.read_csv(path or DATA / "qwen3_embed" / "svo_probes_results_pairwise_qwen3_embed.csv")
    pos = _answered_bool(df["correct_pos"])
    neg = _answered_bool(df["correct_neg"])
    pos_ok, neg_ok = pos.notna().values, neg.notna().values
    types = df[TYPE_COLS].values.astype(bool)
    return {"correct": np.concatenate([pos.values[pos_ok].astype(bool), neg.values[neg_ok].astype(bool)]),
            "types": np.concatenate([types[pos_ok], types[neg_ok]]),
            "is_positive": np.concatenate([np.ones(int(pos_ok.sum()), bool), np.zeros(int(neg_ok.sum()), bool)]),
            "n_rows": int(len(df)), "n_unanswered": int((~pos_ok).sum() + (~neg_ok).sum())}
