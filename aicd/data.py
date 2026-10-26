"""Data loading, label handling and Macro-F1 reporting for AICD Bench."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

CODE_COLS = ["code", "text", "content", "source_code", "snippet"]
LABEL_COLS = ["label", "labels", "target", "class"]
LANG_COLS = ["language", "lang", "programming_language"]
META_COLS = ["language", "lang", "domain", "source", "generator", "model", "split_type", "sub_source"]


def load_any(spec: str) -> pd.DataFrame:
    """spec = local .parquet/.jsonl/.json/.csv path  OR  hf:<repo>:<config or ->:<split>
    e.g. hf:AICD-bench/AICD-Bench:task1:train   (check the HF page for real config/split names)"""
    if spec.startswith("hf:"):
        from datasets import load_dataset
        _, repo, cfg, split = spec.split(":")
        return load_dataset(repo, None if cfg == "-" else cfg, split=split).to_pandas()
    ext = os.path.splitext(spec)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(spec)
    if ext in (".jsonl", ".json"):
        return pd.read_json(spec, lines=(ext == ".jsonl"))
    if ext == ".csv":
        return pd.read_csv(spec)
    raise ValueError(f"Unsupported file type: {spec}")


def _find(df, cands, what, required=True):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find {what} column; have {list(df.columns)}. Pass --code_col/--label_col.")
    return None


def prepare(df, code_col=None, label_col=None, max_rows=None, seed=0):
    code_col = code_col or _find(df, CODE_COLS, "code")
    label_col = label_col or _find(df, LABEL_COLS, "label")
    out = df.rename(columns={code_col: "code", label_col: "label"}).copy()
    if max_rows and len(out) > max_rows:
        out = out.sample(max_rows, random_state=seed).reset_index(drop=True)
    return out


class LabelSpace:
    """Maps raw labels (str or int) to contiguous ids. Optional human-readable names."""

    def __init__(self, raw_labels, names=None):
        self.classes = sorted(pd.unique(raw_labels), key=lambda x: str(x))
        self.idx = {c: i for i, c in enumerate(self.classes)}
        self.names = names or [str(c) for c in self.classes]

    def encode(self, raw):
        return np.array([self.idx.get(r, -1) for r in raw])

    @property
    def n(self):
        return len(self.classes)


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro"))


def report(df, y, p, label_space, out_json=None):
    """Overall Macro-F1 + breakdown by any metadata columns (language/domain/source/generator...)."""
    import json
    res = {"macro_f1": macro_f1(y, p), "n": int(len(y)), "breakdown": {}}
    for col in [c for c in META_COLS if c in df.columns]:
        sub = {}
        for g, idx in df.groupby(col).indices.items():
            if len(idx) >= 20:
                sub[str(g)] = {"macro_f1": macro_f1(y[idx], p[idx]), "n": int(len(idx))}
        res["breakdown"][col] = sub
    print(f"\nMacro-F1: {res['macro_f1']:.4f}  (n={res['n']})")
    print(classification_report(y, p, labels=list(range(label_space.n)), target_names=label_space.names, zero_division=0))
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        json.dump(res, open(out_json, "w"), indent=2)
    return res
