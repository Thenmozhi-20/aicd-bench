"""Data loading, label handling and Macro-F1 reporting for AICD Bench."""
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from .features import guess_language

CODE_COLS = ["code", "text", "content", "source_code", "snippet"]
LABEL_COLS = ["label", "labels", "target", "class"]
LANG_COLS = ["language", "lang", "programming_language"]
META_COLS = ["language", "lang", "domain", "source", "generator", "model", "split_type", "sub_source"]

SEEN_LANGUAGES = {"python", "java", "cpp", "c++"}
UNSEEN_LANGUAGES = {"c", "go", "golang", "php", "csharp", "c#", "javascript", "js", "rust"}


def guess_domain(code: str) -> str:
    """Proxy for algorithmic (seen domain) vs general software (unseen domain)."""
    algo_hints = re.compile(
        r"\bclass\s+Solution\b|\bdef\s+solve\b|\bint\s+main\b|#include\s*<iostream>|"
        r"cin\s*>>|cout\s*<<|\bScanner\b|sys\.stdin|input\(\)|ans\s*=\s*0|dp\s*=\s*\[|MOD\s*=\s*10"
    )
    return "seen (algorithmic)" if algo_hints.search(code) else "unseen (software/research)"


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
    
    # Enrich with language & domain metadata if absent to support paper breakdowns (Table 5 & Fig 4)
    if "language" not in out.columns:
        out["language"] = [guess_language(c) for c in out["code"]]
    if "domain" not in out.columns:
        out["domain"] = [guess_domain(c) for c in out["code"]]

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
    return float(f1_score(y, p, average="macro", zero_division=0))


def report(df, y, p, label_space, out_json=None):
    """Overall Macro-F1 + breakdown by metadata columns, row-normalized confusion matrix (Figs 1,2,7,9),
    and Task 1 four-way seen/unseen matrix (Table 5)."""
    import json
    res = {
        "macro_f1": macro_f1(y, p),
        "n": int(len(y)),
        "report": classification_report(
            y, p, labels=list(range(label_space.n)),
            target_names=label_space.names, zero_division=0, output_dict=True
        ),
        "confusion_row_normalized": confusion_matrix(
            y, p, labels=list(range(label_space.n)), normalize="true"
        ).tolist(),
        "breakdown": {}
    }

    # Metadata breakdowns (e.g. Figure 4 language-wise)
    for col in [c for c in META_COLS if c in df.columns]:
        sub = {}
        for g, idx in df.groupby(col).indices.items():
            if len(idx) >= 10:
                sub[str(g)] = {"macro_f1": macro_f1(y[idx], p[idx]), "n": int(len(idx))}
        res["breakdown"][col] = sub

    # Paper Table 5: Task 1 (Robust Binary Classification) four-way breakdown
    if "language" in df.columns and "domain" in df.columns:
        seen_lang_mask = df["language"].str.lower().isin(SEEN_LANGUAGES).to_numpy()
        seen_dom_mask = df["domain"].str.contains("seen", case=False).to_numpy()
        table5 = {}
        splits = {
            "Seen domain, seen language": seen_dom_mask & seen_lang_mask,
            "Seen domain, unseen language": seen_dom_mask & (~seen_lang_mask),
            "Unseen domain, seen language": (~seen_dom_mask) & seen_lang_mask,
            "Unseen domain, unseen language": (~seen_dom_mask) & (~seen_lang_mask)
        }
        for name, m in splits.items():
            if m.sum() >= 10:
                table5[name] = {"macro_f1": macro_f1(y[m], p[m]), "n": int(m.sum())}
        res["table5_breakdown"] = table5

    print(f"\nMacro-F1: {res['macro_f1']:.4f}  (n={res['n']})")
    print(classification_report(y, p, labels=list(range(label_space.n)), target_names=label_space.names, zero_division=0))
    if "table5_breakdown" in res and res["table5_breakdown"]:
        print("\n--- Task 1 Distribution Shift Breakdown (Paper Table 5) ---")
        for k, v in res["table5_breakdown"].items():
            print(f"  {k:32s}: Macro-F1 = {v['macro_f1']:.4f} (n={v['n']})")

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
    return res
