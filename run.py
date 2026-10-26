#!/usr/bin/env python
"""AICD Bench runner.  Subcommands: classical | encoder | fastdetect | llm | all-classical

Data spec for --train/--val/--test: local file (.parquet/.jsonl/.csv) or hf:<repo>:<config|->:<split>
Example:
  python run.py encoder --task 1 --model modernbert \
     --train hf:AICD-bench/AICD-Bench:task1:train --val hf:AICD-bench/AICD-Bench:task1:validation \
     --test hf:AICD-bench/AICD-Bench:task1:test
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from aicd.data import load_any, prepare, LabelSpace, report, macro_f1

TASK_NAMES = {1: ["Human", "AI"],
              3: ["Human", "AI", "Hybrid", "Adversarial"]}


def common(p):
    p.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--train"); p.add_argument("--val"); p.add_argument("--test", required=True)
    p.add_argument("--code_col"); p.add_argument("--label_col")
    p.add_argument("--max_train", type=int); p.add_argument("--max_eval", type=int)
    p.add_argument("--out", default="results")
    p.add_argument("--label_names", help="comma-separated names in label-id order (needed if labels are ints for llm)")


def load(a, need_train=True):
    tr = prepare(load_any(a.train), a.code_col, a.label_col, a.max_train) if (a.train and need_train) else None
    va = prepare(load_any(a.val), a.code_col, a.label_col, a.max_eval) if a.val else None
    te = prepare(load_any(a.test), a.code_col, a.label_col, a.max_eval)
    ref = tr if tr is not None else te
    names = a.label_names.split(",") if a.label_names else None
    ls = LabelSpace(pd.concat([d["label"] for d in (tr, va, te) if d is not None]), names)
    return tr, va, te, ls


def finish(a, tag, te, ls, pred):
    y = ls.encode(te["label"])
    res = report(te, y, np.asarray(pred), ls, f"{a.out}/task{a.task}_{tag}.json")
    return res


def cmd_classical(a):
    from aicd.classical import run_classical, dummy_baselines
    tr, va, te, ls = load(a)
    ytr = ls.encode(tr["label"])
    for k, v in dummy_baselines(ytr, len(te), ls.n).items():
        print(f"== {k}"); finish(a, k, te, ls, v)
    models = ["lr", "svm", "catboost"] if a.model == "all" else [a.model]
    feats = ["tfidf", "ast", "both"] if a.features == "all" else [a.features]
    for m in models:
        for f in feats:
            print(f"== {m} / {f}")
            pred = run_classical(tr, te, ytr, ls.n, m, f)
            finish(a, f"{m}_{f}", te, ls, pred)


def cmd_encoder(a):
    from aicd.encoder import run_encoder
    tr, va, te, ls = load(a)
    pred = run_encoder(a.model, tr, ls.encode(tr["label"]), va, ls.encode(va["label"]), te, ls.n,
                       f"{a.out}/ckpt_{a.model}_task{a.task}", epochs=a.epochs, batch=a.batch,
                       grad_accum=a.grad_accum, lr=a.lr)
    finish(a, f"enc_{a.model}", te, ls, pred)


def cmd_fastdetect(a):
    from aicd.fast_detectgpt import discrepancy, fit_threshold
    from sklearn.metrics import f1_score
    tr, va, te, ls = load(a, need_train=False)
    # binary target: 1 = machine. Task 3 is binarized to human vs. everything else.
    human_idx = [i for i, c in enumerate(ls.classes) if str(c).lower() in ("human", "human-written", "0")]
    hid = human_idx[0] if human_idx else 0
    to_bin = lambda d: (ls.encode(d["label"]) != hid).astype(int)
    sv = discrepancy(va["code"].tolist(), a.lm); t, f = fit_threshold(sv, to_bin(va))
    print(f"threshold={t:.4f} val macroF1={f:.4f}")
    st = discrepancy(te["code"].tolist(), a.lm)
    pred = (st > t).astype(int); y = to_bin(te)
    from aicd.data import LabelSpace as LS
    res = report(te, y, pred, LS([0, 1], ["Human", "Machine"]), f"{a.out}/task{a.task}_fastdetectgpt.json")


def cmd_llm(a):
    from aicd.llm_zeroshot import make_backend, classify
    _, _, te, ls = load(a, need_train=False)
    names = ls.names
    if a.task in TASK_NAMES and not a.label_names and all(str(c).isdigit() for c in ls.classes):
        names = TASK_NAMES[a.task]  # assumes id order Human, AI, [Hybrid, Adversarial]; override with --label_names
    # 5% sample stratified by label (+ language/domain if present)
    keys = [c for c in ["label", "language", "domain"] if c in te.columns]
    te = te.groupby(keys, group_keys=False).apply(lambda g: g.sample(max(1, int(len(g) * a.frac)), random_state=0))
    te = te.reset_index(drop=True)
    backend = make_backend(a.backend, a.llm)
    fallback = 0
    pred = classify(te["code"].tolist(), a.task, a.mode, names, fallback, backend)
    finish(a, f"llm_{a.backend}_{a.mode}", te, ls, pred)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classical"); common(p)
    p.add_argument("--model", default="all", choices=["lr", "svm", "catboost", "all"])
    p.add_argument("--features", default="all", choices=["tfidf", "ast", "both", "all"])
    p.set_defaults(fn=cmd_classical)
    p = sub.add_parser("encoder"); common(p)
    p.add_argument("--model", required=True, choices=["codebert", "codet5p", "unixcoder", "modernbert", "roberta", "deberta"])
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4); p.add_argument("--lr", type=float, default=2e-5)
    p.set_defaults(fn=cmd_encoder)
    p = sub.add_parser("fastdetect"); common(p)
    p.add_argument("--lm", default="Qwen/Qwen2.5-Coder-1.5B"); p.set_defaults(fn=cmd_fastdetect)
    p = sub.add_parser("llm"); common(p)
    p.add_argument("--backend", default="gemini", choices=["gemini", "hf"])
    p.add_argument("--llm", default="gemini-2.5-flash"); p.add_argument("--mode", default="zs", choices=["zs", "cot"])
    p.add_argument("--frac", type=float, default=0.05); p.set_defaults(fn=cmd_llm)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
