"""Download ALL shards of AICD Bench and take an even random sample from each shard,
so the sample covers the whole split (unlike streaming the first rows).
Usage: python sample_shards.py --tasks T1 T2 T3 --train 60000 --val 5000 --test 20000 --out data"""
import argparse, os
import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "AICD-bench/AICD-Bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", type=int, default=20000)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    sizes = {"train": a.train, "validation": a.val, "test": a.test}
    for t in a.tasks:
        for split, n in sizes.items():
            shards = sorted(f for f in files if f.startswith(f"{t}/{split}") and f.endswith(".parquet"))
            if not shards:
                raise SystemExit(f"No files found for {t}/{split}. Files in repo: {files[:30]}")
            per = -(-n // len(shards))
            parts = []
            for f in shards:
                p = hf_hub_download(REPO, f, repo_type="dataset")
                df = pq.read_table(p, columns=["code", "label"]).to_pandas()
                parts.append(df.sample(min(per, len(df)), random_state=0))
                print(f"  {f}: took {len(parts[-1])} of {len(df)}", flush=True)
            out = pd.concat(parts).sample(frac=1, random_state=0).reset_index(drop=True)
            out.to_parquet(f"{a.out}/{t}_{split}.parquet")
            print(f"{t} {split}: {out.shape}  labels={out['label'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
