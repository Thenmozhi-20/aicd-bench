import itertools, os, sys
import pandas as pd
from datasets import load_dataset

cfg = sys.argv[1] if len(sys.argv) > 1 else "T1"
sizes = {"train": 20000, "validation": 2000, "test": 5000}
os.makedirs("data", exist_ok=True)
for split, n in sizes.items():
    print(f"Downloading {cfg} {split} ({n} rows)...", flush=True)
    ds = load_dataset("AICD-bench/AICD-Bench", cfg, split=split, streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10000)
    df = pd.DataFrame(list(itertools.islice(ds, n)))
    df.to_parquet(f"data/{cfg}_{split}.parquet")
    print(df.shape, df["label"].value_counts().to_dict(), flush=True)
