"""Tiny synthetic dataset for smoke-testing the pipeline offline."""
import pandas as pd, numpy as np
rng = np.random.default_rng(0)
H = ["def f(a,b):\n    return a+b\nli=[1,2]\nprint(f(*li))", "n=int(input())\nnums=list(map(int,input().split()))\nprint(sum(nums))"]
M = ['def add_numbers(first_number, second_number):\n    """Add two numbers."""\n    result = first_number + second_number\n    return result\n',
     'def compute_sum(numbers):\n    """Compute sum."""\n    answer = 0\n    for index in range(len(numbers)):\n        answer += numbers[index]\n    return answer\n']
def mk(n):
    rows = []
    for _ in range(n):
        y = int(rng.integers(0, 2))
        base = (M if y else H)[rng.integers(0, 2)]
        rows.append({"code": base + f"\n# {rng.integers(1000)}", "label": y, "language": "python", "domain": "algorithmic"})
    return pd.DataFrame(rows)
import os; os.makedirs("toy", exist_ok=True)
for s, n in [("train", 300), ("val", 60), ("test", 100)]:
    mk(n).to_parquet(f"toy/{s}.parquet")
