# AICD Bench – reference implementation

Implements the evaluation side of *AICD Bench: A Challenging Benchmark for AI-Generated Code Detection* (EACL 2026):

| Paper section | Where |
|---|---|
| 3 Tasks (robust binary / family attribution / fine-grained) | `--task 1/2/3` |
| 5.1 Encoders (CodeBERT, CodeT5+, UniXcoder, ModernBERT, RoBERTa, DeBERTa-v3) | `run.py encoder` → `aicd/encoder.py` |
| 5.1 Classical (LR / SVM / CatBoost × TF-IDF / AST / both) | `run.py classical` → `aicd/features.py`, `aicd/classical.py` |
| 5.1 Fast-DetectGPT (zero-shot) | `run.py fastdetect` |
| 5.2.4 Zero-shot + CoT LLM (Gemini-2.5-Flash, Gemma3-27B), prompts from App. C.1 | `run.py llm` |
| Macro-F1 + per-language/domain/source breakdowns | `aicd/data.py::report` |

**Not included:** the data-generation pipeline (prompting 77 LLMs, Droid filtering, MinHash dedup). The released dataset
already contains the result: https://huggingface.co/AICD-bench

## Setup
    pip install -r requirements.txt

## Data
Point `--train/--val/--test` at either local files (`.parquet/.jsonl/.csv`) or `hf:<repo>:<config|->:<split>`.
Check the Hugging Face page for the real config/split names, e.g.
`hf:AICD-bench/AICD-Bench:task1:train`. Columns are auto-detected (`code`/`text`/`content`, `label`);
override with `--code_col/--label_col`. If columns `language`, `domain`, `source`, `generator` exist, results are
broken down by them (this is how the seen/unseen language & domain analysis of Table 5 is obtained).

## Run
    # offline smoke test
    python make_toy_data.py
    python run.py classical --task 1 --train toy/train.parquet --val toy/val.parquet --test toy/test.parquet --model lr --features tfidf

    # paper-style baselines
    python run.py classical --task 1 --train data/T1_train.parquet --test data/T1_test.parquet --model all --features all --max_train 60000
    python run.py classical --task 1 --train data/T1_train.parquet --test data/T1_test.parquet --model svm --features tfidf --explain  # App. D.1 token analysis
    python run.py encoder   --task 3 --model modernbert --train data/T3_train.parquet --val data/T3_validation.parquet --test data/T3_test.parquet   # 3 epochs, eff. batch 64, 512 tokens
    python run.py fastdetect --task 1 --val data/T1_validation.parquet --test data/T1_test.parquet
    GEMINI_API_KEY=... python run.py llm --task 1 --test data/T1_test.parquet --backend gemini --mode cot
    python run.py llm --task 1 --test data/T1_test.parquet --backend hf --llm google/gemma-3-27b-it --mode zs

Results are written to `results/task{N}_{method}.json` (including Macro-F1, row-normalized confusion matrix matching Figs 1,2,7,9, per-language breakdown matching Fig 4, and Table 5 Seen/Unseen distribution shift matrix).

## Notes / caveats
* Full-scale training (500K–900K samples) needs a GPU; use `--max_train` to subsample classical models (SVD/AST on 500K files is slow).
* Fast-DetectGPT here uses a single code LM (Qwen2.5-Coder-1.5B) as sampler and scorer, with a threshold tuned on validation;
  the paper doesn't specify its backbone, so numbers will differ.
* For `llm` with integer labels, pass `--label_names` in label-id order (e.g. `Human,AI,Hybrid,Adversarial`). For Task 2 pass the 12 family names.
* Expect paper-like trends (strong in-domain, collapse out-of-domain), not exact numbers.
