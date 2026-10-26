"""Zero-shot Fast-DetectGPT (Bao et al., 2024), analytic conditional-probability-curvature criterion.
Uses one code LM as both sampling and scoring model (same tokenizer required)."""
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def discrepancy(codes, model_name="Qwen/Qwen2.5-Coder-1.5B", max_len=512, batch=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    out = []
    for i in range(0, len(codes), batch):
        enc = tok(codes[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        logits = lm(**enc).logits[:, :-1].float()
        labels = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].float()
        lp = torch.log_softmax(logits, -1)
        probs = lp.exp()
        ll = (lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
        mean = ((probs * lp).sum(-1) * mask).sum(-1)
        var = ((probs * lp ** 2).sum(-1) - (probs * lp).sum(-1) ** 2)
        var = (var * mask).sum(-1)
        out.extend(((ll - mean) / var.clamp_min(1e-8).sqrt()).cpu().tolist())
    return np.array(out)


def fit_threshold(scores, y_bin):
    """y_bin: 1 = machine (anything not human). Pick threshold maximizing Macro-F1 on validation."""
    best_t, best_f = 0.0, -1
    for t in np.quantile(scores, np.linspace(0.01, 0.99, 99)):
        f = f1_score(y_bin, (scores > t).astype(int), average="macro")
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f
