"""Fine-tune encoder classifiers (CodeBERT, CodeT5+, UniXcoder, ModernBERT, RoBERTa, DeBERTa-v3).
Paper setup: 3 epochs, effective batch 64, 512-token window."""
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODELS = {
    "codebert": "microsoft/codebert-base",
    "unixcoder": "microsoft/unixcoder-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "codet5p": "Salesforce/codet5p-110m-embedding",
}


class CodeT5pClassifier(nn.Module):
    """CodeT5+ embedding encoder + linear head."""

    def __init__(self, name, n_classes):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=True)
        dim = getattr(self.enc.config, "embed_dim", getattr(self.enc.config, "hidden_size", 768))
        self.head = nn.Linear(dim, n_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0, :]
        elif isinstance(out, torch.Tensor) and out.dim() == 3:
            emb = out[:, 0, :]
        else:
            emb = out
        logits = self.head(emb)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def _ds(df, y, tok, max_len):
    ds = Dataset.from_dict({"text": df["code"].tolist(), "labels": [int(v) for v in y]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])


def run_encoder(model_key, train, y_train, val, y_val, test, n_classes, out_dir,
                epochs=3, batch=16, grad_accum=4, lr=2e-5, max_len=512):
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model_key == "codet5p":
        model = CodeT5pClassifier(name, n_classes)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_classes)
    tr, va = _ds(train, y_train, tok, max_len), _ds(val, y_val, tok, max_len)
    te = _ds(test, np.zeros(len(test), dtype=int), tok, max_len)

    def metrics(p):
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

    _bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    _fp16 = torch.cuda.is_available() and not _bf16  # e.g. Colab T4
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2, gradient_accumulation_steps=grad_accum,  # 16*4 = 64
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.06, eval_strategy="epoch",
        save_strategy="no", logging_steps=100, report_to="none",
        bf16=_bf16, fp16=_fp16, save_safetensors=False)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    return trainer.predict(te).predictions.argmax(-1)
