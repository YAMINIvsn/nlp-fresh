"""
Day 2, Step 1: retrain FinBERT on the v2 benchmark and SAVE RAW LOGITS.

- Trains on data/v2/train.json (text only; FinBERT never sees the fields).
- Best epoch is chosen on data/v2/val.json ONLY.
- The test set is predicted once, after training, and never used for any
  selection or tuning.
- Raw logits for val and test are saved to eval/results/ so calibration
  (temperature scaling), risk scoring and all ablations can be run
  without retraining.

Run: python scripts/finetune_finbert_v2.py   (CPU is fine, ~5-15 min)
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from data_v2 import ID2LABEL, LABEL2ID, LABELS, load_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "finbert_v2"
RESULTS_DIR = ROOT / "eval" / "results"
MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128
SEED = 42


class ClauseDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.enc = tokenizer([r["text"] for r in records], truncation=True,
                             padding="max_length", max_length=MAX_LENGTH)
        self.labels = [LABEL2ID[r["label"]] for r in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


def save_logits(name, records, logits):
    out = [{"id": r["id"], "case_type": r["case_type"], "label": r["label"],
            "label_id": LABEL2ID[r["label"]], "logits": [float(x) for x in l]}
           for r, l in zip(records, logits)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"v2_logits_{name}.json", "w") as f:
        json.dump(out, f)
    print(f"saved {len(out)} logits -> eval/results/v2_logits_{name}.json")


def main():
    set_seed(SEED)
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True)

    ds_train, ds_val, ds_test = (ClauseDataset(x, tok) for x in (train, val, test))

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to="none",
        use_cpu=True,
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_train,
                      eval_dataset=ds_val, compute_metrics=compute_metrics)
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))

    val_logits = trainer.predict(ds_val).predictions
    test_logits = trainer.predict(ds_test).predictions  # test touched once, here
    save_logits("val", val, val_logits)
    save_logits("test", test, test_logits)

    y = [LABEL2ID[r["label"]] for r in test]
    p = np.argmax(test_logits, axis=1)
    print("\n=== FinBERT-only (v2) on TEST, uncalibrated ===")
    print(classification_report(y, p, target_names=LABELS, zero_division=0))


if __name__ == "__main__":
    main()