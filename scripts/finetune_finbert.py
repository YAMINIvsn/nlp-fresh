"""
Baseline 2: LLM-Only (Fine-tuned FinBERT)

Fine-tunes ProsusAI/finbert (a BERT model pretrained on financial text)
for 3-class KYC/AML compliance classification: Compliant / Non-Compliant
/ Ambiguous.

CPU-friendly by design: small batch size, few epochs — appropriate given
the small (51-example) training set. This will run in a few minutes on
CPU; no GPU required.

Run with: python scripts/finetune_finbert.py
"""

import json
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "finbert_kyc"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["Compliant", "Non-Compliant", "Ambiguous"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

MAX_LENGTH = 128  # clause texts are short; 128 tokens is plenty


class ClauseDataset(Dataset):
    """Wraps tokenized clause text + label into a PyTorch Dataset for Trainer."""

    def __init__(self, texts: list, labels: list, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_split(name: str) -> tuple:
    """Load a train/val/test JSON file and return (texts, label_ids)."""
    path = DATA_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [LABEL2ID[item["label"]] for item in data]
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # FinBERT's original head is for sentiment (3 classes too, but different labels)
    )

    print("Loading dataset splits...")
    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    train_dataset = ClauseDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClauseDataset(val_texts, val_labels, tokenizer)
    test_dataset = ClauseDataset(test_texts, test_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=8,              # small dataset -> more epochs is fine, watch for overfitting via val loss
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=5,
        report_to="none",                # disable wandb/etc. prompts
        use_cpu=True,                    # force CPU explicitly since no GPU is available (renamed from no_cuda in transformers 5.x)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning (CPU) — this should take a few minutes given the small dataset...\n")
    trainer.train()

    print("\nEvaluating on held-out test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test set results: {test_results}")

    # Save the fine-tuned model for later use in the escalation pipeline
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"\nModel saved to {MODEL_OUT_DIR}")

    # Save results for the final comparison table
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "baseline2_finbert_only.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_DIR / 'baseline2_finbert_only.json'}")


if __name__ == "__main__":
    main()
