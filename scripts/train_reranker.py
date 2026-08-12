#!/usr/bin/env python3
"""Fine-tunes the cross-encoder reranker on labelled query/document pairs.

Expects JSONL or JSON with records of the form::

    {"query": "...", "document": "...", "label": 1}

``label`` is 1 for relevant and 0 for irrelevant. Fixes over the original
version: it now creates the validation split that ``eval_strategy="epoch"``
requires (previously training crashed at the first evaluation), passes labels
through tokenisation, uses dynamic padding instead of padding everything to 512
tokens, and uses the current Transformers argument names.

Usage:
    python scripts/train_reranker.py --data data/training/reranker_data.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import REPO_ROOT, settings  # noqa: E402

logger = logging.getLogger("train_reranker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=settings.data_dir / "training" / "reranker_data.jsonl")
    parser.add_argument("--model", type=str, default=settings.reranker_model)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "reranker")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--eval-split", type=float, default=0.1, help="Validation fraction (0 disables eval)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import numpy as np
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        logger.error("Training dependencies missing (%s). Install with: pip install '.[train]'", exc)
        return 1

    if not args.data.exists():
        logger.error("Training data not found at %s", args.data)
        return 1

    dataset = load_dataset("json", data_files={"train": str(args.data)})["train"]
    required = {"query", "document", "label"}
    missing = required - set(dataset.column_names)
    if missing:
        logger.error("Training data is missing required field(s): %s", ", ".join(sorted(missing)))
        return 1
    logger.info("Loaded %d labelled pairs from %s", len(dataset), args.data)

    do_eval = args.eval_split > 0 and len(dataset) >= 10
    if do_eval:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = dataset, None
        if args.eval_split > 0:
            logger.warning("Dataset too small to split; training without evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(examples):
        # Truncate the document side only, so the query is never cut off.
        encoded = tokenizer(
            examples["query"],
            examples["document"],
            truncation="only_second",
            max_length=args.max_length,
        )
        encoded["labels"] = [float(label) for label in examples["label"]]
        return encoded

    columns_to_drop = [c for c in train_dataset.column_names if c not in ("labels",)]
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=columns_to_drop)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=columns_to_drop)

    # ms-marco cross-encoders are single-logit regression models.
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = (np.asarray(logits).squeeze(-1) > 0).astype(int)
        return {"accuracy": float((predictions == np.asarray(labels).astype(int)).mean())}

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch" if do_eval else "no",
        save_strategy="epoch",
        load_best_model_at_end=do_eval,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_total_limit=2,
        seed=args.seed,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if do_eval else None,
    )

    logger.info("Training reranker %s", args.model)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Saved fine-tuned reranker to %s", args.output_dir)
    logger.info("Use it with: export RAG_RERANKER_MODEL=%s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
