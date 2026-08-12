#!/usr/bin/env python3
"""Measures retrieval quality against a labelled query set.

The evaluation file is JSONL with one record per query::

    {"query": "What is deep learning?", "relevant_ids": ["a1b2c3d4e5f6a7b8"]}

``relevant_ids`` are chunk ids from data/vector_store/chunks.jsonl. As a
convenience, ``relevant_sources`` may be given instead to mark every chunk of a
source document as relevant.

Usage:
    python scripts/evaluate.py --qrels data/eval/qrels.jsonl --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import settings  # noqa: E402
from rag.evaluator import EvalExample, RetrievalEvaluator  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.reranker import NoOpReranker, load_reranker  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qrels", type=Path, default=settings.data_dir / "eval" / "qrels.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--compare", action="store_true", help="Also report BM25-only and dense-only baselines")
    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranking")
    return parser.parse_args()


def load_examples(path: Path, chunks) -> list[EvalExample]:
    if not path.exists():
        raise FileNotFoundError(f"No evaluation set at {path}")

    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk.id)

    examples: list[EvalExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        relevant = list(record.get("relevant_ids", []))
        for source in record.get("relevant_sources", []):
            relevant.extend(by_source.get(source, []))
        if not relevant:
            logger.warning("Query %r has no relevant chunks; skipping.", record.get("query"))
            continue
        examples.append(EvalExample(query=record["query"], relevant_ids=relevant))
    return examples


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    body = "  ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    return f"{name:<16} {body}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    reranker = NoOpReranker() if args.no_reranker else load_reranker(settings.reranker_model)
    pipeline = RAGPipeline.load(settings, reranker=reranker)
    retriever = pipeline.retriever

    examples = load_examples(args.qrels, retriever.chunks)
    if not examples:
        logger.error("No usable evaluation examples in %s", args.qrels)
        return 1
    logger.info("Evaluating %d queries over %d chunks", len(examples), len(retriever))

    evaluator = RetrievalEvaluator(k_values=(1, 3, 5, min(10, args.top_k)))

    runs: dict[str, list[list[str]]] = {
        "hybrid": [
            [result.chunk.id for result in pipeline.query(ex.query, top_k=args.top_k).results] for ex in examples
        ]
    }
    if args.compare:
        runs["bm25_only"] = [
            [retriever.chunks[i].id for i, _ in retriever.sparse_search(ex.query, args.top_k)] for ex in examples
        ]
        runs["dense_only"] = [
            [retriever.chunks[i].id for i, _ in retriever.dense_search(ex.query, args.top_k)] for ex in examples
        ]

    print()
    for name, retrieved in runs.items():
        print(format_metrics(name, evaluator.evaluate(retrieved, examples)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
