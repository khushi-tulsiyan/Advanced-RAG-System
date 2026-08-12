"""Evaluation metrics.

Retrieval metrics (recall@k, precision@k, MRR, nDCG, hit rate) are computed
with numpy and always available — these are the metrics that actually diagnose
a retrieval system. Generation metrics (ROUGE, BERTScore) need optional
packages and are skipped with a clear message when those are missing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalExample:
    """One labelled query: the chunk ids (or texts) that count as relevant."""

    query: str
    relevant_ids: Sequence[str]


class RetrievalEvaluator:
    """Standard ranking metrics over retrieved id lists."""

    def __init__(self, k_values: Sequence[int] = (1, 3, 5, 10)) -> None:
        if not k_values or any(k <= 0 for k in k_values):
            raise ValueError("k_values must be positive integers")
        self.k_values = tuple(sorted(k_values))

    def evaluate(self, retrieved: Sequence[Sequence[str]], examples: Sequence[EvalExample]) -> Dict[str, float]:
        """Averages metrics over all queries.

        ``retrieved[i]`` is the ranked list of ids returned for ``examples[i]``.
        """
        if len(retrieved) != len(examples):
            raise ValueError(f"Got {len(retrieved)} result lists for {len(examples)} examples")
        if not examples:
            return {}

        totals: Dict[str, float] = {}
        for ranked_ids, example in zip(retrieved, examples):
            relevant = set(example.relevant_ids)
            for key, value in self._per_query(list(ranked_ids), relevant).items():
                totals[key] = totals.get(key, 0.0) + value

        return {key: value / len(examples) for key, value in sorted(totals.items())}

    def _per_query(self, ranked_ids: List[str], relevant: set[str]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not relevant:
            return {f"recall@{k}": 0.0 for k in self.k_values}

        for k in self.k_values:
            top = ranked_ids[:k]
            hits = sum(1 for doc_id in top if doc_id in relevant)
            metrics[f"recall@{k}"] = hits / len(relevant)
            metrics[f"precision@{k}"] = hits / k
            metrics[f"hit_rate@{k}"] = 1.0 if hits else 0.0
            metrics[f"ndcg@{k}"] = _ndcg(top, relevant, k)

        metrics["mrr"] = next(
            (1.0 / rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in relevant),
            0.0,
        )
        return metrics


def _ndcg(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


class GenerationEvaluator:
    """ROUGE and BERTScore wrappers that degrade gracefully when uninstalled."""

    def __init__(self, rouge_types: Sequence[str] = ("rouge1", "rouge2", "rougeL")) -> None:
        self.rouge_types = tuple(rouge_types)
        self._rouge = None
        try:
            from rouge_score import rouge_scorer

            self._rouge = rouge_scorer.RougeScorer(list(self.rouge_types), use_stemmer=True)
        except ImportError:
            logger.warning("rouge-score is not installed; ROUGE metrics will be skipped.")

    def evaluate_rouge(self, references: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
        _check_pairs(references, predictions)
        if self._rouge is None:
            return {}
        if not references:
            return {rouge_type: 0.0 for rouge_type in self.rouge_types}

        totals = {rouge_type: 0.0 for rouge_type in self.rouge_types}
        for reference, prediction in zip(references, predictions):
            scores = self._rouge.score(reference, prediction)
            for rouge_type in totals:
                totals[rouge_type] += scores[rouge_type].fmeasure
        return {key: value / len(references) for key, value in totals.items()}

    def evaluate_bert_score(self, references: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
        _check_pairs(references, predictions)
        if not references:
            return {}
        try:
            from bert_score import score as bert_score
        except ImportError:
            logger.warning("bert-score is not installed; BERTScore metrics will be skipped.")
            return {}

        precision, recall, f1 = bert_score(list(predictions), list(references), lang="en", rescale_with_baseline=True)
        return {
            "bert_precision": precision.mean().item(),
            "bert_recall": recall.mean().item(),
            "bert_f1": f1.mean().item(),
        }

    def evaluate(self, references: Sequence[str], predictions: Sequence[str]) -> Dict[str, Dict[str, float]]:
        return {
            "rouge": self.evaluate_rouge(references, predictions),
            "bert_score": self.evaluate_bert_score(references, predictions),
        }


def _check_pairs(references: Sequence[str], predictions: Sequence[str]) -> None:
    if len(references) != len(predictions):
        raise ValueError(f"Got {len(predictions)} predictions for {len(references)} references")


# Backwards-compatible alias for the original class name.
Evaluator = GenerationEvaluator
