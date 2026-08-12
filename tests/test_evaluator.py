from __future__ import annotations

import pytest

from rag.evaluator import EvalExample, GenerationEvaluator, RetrievalEvaluator


@pytest.fixture
def evaluator():
    return RetrievalEvaluator(k_values=(1, 3))


def test_perfect_ranking_scores_one(evaluator):
    examples = [EvalExample("q1", ["a"]), EvalExample("q2", ["b"])]
    metrics = evaluator.evaluate([["a", "x", "y"], ["b", "x", "y"]], examples)

    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg@3"] == pytest.approx(1.0)


def test_missed_ranking_scores_zero(evaluator):
    metrics = evaluator.evaluate([["x", "y", "z"]], [EvalExample("q", ["a"])])
    assert metrics["recall@3"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["hit_rate@3"] == 0.0


def test_mrr_reflects_the_rank_of_the_first_hit(evaluator):
    metrics = evaluator.evaluate([["x", "y", "a"]], [EvalExample("q", ["a"])])
    assert metrics["mrr"] == pytest.approx(1 / 3)


def test_recall_at_k_is_cutoff_sensitive(evaluator):
    metrics = evaluator.evaluate([["x", "y", "a"]], [EvalExample("q", ["a"])])
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == pytest.approx(1.0)


def test_precision_accounts_for_the_full_cutoff(evaluator):
    metrics = evaluator.evaluate([["a", "x", "y"]], [EvalExample("q", ["a"])])
    assert metrics["precision@1"] == pytest.approx(1.0)
    assert metrics["precision@3"] == pytest.approx(1 / 3)


def test_recall_with_multiple_relevant_documents(evaluator):
    metrics = evaluator.evaluate([["a", "x", "b"]], [EvalExample("q", ["a", "b", "c"])])
    assert metrics["recall@3"] == pytest.approx(2 / 3)


def test_ndcg_rewards_placing_hits_higher(evaluator):
    examples = [EvalExample("q", ["a"])]
    high = evaluator.evaluate([["a", "x", "y"]], examples)["ndcg@3"]
    low = evaluator.evaluate([["x", "y", "a"]], examples)["ndcg@3"]
    assert high > low


def test_metrics_are_averaged_across_queries(evaluator):
    examples = [EvalExample("q1", ["a"]), EvalExample("q2", ["b"])]
    metrics = evaluator.evaluate([["a"], ["z"]], examples)
    assert metrics["recall@1"] == pytest.approx(0.5)


def test_mismatched_input_lengths_are_rejected(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate([["a"]], [EvalExample("q1", ["a"]), EvalExample("q2", ["b"])])


def test_empty_evaluation_set_returns_no_metrics(evaluator):
    assert evaluator.evaluate([], []) == {}


def test_invalid_k_values_are_rejected():
    with pytest.raises(ValueError):
        RetrievalEvaluator(k_values=(0,))


def test_generation_evaluator_degrades_without_optional_packages():
    """ROUGE/BERTScore are optional; missing them must not raise."""
    evaluator = GenerationEvaluator()
    scores = evaluator.evaluate_rouge(["a reference sentence"], ["a predicted sentence"])
    assert isinstance(scores, dict)  # empty when rouge-score is absent


def test_generation_evaluator_rejects_mismatched_pairs():
    with pytest.raises(ValueError):
        GenerationEvaluator().evaluate_rouge(["one", "two"], ["only one"])
