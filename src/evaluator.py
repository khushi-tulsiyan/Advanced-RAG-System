from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import List, Dict

class Evaluator:
    def __init__(self):
        """Initializes the evaluation metrics: ROUGE and BERTScore."""
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    def evaluate_rouge(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Computes ROUGE scores between references and predictions."""
        scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        for ref, pred in zip(references, predictions):
            rouge_scores = self.rouge_scorer.score(ref, pred)
            for key in scores:
                scores[key] += rouge_scores[key].fmeasure
        
        # Average scores
        for key in scores:
            scores[key] /= len(references)
        
        return scores
    
    def evaluate_bert_score(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Computes BERTScore for similarity evaluation."""
        P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
        return {"bert_precision": P.mean().item(), "bert_recall": R.mean().item(), "bert_f1": F1.mean().item()}
    
    def evaluate(self, references: List[str], predictions: List[str]) -> Dict[str, Dict[str, float]]:
        """Runs both ROUGE and BERTScore evaluations."""
        return {
            "rouge": self.evaluate_rouge(references, predictions),
            "bert_score": self.evaluate_bert_score(references, predictions)
        }

# Example usage
if __name__ == "__main__":
    evaluator = Evaluator()
    references = ["Transformers have revolutionized NLP with self-attention."]
    predictions = ["Transformers changed NLP by introducing self-attention."]
    results = evaluator.evaluate(references, predictions)
    print(results)
