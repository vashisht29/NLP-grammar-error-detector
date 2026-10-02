"""
Evaluation Metrics for English Grammatical Error Detection (GED) & Correction (GEC).
Implements Precision, Recall, F0.5 Score (CoNLL/BEA benchmark standard), and Detection Accuracy.
"""
from typing import List, Dict, Any, Tuple


class GEDMetrics:
    """
    Computes evaluation metrics for Grammatical Error Detection (GED) systems.
    Standard academic metric is F0.5 (weights precision twice as heavily as recall).
    """

    @staticmethod
    def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
        """
        Computes F-beta score:
        F_beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
        """
        beta_sq = beta ** 2
        denominator = (beta_sq * precision) + recall
        if denominator == 0:
            return 0.0
        return (1 + beta_sq) * (precision * recall) / denominator

    @classmethod
    def evaluate_sentence_level(
        cls,
        predictions: List[bool],  # True if predicted as having error, False if clean
        ground_truths: List[bool]  # True if actually has error, False if clean
    ) -> Dict[str, float]:
        """
        Calculates binary classification metrics at the sentence level:
        - True Positives (TP): Correctly flagged as containing errors
        - False Positives (FP): Clean sentence erroneously flagged as error
        - False Negatives (FN): Erroneous sentence missed
        - True Negatives (TN): Clean sentence correctly passed
        """
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p and g)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if not p and g)
        tn = sum(1 for p, g in zip(predictions, ground_truths) if not p and not g)

        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f0_5 = cls.compute_f_beta(precision, recall, beta=0.5)
        f1 = cls.compute_f_beta(precision, recall, beta=1.0)

        return {
            "total_sentences": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f0_5_score": round(f0_5, 4),
            "f1_score": round(f1, 4)
        }

    @classmethod
    def evaluate_token_spans(
        cls,
        predicted_spans: List[List[Tuple[int, int]]],
        ground_truth_spans: List[List[Tuple[int, int]]]
    ) -> Dict[str, float]:
        """
        Evaluates token-level span detection across a test batch.
        """
        tp = 0
        fp = 0
        fn = 0

        for pred_list, gt_list in zip(predicted_spans, ground_truth_spans):
            pred_set = set(pred_list)
            gt_set = set(gt_list)

            tp += len(pred_set.intersection(gt_set))
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f0_5 = cls.compute_f_beta(precision, recall, beta=0.5)

        return {
            "span_tp": tp,
            "span_fp": fp,
            "span_fn": fn,
            "token_precision": round(precision, 4),
            "token_recall": round(recall, 4),
            "token_f0_5_score": round(f0_5, 4)
        }
