"""
Unit and integration tests for the Deep Learning Grammatical Error Detection (GED) System.
Verifies token error localization, linguistic taxonomy classification, and metrics computation.
"""
import unittest
from src.model.detector import DeepGrammarDetector
from src.model.classifier import ErrorClassifier
from src.evaluation.metrics import GEDMetrics
from src.data.synthetic_noise import SyntheticErrorGenerator


class TestGrammarDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = DeepGrammarDetector()

    def test_subject_verb_agreement(self):
        sentence = "She go to market every evening."
        result = self.detector.detect(sentence)
        self.assertFalse(result.is_grammatically_correct)
        self.assertGreater(result.error_count, 0)
        types = [e.error_type for e in result.errors]
        self.assertIn("Subject-Verb Agreement", types)

    def test_verb_tense_consistency(self):
        sentence = "Yesterday I eat dinner at six."
        result = self.detector.detect(sentence)
        self.assertFalse(result.is_grammatically_correct)
        self.assertGreater(result.error_count, 0)
        types = [e.error_type for e in result.errors]
        self.assertIn("Verb Tense & Form", types)

    def test_article_misuse(self):
        sentence = "He is a honest man."
        result = self.detector.detect(sentence)
        self.assertFalse(result.is_grammatically_correct)
        types = [e.error_type for e in result.errors]
        self.assertIn("Article / Determiner", types)

    def test_preposition_error(self):
        sentence = "She is interested on machine learning."
        result = self.detector.detect(sentence)
        self.assertFalse(result.is_grammatically_correct)
        types = [e.error_type for e in result.errors]
        self.assertIn("Preposition Error", types)

    def test_complex_multiclause_sentence(self):
        sentence = "The student which was siting in the back of the class room did not payed attention to the teacher because he was look at his phone and write texts to his friends even though the exam is tomorrow morning."
        result = self.detector.detect(sentence)
        self.assertFalse(result.is_grammatically_correct)
        self.assertGreaterEqual(result.error_count, 5)
        self.assertIn("who was sitting", result.corrected_sentence)
        self.assertIn("classroom", result.corrected_sentence)
        self.assertIn("did not pay attention", result.corrected_sentence)
        self.assertIn("was looking at his phone and writing texts", result.corrected_sentence)

    def test_clean_sentence(self):
        clean = "The quick brown fox jumps over the lazy dog."
        result = self.detector.detect(clean)
        self.assertTrue(result.is_grammatically_correct)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(len(result.errors), 0)

    def test_metrics_f05_calculation(self):
        # Precision 1.0, Recall 0.5
        # F0.5 = (1 + 0.25) * (1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 1.25 * 0.5 / 0.75 = 0.625 / 0.75 = 0.8333
        f05 = GEDMetrics.compute_f_beta(precision=1.0, recall=0.5, beta=0.5)
        self.assertAlmostEqual(f05, 0.8333, places=3)

    def test_sentence_level_metrics(self):
        preds = [True, True, False, False]
        targets = [True, False, False, True]
        metrics = GEDMetrics.evaluate_sentence_level(preds, targets)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_synthetic_noise_generator(self):
        clean = "She goes to school every day and likes learning."
        gen = SyntheticErrorGenerator(seed=123)
        noisy, injected = gen.perturb_sentence(clean)
        self.assertIsInstance(noisy, str)
        self.assertIsInstance(injected, list)


if __name__ == "__main__":
    unittest.main()
