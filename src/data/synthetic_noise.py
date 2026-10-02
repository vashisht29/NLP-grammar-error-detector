"""
Synthetic Grammatical Error Generator for Deep Learning Data Augmentation.
Applies realistic grammatical perturbations to clean English sentences
(Subject-Verb disagreement, tense shifts, article errors, preposition swaps, typos).
"""
import random
import re
from typing import List, Tuple, Dict


class SyntheticErrorGenerator:
    """
    Generates synthetic grammatical errors from clean English text
    to train or evaluate Grammatical Error Detection (GED) models.
    """

    PREPOSITIONS = ["in", "on", "at", "to", "for", "with", "by", "from", "about"]
    ARTICLES = ["a", "an", "the"]

    SVA_MAP = {
        "goes": "go", "has": "have", "does": "do", "is": "are", "was": "were",
        "walks": "walk", "runs": "run", "eats": "eat", "knows": "know",
        "sees": "see", "likes": "like", "works": "work", "plays": "play"
    }

    PAST_TENSE_MAP = {
        "went": "go", "ate": "eat", "saw": "see", "bought": "buy",
        "wrote": "write", "took": "take", "came": "come", "walked": "walk",
        "played": "play", "worked": "work", "studied": "study"
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def perturb_sentence(self, sentence: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Injects realistic synthetic errors into a clean sentence.
        
        Returns:
            Tuple of (erroneous_sentence, list_of_injected_errors)
        """
        words = sentence.split()
        injected = []
        new_words = list(words)

        for i, word in enumerate(words):
            clean_w = re.sub(r'[^\w]', '', word).lower()

            # 1. Subject-Verb Agreement error
            if clean_w in self.SVA_MAP and random.random() < 0.4:
                replacement = self.SVA_MAP[clean_w]
                new_words[i] = word.replace(clean_w, replacement)
                injected.append({
                    "type": "Subject-Verb Agreement",
                    "original": clean_w,
                    "injected": replacement
                })
                continue

            # 2. Past tense error
            if clean_w in self.PAST_TENSE_MAP and random.random() < 0.35:
                replacement = self.PAST_TENSE_MAP[clean_w]
                new_words[i] = word.replace(clean_w, replacement)
                injected.append({
                    "type": "Verb Tense",
                    "original": clean_w,
                    "injected": replacement
                })
                continue

            # 3. Article error (drop or swap)
            if clean_w in self.ARTICLES and random.random() < 0.35:
                other_articles = [a for a in self.ARTICLES if a != clean_w]
                if random.random() < 0.5:
                    replacement = random.choice(other_articles)
                    new_words[i] = word.replace(clean_w, replacement)
                    injected.append({"type": "Article Swap", "original": clean_w, "injected": replacement})
                else:
                    new_words[i] = ""
                    injected.append({"type": "Article Omission", "original": clean_w, "injected": "[omitted]"})
                continue

            # 4. Preposition error
            if clean_w in self.PREPOSITIONS and random.random() < 0.3:
                other_preps = [p for p in self.PREPOSITIONS if p != clean_w]
                replacement = random.choice(other_preps)
                new_words[i] = word.replace(clean_w, replacement)
                injected.append({"type": "Preposition Error", "original": clean_w, "injected": replacement})
                continue

            # 5. Typo / letter swap
            if len(clean_w) > 4 and random.random() < 0.15:
                idx = random.randint(1, len(clean_w) - 2)
                typo_w = clean_w[:idx] + clean_w[idx + 1] + clean_w[idx] + clean_w[idx + 2:]
                new_words[i] = word.replace(clean_w, typo_w)
                injected.append({"type": "Spelling Typo", "original": clean_w, "injected": typo_w})

        noisy_sentence = " ".join([w for w in new_words if w.strip()])
        return noisy_sentence, injected
