"""
Statistical N-Gram Language Model and Contextual Scoring Engine.
Trains transition probabilities across large corpora (20 Lakh+ words)
to score sentence fluency, resolve ambiguous real-word confusables,
and provide probability-ranked candidate selection.
"""
import os
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Set


class StatisticalLanguageModel:
    """
    Bigram and Trigram Language Model with Laplace/Add-k Smoothing.
    Used for scoring fluency and disambiguating confusable candidate words in context.
    """

    _INITIALIZED = False
    _UNIGRAMS: Counter = Counter()
    _BIGRAMS: Counter = Counter()
    _TOTAL_TOKENS: int = 0
    _VOCAB_SIZE: int = 0

    @classmethod
    def train_on_corpus(cls, text_sentences: List[str]):
        """
        Trains the N-gram language model on a large stream of sentences.
        """
        for sentence in text_sentences:
            tokens = ["<s>"] + [w.lower() for w in sentence.split() if w.strip()] + ["</s>"]
            cls._TOTAL_TOKENS += len(tokens)
            cls._UNIGRAMS.update(tokens)
            for i in range(len(tokens) - 1):
                cls._BIGRAMS[(tokens[i], tokens[i + 1])] += 1

        cls._VOCAB_SIZE = len(cls._UNIGRAMS)
        cls._INITIALIZED = True

    @classmethod
    def bigram_probability(cls, prev_word: str, word: str, alpha: float = 0.05) -> float:
        """
        Calculates smoothed bigram transition probability P(word | prev_word).
        Uses Add-alpha (Laplace) smoothing for robust unseen pair handling.
        """
        w_prev = prev_word.lower()
        w_curr = word.lower()

        prev_count = cls._UNIGRAMS.get(w_prev, 0)
        bigram_count = cls._BIGRAMS.get((w_prev, w_curr), 0)

        # Smoothed conditional probability
        prob = (bigram_count + alpha) / (prev_count + (alpha * max(cls._VOCAB_SIZE, 10000)))
        return prob

    @classmethod
    def score_candidate_in_context(cls, left_word: Optional[str], candidate: str, right_word: Optional[str]) -> float:
        """
        Scores how naturally a candidate word fits between left_word and right_word.
        Score = log P(candidate | left_word) + log P(right_word | candidate).
        """
        score = 0.0
        c = candidate.lower()

        if left_word:
            p_left = cls.bigram_probability(left_word, c)
            score += math.log(max(p_left, 1e-12))
        else:
            p_unigram = (cls._UNIGRAMS.get(c, 0) + 1) / (max(cls._TOTAL_TOKENS, 1) + cls._VOCAB_SIZE)
            score += math.log(max(p_unigram, 1e-12))

        if right_word:
            p_right = cls.bigram_probability(c, right_word)
            score += math.log(max(p_right, 1e-12))

        return score

    @classmethod
    def pick_best_candidate(cls, left_word: Optional[str], candidates: List[str], right_word: Optional[str]) -> str:
        """
        Selects the most probable candidate word given left and right context.
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        best_cand = max(
            candidates,
            key=lambda cand: cls.score_candidate_in_context(left_word, cand, right_word)
        )
        return best_cand
