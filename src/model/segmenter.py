"""
Subword and Compound Word Segmenter for GED & NLP.
Splits run-on and glued words (e.g. 'speelingmistake' -> 'spelling mistake', 'verygood' -> 'very good')
using statistical unigram frequency scoring while strictly protecting genuine English compound words
(e.g. 'understand', 'sometimes', 'therefore', 'everything', 'without', 'football').
"""
import math
from typing import Optional, Tuple, List, Dict
from src.model.spellchecker import SpellChecker


# Authentic English compound words that must NEVER be split
AUTHENTIC_COMPOUND_WORDS = {
    "understand", "understanding", "understood", "understands",
    "sometimes", "therefore", "everything", "everybody", "everyone", "everywhere",
    "without", "within", "throughout", "meanwhile", "nevertheless", "furthermore",
    "someone", "somebody", "somewhere", "something", "someday",
    "anyone", "anybody", "anywhere", "anything", "anytime",
    "nobody", "nowhere", "nothing", "none",
    "cannot", "into", "onto", "upon",
    "themselves", "himself", "herself", "itself", "ourselves", "myself", "yourself", "yourselves",
    "background", "football", "baseball", "basketball", "sunflower", "butterfly",
    "supermarket", "railway", "notebook", "textbook", "keyboard", "newspaper",
    "underground", "overseas", "overnight", "overall", "network", "software", "hardware",
    "database", "online", "offline", "website", "framework", "lifecycle", "pipeline",
    "guideline", "guidelines", "highlight", "highlights", "milestone", "milestones",
    "breakthrough", "feedback", "dashboard", "timeline", "workflow", "workforce",
    "workplace", "counterpart", "infrastructure", "masterpiece", "widespread"
}


class SubwordSegmenter:
    """
    Splits glued/merged words into standard English words using unigram frequency
    optimization and spellchecker integration.
    """

    _INITIALIZED = False

    @classmethod
    def _init(cls):
        if not cls._INITIALIZED:
            SpellChecker._initialize()
            cls._INITIALIZED = True

    @classmethod
    def split_compound(cls, token: str) -> Optional[str]:
        """
        Attempts to split an unrecognized run-on token into 2 standard English words.
        Returns None if the word is already valid or is a single-word typo/anagram.
        """
        cls._init()
        raw = token.strip()
        if len(raw) < 5 or not raw.isalpha():
            return None

        w_lower = raw.lower()

        # Rule 1: NEVER split authentic English compound words
        if w_lower in AUTHENTIC_COMPOUND_WORDS:
            return None

        # Rule 2: If the word is already valid in dictionary, NEVER split it
        if SpellChecker.is_valid_word(w_lower):
            return None

        # Rule 3: Known human typo or conversational typo -> NEVER split into subwords
        if w_lower in SpellChecker._HUMAN_ERRORS:
            return None

        # Rule 4: Typoglycemia scrambled word (e.g. porject -> project, mtiaske -> mistake, setneces -> sentences)
        if SpellChecker.resolve_typoglycemia(w_lower) is not None:
            return None

        # Check if this is a classic single-word spelling error (e.g. archeologists, finalistas, unforseeable)
        single_cand = SpellChecker.correct_word(w_lower)

        # 2-word partition
        VALID_2_LETTER = {"my", "no", "we", "he", "it", "me", "us", "up", "do", "go", "so", "to", "in", "on", "at", "by", "as", "is", "am", "of"}
        INVALID_SUFFIXES = {"ed", "al", "ly", "er", "es", "ing", "tion", "ment", "able", "ness", "its", "as"}

        best_split = None
        best_score = -float("inf")

        for k in range(2, len(w_lower) - 1):
            p1 = w_lower[:k]
            p2 = w_lower[k:]
            if len(p1) == 2 and p1 not in VALID_2_LETTER:
                continue
            if len(p2) == 2 and p2 not in VALID_2_LETTER:
                continue
            if p2 in INVALID_SUFFIXES:
                continue

            # Resolve p1, p2
            w1 = p1 if SpellChecker.is_valid_word(p1) else (SpellChecker._HUMAN_ERRORS.get(p1) or SpellChecker.correct_word(p1))
            w2 = p2 if SpellChecker.is_valid_word(p2) else (SpellChecker._HUMAN_ERRORS.get(p2) or SpellChecker.correct_word(p2))
            if not w1 or not w2:
                continue

            f1 = max(SpellChecker._WORD_FREQ.get(w1.lower(), 10), 1)
            f2 = max(SpellChecker._WORD_FREQ.get(w2.lower(), 10), 1)
            score = math.log10(f1) + math.log10(f2)
            if w1.lower() == p1 and w2.lower() == p2:
                score += 3.0
            else:
                if w1.lower() != p1: score -= 3.0
                if w2.lower() != p2: score -= 3.0

            if score > best_score:
                best_score = score
                best_split = (w1, w2)

        if best_split:
            is_exact = (best_split[0].lower() == w_lower[:len(best_split[0])] and best_split[1].lower() == w_lower[len(best_split[0]):])
            # If single candidate exists and split is NOT an exact combination of two major words, prefer single word!
            if single_cand and not is_exact:
                return None
            if single_cand and is_exact:
                # If suffix is just a short inflection like "finalistas" -> "finalists", prefer single candidate
                if len(best_split[1]) <= 2:
                    return None
            res = list(best_split)
            if raw[0].isupper():
                res[0] = res[0].capitalize()
            return " ".join(res)

        return None

    @classmethod
    def _resolve_token(cls, part: str) -> Optional[str]:
        """Returns the valid word for part (either direct match or 1-edit correction)."""
        if SpellChecker.is_valid_word(part):
            return part
        # Check human typo table
        if part in SpellChecker._HUMAN_ERRORS:
            return SpellChecker._HUMAN_ERRORS[part]
        # Check spell checker candidate
        corr = SpellChecker.correct_word(part)
        if corr:
            return corr
        return None

