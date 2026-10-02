"""
Lexical Semantics, Synonyms, and Antonyms Knowledge Base.
Provides rich vocabulary understanding, synonym substitution, antonym contrasts,
and semantic contextual disambiguation for the GED pipeline.
"""
from typing import List, Dict, Set, Optional, Tuple


# Rich vocabulary synonyms mapping (covering core English semantic fields)
SYNONYMS_MAP: Dict[str, List[str]] = {
    # Quality & Merit
    "good": ["fine", "excellent", "superb", "great", "worthy", "favorable", "splendid"],
    "bad": ["poor", "inferior", "substandard", "defective", "awful", "terrible", "dreadful"],
    "excellent": ["outstanding", "superb", "exceptional", "marvelous", "magnificent"],
    "terrible": ["awful", "dreadful", "horrible", "atrocious", "abysmal"],
    
    # Beauty & Appearance
    "beautiful": ["pretty", "gorgeous", "attractive", "handsome", "stunning", "lovely"],
    "ugly": ["unattractive", "hideous", "unsightly", "grotesque"],
    "pretty": ["attractive", "lovely", "charming", "comely"],
    "handsome": ["good-looking", "attractive", "well-favored", "dashing"],

    # Size & Scale
    "big": ["large", "huge", "enormous", "gigantic", "massive", "immense", "colossal"],
    "small": ["little", "tiny", "diminutive", "compact", "miniature", "petite"],
    "huge": ["enormous", "gigantic", "massive", "colossal", "vast", "immense"],
    "tiny": ["minuscule", "microscopic", "minute", "petite", "slight"],

    # Speed & Pace
    "fast": ["quick", "rapid", "swift", "speedy", "brisk", "hasty"],
    "slow": ["sluggish", "unhurried", "leisurely", "gradual", "dilatory"],
    "quick": ["rapid", "fast", "prompt", "instant", "immediate"],

    # Difficulty & Complexity
    "difficult": ["hard", "challenging", "tough", "arduous", "complex", "demanding"],
    "easy": ["simple", "effortless", "straightforward", "uncomplicated", "facile"],
    "complex": ["complicated", "intricate", "involved", "sophisticated", "convoluted"],
    "simple": ["basic", "plain", "elementary", "straightforward", "uncomplicated"],

    # Emotion & Mind
    "happy": ["glad", "joyful", "cheerful", "delighted", "content", "ecstatic"],
    "sad": ["unhappy", "sorrowful", "depressed", "gloomy", "mournful", "dejected"],
    "angry": ["furious", "irate", "enraged", "indignant", "wrathful", "vexed"],
    "calm": ["peaceful", "serene", "tranquil", "placid", "composed", "collected"],
    "smart": ["intelligent", "clever", "bright", "wise", "sharp", "astute"],
    "stupid": ["foolish", "unwise", "silly", "mindless", "dense"],

    # Power & Intensity
    "strong": ["powerful", "robust", "sturdy", "forceful", "vigorous", "tough"],
    "weak": ["fragile", "feeble", "frail", "delicate", "faint", "debilitated"],
    "powerful": ["potent", "mighty", "forceful", "dominant", "commanding"],

    # Temperature & Environment
    "hot": ["warm", "scorching", "boiling", "sweltering", "torrid", "scalding"],
    "cold": ["chilly", "frigid", "freezing", "frosty", "glacial", "wintry"],
    "quiet": ["silent", "peaceful", "still", "hushed", "soundless", "calm"],
    "loud": ["noisy", "clamorous", "boisterous", "deafening", "vociferous"],

    # Common Actions & Communication
    "begin": ["start", "commence", "initiate", "launch", "originate"],
    "end": ["finish", "conclude", "terminate", "finalize", "cease"],
    "help": ["assist", "aid", "support", "serve", "facilitate"],
    "talk": ["speak", "chat", "converse", "discuss", "communicate"],
    "make": ["create", "build", "construct", "produce", "generate", "fabricate"],
    "think": ["ponder", "reflect", "contemplate", "deliberate", "consider"],
    "understand": ["comprehend", "grasp", "apprehend", "discern", "fathom"],
    "multiple": ["many", "several", "numerous", "various", "manifold", "diverse"],
    "parameters": ["variables", "factors", "criteria", "dimensions", "measurements"],
    "vocabulary": ["lexicon", "words", "glossary", "terminology", "parlance"],
    "sentence": ["phrase", "statement", "clause", "utterance", "expression"]
}

# Antonyms mapping (opposites)
ANTONYMS_MAP: Dict[str, str] = {
    "good": "bad", "bad": "good",
    "hot": "cold", "cold": "hot",
    "big": "small", "small": "big",
    "huge": "tiny", "tiny": "huge",
    "fast": "slow", "slow": "fast",
    "quick": "slow",
    "happy": "sad", "sad": "happy",
    "difficult": "easy", "easy": "difficult",
    "hard": "easy",
    "complex": "simple", "simple": "complex",
    "strong": "weak", "weak": "strong",
    "quiet": "loud", "loud": "quiet",
    "peace": "war", "war": "peace",
    "accept": "reject", "reject": "accept",
    "loose": "tight", "tight": "loose",
    "lose": "find", "find": "lose",
    "begin": "end", "end": "begin",
    "start": "finish", "finish": "start",
    "beautiful": "ugly", "ugly": "beautiful",
    "smart": "foolish", "foolish": "smart",
    "true": "false", "false": "true",
    "right": "wrong", "wrong": "right",
    "rich": "poor", "poor": "rich",
    "bright": "dark", "dark": "bright",
    "clean": "dirty", "dirty": "clean",
    "safe": "dangerous", "dangerous": "safe",
    "increase": "decrease", "decrease": "increase",
    "lead": "follow", "follow": "lead",
    "give": "take", "take": "give",
    "love": "hate", "hate": "love",
    "arrive": "depart", "depart": "arrive"
}


class LexicalSemantics:
    """Provides vocabulary semantics, synonyms, and antonym relations."""

    @classmethod
    def get_synonyms(cls, word: str) -> List[str]:
        """Returns known synonyms for a word."""
        w = word.lower().strip()
        return SYNONYMS_MAP.get(w, [])

    @classmethod
    def get_antonym(cls, word: str) -> Optional[str]:
        """Returns the primary antonym for a word if known."""
        w = word.lower().strip()
        return ANTONYMS_MAP.get(w)

    @classmethod
    def are_antonyms(cls, w1: str, w2: str) -> bool:
        """Checks if two words are known antonyms."""
        w1_l, w2_l = w1.lower().strip(), w2.lower().strip()
        return ANTONYMS_MAP.get(w1_l) == w2_l or ANTONYMS_MAP.get(w2_l) == w1_l

    @classmethod
    def are_synonyms(cls, w1: str, w2: str) -> bool:
        """Checks if two words are in the same synonym cluster."""
        w1_l, w2_l = w1.lower().strip(), w2.lower().strip()
        if w1_l == w2_l:
            return True
        if w2_l in SYNONYMS_MAP.get(w1_l, []):
            return True
        if w1_l in SYNONYMS_MAP.get(w2_l, []):
            return True
        return False
