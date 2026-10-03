"""
Language Detection & Non-English Guardrail for GED.
Detects Romanized Hindi (Hinglish) and other non-English language inputs
to prevent invalid grammar corrections and prompt users to write in English only.
"""
import re
from typing import Tuple, List, Set


# Distinctive Romanized Hindi / Hinglish words (NEVER collide with genuine English words)
DISTINCTIVE_HINGLISH_WORDS: Set[str] = {
    # Question words
    "kya", "kyu", "kyun", "kyo", "kaise", "kaisa", "kaisi", "kab", "kahan", "kaha",
    "kidhar", "kaun", "kon", "kisko", "kisne", "kiske", "kitna", "kitni", "kitne",
    
    # Pronouns & Possessives
    "mera", "meri", "mere", "tera", "teri", "tere", "apna", "apni", "apne",
    "uska", "uski", "uske", "unka", "unki", "unke", "inka", "inki", "inke",
    "humara", "humari", "humare", "hamara", "hamari", "hamare", "tumhara", "tumhari", "tumhare",
    "aapka", "aapki", "aapke", "mujhe", "tujhe", "hume", "hame", "unhe", "inhe",
    "hum", "tum", "aap", "unhone", "inhone", "kisi", "kuch", "sabko", "sabhi",
    
    # Verbs, Auxiliaries & Tense markers
    "hai", "hain", "hoon", "hun", "tha", "thi", "hoga", "hogi", "hoge", "honge",
    "raha", "rahi", "rahe", "rha", "rhi", "rhe", "kare", "karo", "karna", "krna",
    "karenge", "karega", "karegi", "kiya", "kiye",
    "karta", "karti", "karte", "krta", "krti", "krte",
    "hota", "hoti", "hote", "hona", "hua", "hue", "hui",
    "gaya", "gayi", "gaye", "gya", "gyi", "gye",
    "aata", "aati", "aate", "aana", "aao", "aaya", "aayi", "aaye",
    "jaata", "jaati", "jaate", "jana", "jao",
    "dekh", "dekho", "dekha", "dekhna", "dekhne",
    "sun", "suno", "suna", "sunna",
    "bol", "bolo", "bola", "bolna",
    "bata", "batao", "batana", "bataya",
    "chal", "chalo", "chala", "chalna",
    "sakta", "sakti", "sakte", "skta", "skti", "skte",
    "chahiye", "chahta", "chahti", "chahte",
    "samajh", "samjha", "samjho", "samajhna",
    "lag", "laga", "lage", "lagi", "lagta", "lagti", "lagte",
    
    # Common Adverbs, Conjunctions & Particles
    "nahi", "nhi", "mat", "haan", "han", "bhi", "toh", "phir", "fir",
    "lekin", "magar", "parantu", "kintu", "kyunki", "kyuki", "taaki", "taki",
    "agar", "yadi", "aur", "ya", "bas", "sirf",
    "bohot", "bahut", "bhot", "bhout", "jyada", "zyada", "kam", "thoda", "thodi", "thode",
    "theek", "thik", "sahi", "galat", "accha", "achha", "achi", "ache", "achhe",
    "bura", "buri", "bure",
    "aaj", "kal", "parso", "abhi", "pehle", "baad", "subah", "shaam", "raat",
    "andar", "bahar", "upar", "niche", "aage", "piche",
    
    # Colloquial Address & Slang
    "bhai", "yaar", "yar", "dost", "banda", "bande", "log", "loge", "logi",
    "baat", "baatein", "kaam", "pata", "pta", "malum", "matlab", "matlb",
    "kripya", "dhanyawad", "namaste", "pranam"
}

# English collision protection - words that appear in English and must NOT trigger Hindi detection
ENGLISH_COLLISION_WORDS: Set[str] = {
    "the", "me", "to", "in", "is", "so", "he", "an", "at", "as", "us", "on", "do", "go", "no", "are", "am", "or"
}

# Strong Distinctive Hinglish N-Grams
HINGLISH_PHRASES: List[str] = [
    r"\bkya\s+(kar|hua|hai|baat|hoga|kare|bol)\b",
    r"\b(kar|kr)\s+(rha|raha|rahi|rhe|rahe|diya|de)\b",
    r"\b(nhi|nahi)\s+(hai|tha|hoga|karo|pata|hua|mila)\b",
    r"\b(ho|gaya|gya)\s+(hai|tha|hoga)\b",
    r"\b(theek|thik|sahi)\s+(hai|tha|karo|hoga)\b",
    r"\b(ye|yeh)\s+(kya|kuch|theek|sahi|hai)\b",
    r"\b(bhai|yaar|yar)\s+(ye|kya|dekh|sun|batao|kuch)\b",
    r"\b(kaise|kaisa|kaisi)\s+(ho|hai|kare|hua)\b",
    r"\b(kuch|koi)\s+(bhi|nahi|nhi|baat)\b",
    r"\b(mujhko|tujhko|humko|aapko)\s+(bhi|ye|kuch)\b",
    r"\b(dekh|dekho)\s+(na|bhai|yaar|ye)\b",
    r"\b(bol|bolo)\s+(na|bhai|kya)\b",
    r"\b(mat|mt)\s+(kar|karo|bol|bolo|ja|jao)\b",
    r"\b(aaj|kal)\s+(ka|ki|ke|ko|mein|me)\b",
    r"\b(mera|meri|mere)\s+(naam|dost|bhai|kaam|phone)\b",
    r"\b(gaye|kiye|aaye)\s+the\b",
]


class HinglishDetector:
    """
    High-accuracy detector for Romanized Hindi / Hinglish.
    Evaluates lexical density and syntax n-grams to flag non-English queries.
    """

    @classmethod
    def check_hinglish(cls, text: str) -> Tuple[bool, float, List[str]]:
        """
        Determines whether the provided text is Hinglish / Romanized Hindi.
        
        Args:
            text: Input user sentence.
            
        Returns:
            Tuple of:
            - is_hinglish (bool)
            - confidence (float between 0.0 and 1.0)
            - detected_markers (list of detected Hindi tokens or phrases)
        """
        clean = text.strip().lower()
        if not clean:
            return False, 0.0, []

        detected_markers: List[str] = []

        # 1. Check strong multi-word Hinglish phrase patterns
        for pat in HINGLISH_PHRASES:
            m = re.search(pat, clean, re.IGNORECASE)
            if m:
                detected_markers.append(m.group(0))

        if detected_markers:
            return True, 0.98, detected_markers

        # 2. Tokenize and evaluate distinctive word vocabulary
        words = re.findall(r"\b[a-zA-Z]+\b", clean)
        if not words:
            return False, 0.0, []

        hindi_words_found = []
        for w in words:
            if w in ENGLISH_COLLISION_WORDS:
                continue
            if w in DISTINCTIVE_HINGLISH_WORDS:
                hindi_words_found.append(w)

        match_count = len(hindi_words_found)
        total_words = len(words)
        ratio = match_count / total_words

        # Rule A: 2 or more distinct Hindi words
        if match_count >= 2:
            return True, min(0.99, 0.85 + (match_count * 0.05)), hindi_words_found

        # Rule B: 1 Hindi word in a very short sentence (e.g. "kya?", "kaise?", "theek hai")
        if match_count >= 1 and total_words <= 3:
            return True, 0.92, hindi_words_found

        # Rule C: Ratio > 25% in longer text
        if match_count >= 1 and ratio >= 0.25:
            return True, 0.88, hindi_words_found

        return False, 0.0, []
