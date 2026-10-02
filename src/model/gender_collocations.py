"""
Gender Collocation and Semantic Concord Engine.
Ensures gender appropriateness for adjectives, titles, and gender-specific nouns:
- 'she is handsome' -> 'she is beautiful'
- 'he is an actress' -> 'he is an actor'
- 'she is a waiter' -> 'she is a waitress'
"""
import re
from typing import Tuple, List, Dict


GENDER_RULES = [
    # 1. Female Subject with 'handsome' -> 'beautiful'
    (
        r'\b(she|the\s+girl|the\s+woman|the\s+lady|my\s+mother|my\s+mom|my\s+sister|my\s+daughter|the\s+queen|the\s+princess|my\s+wife|my\s+aunt|my\s+niece)\s+((?:is|was|looks|seems|became|appears)\s+(?:a\s+)?(?:very\s+|so\s+|really\s+|quite\s+|extremely\s+)?)(handsome)\b',
        r'\1 \2beautiful',
        "Gender Collocation: 'handsome' is typically used for men or boys. For women or girls, 'beautiful', 'pretty', or 'gorgeous' is preferred."
    ),

    # 2. Male Subject with feminine titles/occupations
    (
        r'\b(he|the\s+boy|the\s+man|my\s+father|my\s+dad|my\s+brother|my\s+son|the\s+king|the\s+prince|my\s+husband|my\s+uncle|my\s+nephew)\s+((?:is|was)\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)?)(actress)\b',
        r'\1 \2actor',
        "Gender Concord: The subject is male, so masculine noun 'actor' should be used instead of 'actress'."
    ),
    (
        r'\b(he|the\s+boy|the\s+man|my\s+father|my\s+dad|my\s+brother|my\s+son|the\s+king|the\s+prince|my\s+husband|my\s+uncle|my\s+nephew)\s+((?:is|was)\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)?)(waitress)\b',
        r'\1 \2waiter',
        "Gender Concord: The subject is male, so masculine noun 'waiter' should be used instead of 'waitress'."
    ),
    (
        r'\b(he|the\s+boy|the\s+man|my\s+father|my\s+dad|my\s+brother|my\s+son|the\s+king|the\s+prince|my\s+husband|my\s+uncle|my\s+nephew)\s+((?:is|was)\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)?)(heroine)\b',
        r'\1 \2hero',
        "Gender Concord: The subject is male, so masculine noun 'hero' should be used instead of 'heroine'."
    ),
    (
        r'\b(he|the\s+boy|the\s+man|my\s+father|my\s+dad|my\s+brother|my\s+son|the\s+king|the\s+prince|my\s+husband|my\s+uncle|my\s+nephew)\s+((?:is|was)\s+(?:a\s+)?(?:[a-zA-Z\-]+\s+)?)(widow)\b',
        r'\1 \2widower',
        "Gender Concord: A man whose spouse has passed away is a 'widower'."
    ),

    # 3. Female Subject with masculine titles
    (
        r'\b(she|the\s+girl|the\s+woman|the\s+lady|my\s+mother|my\s+mom|my\s+sister|my\s+daughter|the\s+queen|the\s+princess|my\s+wife|my\s+aunt|my\s+niece)\s+((?:is|was)\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)?)(actor)\b',
        r'\1 \2actress',
        "Gender Concord: The subject is female, so feminine noun 'actress' should be used."
    ),
    (
        r'\b(she|the\s+girl|the\s+woman|the\s+lady|my\s+mother|my\s+mom|my\s+sister|my\s+daughter|the\s+queen|the\s+princess|my\s+wife|my\s+aunt|my\s+niece)\s+((?:is|was)\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)?)(waiter)\b',
        r'\1 \2waitress',
        "Gender Concord: The subject is female, so feminine noun 'waitress' should be used."
    ),
    (
        r'\b(she|the\s+girl|the\s+woman|the\s+lady|my\s+mother|my\s+mom|my\s+sister|my\s+daughter|the\s+queen|the\s+princess|my\s+wife|my\s+aunt|my\s+niece)\s+((?:is|was)\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)?)(hero)\b',
        r'\1 \2heroine',
        "Gender Concord: The subject is female, so feminine noun 'heroine' should be used."
    ),
    (
        r'\b(she|the\s+girl|the\s+woman|the\s+lady|my\s+mother|my\s+mom|my\s+sister|my\s+daughter|the\s+queen|the\s+princess|my\s+wife|my\s+aunt|my\s+niece)\s+((?:is|was)\s+(?:a\s+)?(?:[a-zA-Z\-]+\s+)?)(widower)\b',
        r'\1 \2widow',
        "Gender Concord: A woman whose spouse has passed away is a 'widow'."
    ),

    # 4. Pronoun-Antecedent Gender Concord
    (
        r'\b(she|the\s+girl|the\s+woman|the\s+lady|my\s+mother|my\s+mom|my\s+sister|my\s+daughter|the\s+queen|the\s+princess|my\s+wife|my\s+aunt|my\s+niece|mary|sarah|emma|alice)\s+((?:[a-zA-Z]+ed|[a-zA-Z]+s|forgot|lost|brought|took|finished|did|called|found)\s+)his\s+([a-zA-Z]+)\b',
        r'\1 \2her \3',
        "Pronoun-Antecedent Agreement: The antecedent subject is female, requiring the feminine possessive pronoun 'her' instead of 'his'."
    ),
    (
        r'\b(he|the\s+boy|the\s+man|my\s+father|my\s+dad|my\s+brother|my\s+son|the\s+king|the\s+prince|my\s+husband|my\s+uncle|my\s+nephew|john|david|michael|james)\s+((?:[a-zA-Z]+ed|[a-zA-Z]+s|forgot|lost|brought|took|finished|did|called|found)\s+)her\s+([a-zA-Z]+)\b',
        r'\1 \2his \3',
        "Pronoun-Antecedent Agreement: The antecedent subject is male, requiring the masculine possessive pronoun 'his' instead of 'her'."
    ),

    # 5. Semantic / biological consistency
    (
        r'\b(he|the\s+boy|the\s+man)\s+((?:is|was|got|became)\s+)(pregnant)\b',
        r'\1 \2expecting a baby',
        "Biological/Semantic Inconsistency: Biologically, men do not become pregnant. Phrased as 'expecting a baby' or check subject."
    )
]


class GenderCollocationEngine:
    """Detects and corrects gender collocation mismatches."""

    @classmethod
    def apply(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        revised = text
        detected = []

        for pat, repl, expl in GENDER_RULES:
            matches = list(re.finditer(pat, revised, re.IGNORECASE))
            if matches:
                for m in matches:
                    detected.append({
                        "matched": m.group(0),
                        "explanation": expl
                    })
                revised = re.sub(pat, repl, revised, flags=re.IGNORECASE)

        return revised, detected
