"""
Linguistic Error Classifier for Grammatical Error Detection (GED).
Categorizes detected differences between original and corrected sentences
into linguistic error taxonomies (Subject-Verb Agreement, Tense, Articles, Prepositions, etc.)
"""
import re
from typing import Tuple, List, Dict, Optional


COMMON_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "since", "without", "against", "towards"
}

ARTICLES_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "every"
}

SINGULAR_PRONOUNS = {"he", "she", "it", "this", "that", "everyone", "everybody", "someone", "nobody"}
PLURAL_PRONOUNS = {"they", "we", "these", "those", "both"}

AUXILIARY_VERBS = {
    "is", "are", "am", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must"
}


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1.lower() != c2.lower())
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class ErrorClassifier:
    """
    Classifies token-level differences into grammatical categories
    with detailed educational explanations.
    """

    @classmethod
    def classify(
        cls,
        orig_tokens: List[str],
        corr_tokens: List[str],
        context_before: List[str],
        context_after: List[str]
    ) -> Tuple[str, str, float]:
        """
        Classifies an error given original tokens, suggested tokens, and surrounding context.
        
        Returns:
            Tuple of (category_name, explanation, confidence)
        """
        orig_str = " ".join(orig_tokens).strip()
        corr_str = " ".join(corr_tokens).strip()
        orig_lower = orig_str.lower()
        corr_lower = corr_str.lower()

        # 1. Punctuation errors
        if cls._is_punctuation_error(orig_str, corr_str):
            src_disp = orig_str if orig_str else "[missing]"
            tgt_disp = corr_str if corr_str else "[omitted]"
            return (
                "Punctuation",
                f"Punctuation correction: '{src_disp}' should be replaced by '{tgt_disp}'.",
                0.95
            )

        # 2. Capitalization errors
        if orig_str.lower() == corr_str.lower() and orig_str != corr_str:
            return (
                "Capitalization",
                f"Word '{orig_str}' should be capitalized as '{corr_str}'.",
                0.96
            )

        # 3. Missing Contraction Apostrophes (e.g. dont -> don't, cant -> can't, im -> I'm)
        if re.sub(r"['\’]", "", corr_str).lower() == orig_str.lower() and "'" in corr_str:
            return (
                "Contraction Apostrophe",
                f"Missing apostrophe in contraction: '{orig_str}' should be written as '{corr_str}'.",
                0.96
            )

        # 4. Merged Word Spacing (e.g. alot -> a lot, speelingmistake -> spelling mistake, needto talkto -> need to talk to)
        if len(corr_tokens) > len(orig_tokens):
            orig_no_space = "".join(orig_tokens).lower()
            corr_no_space = "".join(corr_tokens).lower()
            if orig_no_space == corr_no_space or compute_levenshtein_distance(orig_no_space, corr_no_space) <= 4:
                return (
                    "Merged Word Spacing",
                    f"Missing space between words: '{orig_str}' should be written as separate words '{corr_str}'.",
                    0.96
                )

        # 4a. Split Compound Words (e.g. 'wel come' -> 'welcome', 'with out' -> 'without')
        if len(orig_tokens) > len(corr_tokens):
            orig_no_space = "".join(orig_tokens).lower()
            corr_no_space = "".join(corr_tokens).lower().rstrip(",.!?")
            if orig_no_space == corr_no_space:
                return (
                    "Compound Word / Spacing",
                    f"Compound word error: '{orig_str}' should be written as a single combined word '{corr_str}'.",
                    0.96
                )

        # 4b. Jumbled Letter Typo / Typoglycemia (e.g. mtiaske -> mistake, porject -> project)
        if len(orig_tokens) == 1 and len(corr_tokens) == 1 and len(orig_lower) >= 4 and len(corr_lower) >= 4:
            if orig_lower[0] == corr_lower[0] and orig_lower[-1] == corr_lower[-1] and sorted(orig_lower[1:-1]) == sorted(corr_lower[1:-1]) and orig_lower != corr_lower:
                return (
                    "Jumbled Letter Typo (Typoglycemia)",
                    f"Scrambled internal letters: '{orig_str}' was restored to '{corr_str}'.",
                    0.96
                )

        # 5. Informal Chat Slang & Shorthand (e.g. u -> you, what sup -> what's up, pls -> please, fr -> for real)
        chat_slang_words = (
            "u", "r", "ar", "wat", "wats", "wit", "wid", "dem", "dey", "dat", "dis",
            "pls", "plz", "thx", "ty", "thnx", "tysm", "bcoz", "cuz", "bcz", "wanna", "gonna",
            "lemme", "kinda", "sorta", "idk", "idc", "dunno", "btw", "imo", "imho", "omg", "tbh", "brb", "fyi", "smh", "rn", "asap",
            "fr", "ngl", "tldr", "aka", "np", "yw",
            "what sup", "whats up", "wassup", "wazzup", "wat sup", "wats up", "sup", "how r u", "how are u", "how ar you", "hru", "wru", "wyd", "wbu", "hbu"
        )
        if orig_lower in chat_slang_words or orig_lower.startswith("what sup") or orig_lower.startswith("whats up"):
            return (
                "Informal Chat Slang",
                f"Informal conversational expression '{orig_str}' converted to standard English '{corr_str}'.",
                0.95
            )

        # 5b. Indian English Calques & Collocations
        indian_collocations = {
            ("cousin brother", "cousin"): "Collocation: In English, simply use 'cousin' rather than 'cousin brother'.",
            ("cousin sister", "cousin"): "Collocation: In English, simply use 'cousin' rather than 'cousin sister'.",
            ("cousins brothers", "cousins"): "Collocation: In English, simply use 'cousins'.",
            ("cousins sisters", "cousins"): "Collocation: In English, simply use 'cousins'.",
            ("revert back", "reply"): "Redundant phrasing: 'revert back' is redundant; use 'reply' or 'revert'.",
            ("reverting back", "replying"): "Redundant phrasing: use 'replying'.",
            ("reverted back", "replied"): "Redundant phrasing: use 'replied'.",
            ("do the needful", "take the necessary action"): "Archaic idiom: 'do the needful' should be replaced by 'take the necessary action'.",
            ("out of station", "out of town"): "Idiom: In standard English, use 'out of town' rather than 'out of station'.",
            ("prepone", "advance"): "Regional vocabulary: 'prepone' is Indian English; standard English uses 'advance' or 'reschedule earlier'.",
            ("preponed", "advanced"): "Regional vocabulary: 'preponed' should be 'advanced' or 'rescheduled earlier'.",
            ("cope up with", "cope with"): "Redundant preposition: In standard English, use 'cope with', not 'cope up with'.",
        }
        if (orig_lower, corr_lower) in indian_collocations:
            return ("Indian English Collocation", indian_collocations[(orig_lower, corr_lower)], 0.95)

        # 6. Confused Words / Homophones
        confused_map = {
            ("accept", "except"): "Confused word: 'accept' is a verb meaning to receive. The preposition 'except' (meaning excluding) should be used.",
            ("except", "accept"): "Confused word: 'except' means excluding. Use 'accept' (verb to receive).",
            ("affect", "effect"): "Confused word: 'effect' is usually a noun (result), while 'affect' is usually a verb (to influence).",
            ("effect", "affect"): "Confused word: 'affect' is usually a verb (to influence), while 'effect' is usually a noun (result).",
            ("their", "there"): "Confused homophone: 'there' refers to place/location, while 'their' is possessive.",
            ("their", "they're"): "Confused homophone: use contraction 'they're' (they are), not possessive 'their'.",
            ("there", "their"): "Confused homophone: use possessive 'their' before a noun, not 'there'.",
            ("your", "you're"): "Confused homophone: use contraction 'you're' (you are), not possessive 'your'.",
            ("you're", "your"): "Confused homophone: use possessive 'your' before a noun, not contraction 'you're'.",
            ("then", "than"): "Confused word: use 'than' for comparisons. 'Then' refers to time or sequence.",
            ("than", "then"): "Confused word: use 'then' for time/sequence. 'Than' is used for comparisons.",
            ("to", "too"): "Confused word: use 'too' meaning excessively or also.",
            ("too", "to"): "Confused word: use preposition 'to'.",
            ("lose", "loose"): "Confused word: 'lose' is a verb (misplace/defeat), 'loose' is an adjective (not tight).",
            ("loose", "lose"): "Confused word: 'lose' is a verb (misplace/defeat), 'loose' is an adjective (not tight).",
            ("it's", "its"): "Possessive Determiner: use 'its' (without apostrophe) before a noun. 'It's' is a contraction for 'it is'.",
            ("its", "it's"): "Use contraction 'it's' (it is / it has) when followed by a verb, article, or adjective.",
            ("weather", "whether"): "Confused word: use conjunction 'whether' for alternatives/uncertainty, not noun 'weather' (climate).",
            ("whether", "weather"): "Confused word: use noun 'weather' for climatic conditions.",
            ("form", "from"): "Real-Word Spelling Error: 'form' was written instead of preposition 'from'.",
            ("from", "form"): "Real-Word Spelling Error: 'from' was written instead of noun/verb 'form'.",
            ("quiet", "quite"): "Real-Word Spelling Error: use adverb 'quite' (meaning very or completely), not adjective 'quiet' (silent).",
            ("quite", "quiet"): "Real-Word Spelling Error: use adjective 'quiet' (calm/silent), not adverb 'quite'.",
            ("peace", "piece"): "Confused homophone: use 'piece' (a part or slice), not 'peace' (absence of war/conflict).",
            ("piece", "peace"): "Confused homophone: use 'peace' (tranquility/state of harmony), not 'piece'.",
            ("break", "brake"): "Confused homophone: use 'brake' (stopping mechanism), not 'break'.",
            ("brake", "break"): "Confused homophone: use 'break' (rest or fracture), not 'brake'.",
            ("desert", "dessert"): "Real-word spelling error: 'dessert' (sweet course) should be spelled with double 's'.",
            ("dessert", "desert"): "Real-word spelling error: 'desert' (arid land) has a single 's'.",
            ("here", "hear"): "Confused homophone: use verb 'hear' (to listen/perceive sound), not 'here' (location).",
            ("hear", "here"): "Confused homophone: use adverb 'here' (in this place), not 'hear'.",
            ("bare", "bear"): "Confused homophone: use verb 'bear' (to endure) in 'bear with me' / 'cannot bear'.",
            ("bear", "bare"): "Confused homophone: use adjective 'bare' (uncovered) in 'bare hands/feet'.",
            ("dairy", "diary"): "Real-word spelling error: 'diary' is a personal journal; 'dairy' refers to milk products.",
            ("diary", "dairy"): "Real-word spelling error: 'dairy' refers to milk products, not personal journal 'diary'.",
            ("angle", "angel"): "Real-word spelling error: 'angel' is a heavenly being; 'angle' is a geometric shape.",
            ("angel", "angle"): "Real-word spelling error: 'angle' is a geometric corner, not 'angel'.",
            ("trial", "trail"): "Real-word spelling error: 'trail' is a path or hiking track; 'trial' is a court proceeding.",
            ("trail", "trial"): "Real-word spelling error: 'trial' is a judicial test or proceeding, not 'trail'.",
            ("principle", "principal"): "Confused word: 'principal' is the head of a school or main item; 'principle' is a fundamental rule.",
            ("principal", "principle"): "Confused word: 'principle' is a moral truth or law, not 'principal'.",
            ("principal", "principles"): "Contextual Word Choice: 'principles' (fundamental truths or rules of a discipline) should be used instead of 'principal'.",
            ("principals", "principles"): "Confused word: 'principles' (fundamental truths or rules) should be used instead of 'principals'.",
            ("consensus", "concentration"): "Contextual Word Choice (Malapropism): In this cognitive/study context, 'concentration' (mental focus) is required instead of 'consensus' (general agreement).",
            ("coarse", "course"): "Confused word: 'of course' requires 'course', not adjective 'coarse' (rough).",
            ("course", "coarse"): "Confused word: 'coarse' means rough or abrasive texture.",
            ("aloud", "allowed"): "Confused word: 'allowed' means permitted; 'aloud' means audibly.",
            ("allowed", "aloud"): "Confused word: 'aloud' means spoken out loud, not permitted 'allowed'.",
            ("breath", "breathe"): "Real-word error: 'breathe' is the verb (to inhale/exhale); 'breath' is the noun.",
            ("breathe", "breath"): "Real-word error: 'breath' is the noun in 'deep breath'; 'breathe' is the verb.",
            ("cloths", "clothes"): "Real-word error: 'clothes' are garments to wear; 'cloths' are cleaning rags or fabrics.",
            ("sight", "site"): "Confused word: 'site' is a place or website; 'sight' is vision.",
            ("site", "sight"): "Confused word: 'sight' is vision or view; 'site' is location.",
            ("passed", "past"): "Confused word: 'in the past' refers to previous time; 'passed' is past tense of pass.",
            ("illicit", "elicit"): "Confused word: 'elicit' is a verb meaning to draw out or evoke; 'illicit' means illegal.",
            ("elicit", "illicit"): "Confused word: 'illicit' means illegal; 'elicit' means to draw out.",
            ("allusion", "illusion"): "Confused word: 'illusion' is a false perception; 'allusion' is an indirect reference.",
            ("illusion", "allusion"): "Confused word: 'allusion' is an indirect reference; 'illusion' is a deception.",
            ("complement", "compliment"): "Confused word: 'compliment' is praise; 'complement' means completing or balancing.",
            ("compliment", "complement"): "Confused word: 'complement' means enhancing or completing; 'compliment' is praise.",
            ("discreet", "discrete"): "Confused word: 'discrete' means distinct/separate; 'discreet' means cautious.",
            ("discrete", "discreet"): "Confused word: 'discreet' means cautious; 'discrete' means separate.",
            ("flout", "flaunt"): "Confused word: 'flaunt' means to display ostentatiously; 'flout' means to openly disregard.",
            ("flaunt", "flout"): "Confused word: 'flout' means to disobey rules; 'flaunt' means to show off.",
            ("eminent", "imminent"): "Confused word: 'imminent' means impending; 'eminent' means famous.",
            ("imminent", "eminent"): "Confused word: 'eminent' means distinguished; 'imminent' means impending.",
            ("adverse", "averse"): "Confused word: 'averse' means strongly disliking; 'adverse' means harmful.",
            ("averse", "adverse"): "Confused word: 'adverse' means unfavorable; 'averse' means opposed.",
            ("perspective", "prospective"): "Confused word: 'prospective' means potential; 'perspective' means viewpoint.",
            ("prospective", "perspective"): "Confused word: 'perspective' means viewpoint; 'prospective' means potential.",
            ("assure", "ensure"): "Confused word: 'ensure' means make certain; 'assure' means remove doubt.",
        }
        if (orig_lower, corr_lower) in confused_map:
            msg = confused_map[(orig_lower, corr_lower)]
            cat = "Contextual Word Choice (Malapropism)" if "Malapropism" in msg else "Confused Word / Homophone"
            return (cat, msg, 0.96)

        # 4. Gender Collocations & Concord
        gender_map = {
            ("handsome", "beautiful"): "Gender Collocation: 'handsome' is traditionally used for men or boys. For women or girls, 'beautiful', 'pretty', or 'gorgeous' is preferred.",
            ("handsome", "pretty"): "Gender Collocation: 'handsome' is typically used for men or boys.",
            ("handsome", "gorgeous"): "Gender Collocation: 'handsome' is typically used for men or boys.",
            ("actress", "actor"): "Gender Concord: Subject is male, so masculine noun 'actor' should be used.",
            ("waitress", "waiter"): "Gender Concord: Subject is male, so masculine noun 'waiter' should be used.",
            ("actor", "actress"): "Gender Concord: Subject is female, so feminine noun 'actress' should be used.",
            ("waiter", "waitress"): "Gender Concord: Subject is female, so feminine noun 'waitress' should be used.",
        }
        if (orig_lower, corr_lower) in gender_map:
            return ("Gender Collocation / Word Choice", gender_map[(orig_lower, corr_lower)], 0.96)

        # Pronoun-Antecedent Agreement ("their" -> "its" / "its" -> "their")
        if orig_lower == "their" and corr_lower == "its":
            return (
                "Pronoun-Antecedent Agreement",
                "The collective noun antecedent is singular, requiring the singular possessive pronoun 'its' instead of plural 'their'.",
                0.96
            )
        if orig_lower == "its" and corr_lower == "their":
            return (
                "Pronoun-Antecedent Agreement",
                "Plural antecedent requires plural possessive pronoun 'their' instead of singular 'its'.",
                0.96
            )

        # Pronoun-Antecedent Gender Concord ("his" -> "her" / "her" -> "his")
        if (orig_lower, corr_lower) in (("his", "her"), ("her", "his"), ("himself", "herself"), ("herself", "himself")):
            return (
                "Pronoun-Antecedent Gender Agreement",
                f"Pronoun gender mismatch: '{orig_str}' refers to a gendered antecedent subject and must be replaced by '{corr_str}'.",
                0.97
            )

        # Distributive Pronoun Concord ("they" -> "he or she")
        if orig_lower == "they" and corr_lower == "he or she":
            return (
                "Pronoun-Antecedent Agreement",
                "Distributive Pronoun Agreement: The singular distributive subject ('Every one') formally takes singular pronoun 'he or she' instead of plural 'they'.",
                0.96
            )


        # 2. Article / Determiner errors
        if cls._is_article_error(orig_tokens, corr_tokens):
            if orig_lower in ARTICLES_DETERMINERS and corr_lower in ARTICLES_DETERMINERS:
                explanation = f"Incorrect article '{orig_str}'. Use '{corr_str}' based on phonetic vowel/consonant rules."
            elif not orig_str and corr_lower in ARTICLES_DETERMINERS:
                explanation = f"Missing determiner/article: '{corr_str}' is required here."
            else:
                explanation = f"Redundant article '{orig_str}' should be omitted."
            return ("Article / Determiner", explanation, 0.92)

        # 3. Preposition errors
        if cls._is_preposition_error(orig_tokens, corr_tokens):
            if not corr_tokens or not corr_str:
                after_info = f" before '{context_after[0]}'" if context_after else ""
                return (
                    "Redundant Preposition",
                    f"Redundant preposition: '{orig_str}' should be omitted{after_info} in standard English.",
                    0.93
                )
            return (
                "Preposition Error",
                f"Incorrect preposition '{orig_str}'. The correct idiom/collocation requires '{corr_str}'.",
                0.90
            )

        # 4. Subject-Verb Agreement
        sva_detected, sva_expl = cls._is_subject_verb_agreement(
            orig_tokens, corr_tokens, context_before, context_after
        )
        if sva_detected:
            return ("Subject-Verb Agreement", sva_expl, 0.94)

        # Third Conditional ('would have [participle]' -> 'had [participle]' in if-clause)
        if (("would have" in orig_lower or "would of" in orig_lower or "had have" in orig_lower) and "had" in corr_lower and any(w in (x.lower() for x in context_before + orig_tokens) for w in ("if", "even if", "as if"))) or \
           (orig_tokens == ["would", "have"] and corr_tokens == ["had"]) or \
           (orig_tokens == ["would"] and corr_tokens == ["had"]) or \
           (orig_lower == "would" and corr_lower == "had"):
            return (
                "Conditional Clause / Verb Form",
                "Third Conditional error: The 'if'-clause takes past perfect ('had + past participle'), not 'would have'. Use 'had' instead of 'would have'.",
                0.97
            )

        # Subjunctive Mood (was -> were)
        if orig_lower == "was" and corr_lower == "were" and any(w in (x.lower() for x in context_before) for w in ("if", "as", "wish")):
            return (
                "Subjunctive Mood",
                "Hypothetical/unreal conditional statements require the subjunctive 'were' instead of indicative 'was'.",
                0.96
            )

        # Mandative Subjunctive (demands/recommendations taking base form verb)
        if any(w in (x.lower() for x in context_before) for w in ("recommend", "recommends", "recommended", "suggest", "suggests", "suggested", "insist", "insists", "insisted", "demand", "demands", "essential", "vital", "imperative")) and (orig_lower.endswith("s") and corr_lower == orig_lower[:-1] or corr_lower == "be"):
            return (
                "Mandative Subjunctive",
                f"Clauses expressing demand, recommendation, or urgency require the subjunctive base form verb '{corr_str}'.",
                0.95
            )

        # Comparative Preposition Idioms (prefer than -> to, different than -> from)
        if orig_lower == "than" and corr_lower in ("to", "from"):
            return (
                "Comparative / Preposition Idiom",
                f"Incorrect preposition '{orig_str}' after comparative adjective/verb. Use '{corr_str}'.",
                0.95
            )


        # Modal Perfects ("could of" -> "could have", "should of" -> "should have")
        if (orig_lower in ("could of", "should of", "would of", "might of", "must of", "couldn't of", "shouldn't of", "wouldn't of") or
            (orig_lower == "of" and corr_lower == "have" and context_before and context_before[-1].lower() in ("could", "should", "would", "might", "must", "couldn't", "shouldn't", "wouldn't"))):
            return (
                "Modal Auxiliary / Verb Form",
                f"Spoken contraction error: '{orig_str}' should be written as '{corr_str}'. Use auxiliary 'have', not preposition 'of'.",
                0.96
            )

        # Passive Voice & Past Participle Form
        KNOWN_PARTICIPLES = {
            "written", "taken", "given", "broken", "chosen", "driven", "stolen", "frozen",
            "hidden", "ridden", "spoken", "woken", "done", "seen", "eaten", "fallen",
            "drawn", "flown", "blown", "thrown", "known", "grown", "torn", "worn",
            "sworn", "bitten", "shaken", "risen", "sunk", "shrunk", "begun", "drunk",
            "sung", "swum", "beaten", "forgiven", "forgotten", "proven", "gone"
        }
        prev_passive_aux = next((w.lower() for w in reversed(context_before[-3:]) if w.lower() in ("was", "were", "is", "are", "am", "been", "being", "be", "get", "gets", "got", "gotten")), None)
        if prev_passive_aux and orig_lower != corr_lower:
            if corr_lower in KNOWN_PARTICIPLES or corr_lower.endswith(("en", "ed", "ne", "wn", "t", "ne", "un")):
                return (
                    "Passive Voice / Past Participle",
                    f"Passive voice with auxiliary '{prev_passive_aux}' requires past participle form '{corr_str}', not '{orig_str}'.",
                    0.96
                )

        # Perfect Aspect Past Participle Form
        prev_perfect_aux = next((w.lower() for w in reversed(context_before[-3:]) if w.lower() in ("have", "has", "had", "having")), None)
        if prev_perfect_aux and orig_lower != corr_lower:
            if corr_lower in KNOWN_PARTICIPLES or corr_lower.endswith(("en", "ed", "ne", "wn", "t", "ne", "un")):
                return (
                    "Perfect Aspect / Past Participle",
                    f"Perfect aspect with auxiliary '{prev_perfect_aux}' requires past participle form '{corr_str}', not '{orig_str}'.",
                    0.96
                )

        # 5. Verb Tense & Form
        tense_detected, tense_expl = cls._is_verb_tense_error(
            orig_tokens, corr_tokens, context_before, context_after
        )
        if tense_detected:
            return ("Verb Tense & Form", tense_expl, 0.89)

        # 6. Quantifier / Noun Number Agreement (e.g., "every employees" -> "every employee")
        prev_quant = next((w.lower() for w in reversed(context_before[-2:]) if w.lower() in ("every", "each", "another", "one")), None)
        if prev_quant and orig_lower != corr_lower and (orig_lower.endswith("s") or corr_lower in ("employee", "student", "person", "member", "decision", "strategy")):
            return (
                "Quantifier Agreement",
                f"Quantifier '{prev_quant}' requires singular count noun '{corr_str}', not plural '{orig_str}'.",
                0.95
            )

        # 7. Spelling / Typo
        if cls._is_spelling_error(orig_str, corr_str):
            return (
                "Spelling / Typo",
                f"Probable spelling mistake: '{orig_str}' corrected to '{corr_str}'.",
                0.88
            )

        # 7. Word Order
        if cls._is_word_order_error(orig_tokens, corr_tokens):
            return (
                "Word Order",
                f"Phrasing or word order rearranged: '{orig_str}' to '{corr_str}'.",
                0.85
            )

        # 8. Missing / Extra Word
        if not orig_str and corr_str:
            return (
                "Missing Word",
                f"Missing required word/phrase: '{corr_str}'.",
                0.80
            )
        if orig_str and not corr_str:
            return (
                "Redundant Word",
                f"Redundant word '{orig_str}' should be removed.",
                0.82
            )

        # Default fallback
        return (
            "General Grammatical Error",
            f"Phrasing correction from '{orig_str}' to '{corr_str}'.",
            0.75
        )

    @classmethod
    def _is_punctuation_error(cls, orig: str, corr: str) -> bool:
        punct_pattern = r'^[^\w\s]*$'
        return bool(re.match(punct_pattern, orig) and re.match(punct_pattern, corr))

    @classmethod
    def _is_article_error(cls, orig_tokens: List[str], corr_tokens: List[str]) -> bool:
        orig_set = {t.lower() for t in orig_tokens}
        corr_set = {t.lower() for t in corr_tokens}
        
        # Swapping articles (e.g., 'a' -> 'an', 'the' -> 'a')
        if len(orig_tokens) <= 1 and len(corr_tokens) <= 1:
            if (orig_set and orig_set.issubset(ARTICLES_DETERMINERS)) or \
               (corr_set and corr_set.issubset(ARTICLES_DETERMINERS)):
                return True
        return False

    @classmethod
    def _is_preposition_error(cls, orig_tokens: List[str], corr_tokens: List[str]) -> bool:
        orig_set = {t.lower() for t in orig_tokens}
        corr_set = {t.lower() for t in corr_tokens}
        
        if len(orig_tokens) <= 1 and len(corr_tokens) <= 1:
            if (orig_set and orig_set.issubset(COMMON_PREPOSITIONS)) or \
               (corr_set and corr_set.issubset(COMMON_PREPOSITIONS)):
                return True
        return False

    @classmethod
    def _is_subject_verb_agreement(
        cls,
        orig_tokens: List[str],
        corr_tokens: List[str],
        context_before: List[str],
        context_after: List[str]
    ) -> Tuple[bool, str]:
        orig = " ".join(orig_tokens).lower()
        corr = " ".join(corr_tokens).lower()

        # Check common SVA patterns:
        sva_pairs = {
            ("have", "has"): "Singular 3rd person subjects require 'has' instead of 'have'.",
            ("has", "have"): "Plural subjects require 'have' instead of 'has'.",
            ("is", "are"): "Plural subjects require 'are' instead of 'is'.",
            ("are", "is"): "Singular subjects require 'is' instead of 'are'.",
            ("was", "were"): "Plural subjects require 'were' instead of 'was'.",
            ("don't", "doesn't"): "Singular 3rd person requires 'doesn't' instead of 'don't'.",
            ("do", "does"): "Singular 3rd person requires 'does' instead of 'do'.",
            ("does", "do"): "Plural subjects require 'do' instead of 'does'.",
        }

        if (orig, corr) == ("were", "was"):
            cb_str = " ".join(context_before).lower()
            for dp in ("neither of", "either of", "every one of", "each of", "every one", "each one", "one of"):
                if dp in cb_str:
                    head_word = dp.split()[0].capitalize()
                    return True, f"Subject-Verb Agreement (Distributive Pronoun): The head subject '{head_word}' is grammatically singular and requires singular verb 'was' instead of plural 'were' (prepositional phrase 'of the ...' does not make the subject plural)."
            if any(p in cb_str for p in ("as well as", "along with", "together with", "in addition to", "accompanied by")):
                return True, "Subject-Verb Agreement (Parenthetical Coordination): When a subject is joined with parenthetical phrases ('as well as', 'along with', 'together with', 'in addition to'), the verb agrees with the primary subject. Since the primary subject is singular, singular verb 'was' is required instead of 'were'."
            if " nor " in f" {cb_str} " or " or " in f" {cb_str} ":
                return True, "Subject-Verb Agreement (Correlative Conjunction): In 'neither... nor' / 'either... or' structures, the verb agrees with the closer subject. Since the closer subject is singular, singular verb 'was' is required instead of 'were'."
            if any(cn in cb_str for cn in ("board", "committee", "panel", "team", "group", "jury", "fleet", "series", "collection", "pair")):
                return True, "Subject-Verb Agreement (Collective Noun): The singular collective head noun requires singular verb 'was' instead of plural 'were' (prepositional phrase modifier 'of ...' does not make the subject plural)."
            return True, "Singular subjects require 'was' instead of 'were'."

        if (orig, corr) == ("was", "were"):
            cb_str = " ".join(context_before).lower()
            if "both " in cb_str and " and " in cb_str:
                return True, "Subject-Verb Agreement (Compound Subject): Compound subjects joined by 'both... and' refer to multiple entities and always require plural verb 'were' instead of singular 'was'."
            if any(p in cb_str for p in ("as well as", "along with", "together with", "in addition to", "accompanied by")):
                return True, "Subject-Verb Agreement (Parenthetical Coordination): When a subject is joined with parenthetical phrases ('along with', 'as well as'), the verb agrees with the primary subject. Since the primary subject is plural, plural verb 'were' is required."
            if any(lp in cb_str for lp in ("data", "criteria", "phenomena", "strata", "bacteria", "media", "analyses", "hypotheses", "theses")):
                return True, "Subject-Verb Agreement (Latin/Foreign Plural): In formal and academic usage, 'data' and classical loanwords are irregular plural nouns, requiring plural verb 'were' instead of singular 'was'."
            return True, "Plural subjects require 'were' instead of 'was'."


        if (orig, corr) in sva_pairs:
            return True, sva_pairs[(orig, corr)]

        # Check singular 3rd person verb ending (-s or -es): e.g., "go" -> "goes", "walk" -> "walks"
        prev_word = context_before[-1].lower() if context_before else ""
        if prev_word in SINGULAR_PRONOUNS:
            if (corr == orig + "s" or corr == orig + "es" or 
               (orig.endswith("y") and corr == orig[:-1] + "ies")):
                return True, f"Subject '{prev_word}' is 3rd-person singular, requiring verb '{corr}'."

        if prev_word in PLURAL_PRONOUNS:
            if orig.endswith("s") and corr == orig[:-1]:
                return True, f"Plural subject '{prev_word}' requires base verb '{corr}'."

        return False, ""

    @classmethod
    def _is_verb_tense_error(
        cls,
        orig_tokens: List[str],
        corr_tokens: List[str],
        context_before: List[str],
        context_after: List[str]
    ) -> Tuple[bool, str]:
        orig = " ".join(orig_tokens).lower()
        corr = " ".join(corr_tokens).lower()

        # Past tense indicators in surrounding context
        context_text = " ".join(context_before + context_after).lower()
        has_past_context = any(w in context_text for w in [
            "yesterday", "ago", "last", "earlier", "past", "previously", "once", "did"
        ])

        # Common irregular past tense pairs
        irregular_verbs = {
            "go": "went", "eat": "ate", "run": "ran", "see": "saw", "take": "took",
            "come": "came", "make": "made", "know": "knew", "think": "thought",
            "buy": "bought", "bring": "brought", "teach": "taught", "write": "wrote",
            "speak": "spoke", "give": "gave", "find": "found", "get": "got",
            "lead": "led", "meet": "met", "choose": "chose", "drive": "drove",
            "break": "broke", "fall": "fell", "feel": "felt", "keep": "kept",
            "leave": "left", "lose": "lost", "pay": "paid", "say": "said",
            "send": "sent", "spend": "spent", "tell": "told", "win": "won",
            "build": "built", "hear": "heard", "hold": "held", "stand": "stood"
        }

        if irregular_verbs.get(orig) == corr:
            return True, f"Verb tense inconsistency: '{orig}' should be past tense '{corr}'."

        if corr.endswith("ed") and (corr == orig + "ed" or corr == orig + "d"):
            return True, f"Tense error: '{orig}' should be past tense form '{corr}'."

        if has_past_context and orig != corr:
            return True, f"Sentence context indicates past timeframe, so '{orig}' must be formatted as '{corr}'."

        return False, ""

    @classmethod
    def _is_spelling_error(cls, orig: str, corr: str) -> bool:
        if not orig or not corr or " " in orig or " " in corr:
            return False
        dist = compute_levenshtein_distance(orig, corr)
        # 1 or 2 character edit distance on words longer than 3 characters is almost always a spelling typo
        return 1 <= dist <= 2 and min(len(orig), len(corr)) >= 3

    @classmethod
    def _is_word_order_error(cls, orig_tokens: List[str], corr_tokens: List[str]) -> bool:
        if len(orig_tokens) > 1 and len(corr_tokens) > 1:
            return sorted([t.lower() for t in orig_tokens]) == sorted([t.lower() for t in corr_tokens])
        return False
