"""
Syntactic & Clausal Reasoning Engine for Grammatical Error Correction.
Solves universal ESL/conversational linguistic error categories:
1. WH-Question Inversion & Interrogative Syntax (e.g. 'why this is not' -> 'why are they not', 'what you are doing' -> 'what are you doing').
2. Evaluative Clause Run-On Segmentation (e.g. '[clause] this is not good' -> '[clause]? It is not good').
3. Bare Participle Auxiliary Recovery (e.g. 'we playing' -> 'we are playing', 'they coming' -> 'they are coming').
4. Spatial Location Prepositions (e.g. 'playing the class' -> 'playing in class', 'sitting the room' -> 'sitting in the room').
"""
import re
from typing import Tuple, List, Dict, Any, Optional


class SyntacticReasoningEngine:
    """
    Evaluates and corrects fundamental syntactic structures: direct question inversion,
    location preposition requirements, bare participle auxiliaries, and evaluative run-ons.
    """

    # 1. Action verbs that require human agents (cannot take dummy demonstrative 'this' as agent)
    HUMAN_ACTION_VERBS = {
        'studying', 'study', 'playing', 'play', 'learning', 'learn', 'working', 'work',
        'reading', 'read', 'writing', 'write', 'speaking', 'speak', 'talking', 'talk',
        'listening', 'listen', 'watching', 'watch', 'running', 'run', 'walking', 'walk',
        'eating', 'eat', 'sleeping', 'sleep', 'teaching', 'teach', 'attending', 'attend',
        'coming', 'come', 'going', 'go', 'helping', 'help', 'doing', 'do', 'practicing'
    }

    # 2. Place nouns requiring spatial prepositions (in, on, at) after activity verbs
    PLACE_IN_NOUNS = r'class|classroom|room|hall|school|college|university|office|hospital|house|kitchen|library|lab|laboratory'
    PLACE_ON_NOUNS = r'ground|field|road|street|stage|floor|roof|terrace|playground|campus'
    PLACE_AT_NOUNS = r'bus\s+stop|station|airport|entrance|gate|desk|table'

    @classmethod
    def fix_question_inversion(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Fixes question inversion and human pronoun concord for direct WH-questions.
        e.g.:
        - 'why this is not studying' -> 'Why are they not studying'
        - 'why you are going' -> 'Why are you going'
        - 'what you are doing' -> 'What are you doing'
        - 'where he is living' -> 'Where is he living'
        """
        corrected = text
        changes = []

        # A. 'Why this is (not)? [human action verb]' -> 'Why are they (not)? [verb]'
        def repl_why_this(m):
            wh = m.group(1).capitalize()
            neg = m.group(2) or ''
            verb = m.group(3).lower()
            from .spellchecker import SpellChecker
            clean_verb = SpellChecker.correct_word(verb) or verb
            neg_part = 'not ' if 'not' in neg.lower() else ''
            # In English, students/people studying are referred to as 'they' (or 'he/she')
            return f"{wh} are they {neg_part}{clean_verb}"

        pat_why_this = r'\b(why|how|where|when)\s+this\s+is\s+(not\s+)?([a-z]+ing)\b'
        for m in list(re.finditer(pat_why_this, corrected, re.IGNORECASE)):
            verb = m.group(3).lower()
            from .spellchecker import SpellChecker
            clean_verb = SpellChecker.correct_word(verb) or verb
            if clean_verb in cls.HUMAN_ACTION_VERBS:
                orig = m.group(0)
                repl = repl_why_this(m)
                corrected = corrected[:m.start()] + repl + corrected[m.end():]
                changes.append({"original": orig, "replacement": repl, "type": "question_inversion"})

        # B. Direct WH question non-inversion: 'wh-word + subject + auxiliary' -> 'wh-word + auxiliary + subject'
        # e.g. 'why you are' -> 'why are you', 'what you are' -> 'what are you'
        pat_inversion = r'\b(why|what|where|when|how|who)\s+(you|they|we|he|she)\s+(are|is|were|was|can|could|will|would|should|do|does|did)\b'
        def repl_inv(m):
            wh = m.group(1)
            subj = m.group(2)
            aux = m.group(3)
            if m.start() == 0 or corrected[m.start() - 1] in '.!?\n':
                wh = wh.capitalize()
            return f"{wh} {aux} {subj}"

        for m in list(re.finditer(pat_inversion, corrected, re.IGNORECASE)):
            orig = m.group(0)
            repl = repl_inv(m)
            corrected = corrected[:m.start()] + repl + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl, "type": "question_inversion"})

        # C. Missing auxiliary in WH question: 'why you not [verb]' -> 'why are you not [verb]'
        pat_missing_aux = r'\b(why|how|where)\s+(you|they|we)\s+not\s+([a-z]+ing)\b'
        def repl_missing_aux(m):
            wh = m.group(1)
            subj = m.group(2)
            verb = m.group(3)
            if m.start() == 0:
                wh = wh.capitalize()
            return f"{wh} are {subj} not {verb}"

        for m in list(re.finditer(pat_missing_aux, corrected, re.IGNORECASE)):
            orig = m.group(0)
            repl = repl_missing_aux(m)
            corrected = corrected[:m.start()] + repl + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl, "type": "question_inversion"})

        return corrected, changes

    @classmethod
    def fix_spatial_location_prepositions(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Inserts required spatial prepositions (in, on, at) between activity verbs and location nouns.
        e.g.:
        - 'playing the class' -> 'playing in class'
        - 'sitting the room' -> 'sitting in the room'
        - 'playing the ground' -> 'playing on the ground'
        - 'waiting the bus stop' -> 'waiting at the bus stop'
        """
        corrected = text
        changes = []

        # 1. Activities in enclosed locations (class, classroom, room, library, office, school, college)
        # e.g. "playing the class" -> "playing in class", "playing class" -> "playing in class"
        pat_in = r'\b(play|plays|playing|played|sit|sits|sitting|sat|study|studies|studying|studied|read|reads|reading|talk|talks|talking|talked|work|works|working|worked|sleep|sleeps|sleeping|slept)\s+(?:the\s+)?(class|classroom|room|library|office|hall|lab|laboratory)\b'
        def repl_in(m):
            verb = m.group(1)
            loc = m.group(2)
            # In English, for 'class', idiomatically it is 'in class' (without article)
            if loc.lower() == 'class':
                return f"{verb} in class"
            return f"{verb} in the {loc}"

        for m in list(re.finditer(pat_in, corrected, re.IGNORECASE)):
            orig = m.group(0)
            repl = repl_in(m)
            corrected = corrected[:m.start()] + repl + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl, "type": "preposition_location"})

        # 2. Activities on open locations (ground, field, playground, road, street, floor)
        pat_on = r'\b(play|plays|playing|played|run|runs|running|ran|walk|walks|walking|walked|sit|sits|sitting|sat|stand|stands|standing|stood)\s+(?:the\s+)?(ground|field|playground|road|street|floor|roof)\b'
        def repl_on(m):
            verb = m.group(1)
            loc = m.group(2)
            return f"{verb} on the {loc}"

        for m in list(re.finditer(pat_on, corrected, re.IGNORECASE)):
            orig = m.group(0)
            repl = repl_on(m)
            corrected = corrected[:m.start()] + repl + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl, "type": "preposition_location"})

        # 3. Activities at point locations (bus stop, station, gate, table, desk)
        pat_at = r'\b(wait|waits|waiting|waited|stand|stands|standing|stood|sit|sits|sitting|sat)\s+(?:the\s+)?(bus\s+stop|station|gate|desk|table)\b'
        def repl_at(m):
            verb = m.group(1)
            loc = m.group(2)
            return f"{verb} at the {loc}"

        for m in list(re.finditer(pat_at, corrected, re.IGNORECASE)):
            orig = m.group(0)
            repl = repl_at(m)
            corrected = corrected[:m.start()] + repl + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl, "type": "preposition_location"})

        return corrected, changes

    @classmethod
    def fix_bare_participle_auxiliaries(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Supplies missing auxiliary verbs before bare present participles.
        e.g. 'if we playing' -> 'if we are playing' (or 'if we play')
             'they studying' -> 'they are studying'
             'he working' -> 'he is working'
        """
        corrected = text
        changes = []

        # Pattern: pronoun + V-ing without auxiliary verb
        # e.g. "if we playing" -> "if we are playing"
        pat = r'\b(if|when|while|because|although|since)?\s*(we|they|you)\s+([a-z]+ing)\b'
        def repl(m):
            conj = m.group(1)
            subj = m.group(2)
            verb = m.group(3)
            # Avoid matching if previous word was an auxiliary (e.g. 'are we playing')
            lead = f"{conj} " if conj else ""
            return f"{lead}{subj} are {verb}"

        for m in list(re.finditer(pat, corrected, re.IGNORECASE)):
            tokens_before = corrected[:m.start(2)].split()
            if tokens_before and tokens_before[-1].lower() in ('are', 'were', 'have', 'been', 'is', 'was', 'am'):
                continue
            orig = m.group(0)
            repl_str = repl(m)
            corrected = corrected[:m.start()] + repl_str + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl_str, "type": "bare_participle"})

        # Singular subjects: he/she + V-ing -> he/she is V-ing
        pat_sing = r'\b(if|when|while|because)?\s*(he|she)\s+([a-z]+ing)\b'
        def repl_sing(m):
            conj = m.group(1)
            subj = m.group(2)
            verb = m.group(3)
            lead = f"{conj} " if conj else ""
            return f"{lead}{subj} is {verb}"

        for m in list(re.finditer(pat_sing, corrected, re.IGNORECASE)):
            tokens_before = corrected[:m.start(2)].split()
            if tokens_before and tokens_before[-1].lower() in ('is', 'was', 'has', 'been', 'are'):
                continue
            orig = m.group(0)
            repl_str = repl_sing(m)
            corrected = corrected[:m.start()] + repl_str + corrected[m.end():]
            changes.append({"original": orig, "replacement": repl_str, "type": "bare_participle"})

        return corrected, changes

    @classmethod
    def split_evaluative_runon(cls, text: str) -> Tuple[str, bool]:
        """
        Segments run-on sentences joining a clause with an evaluative appraisal
        (e.g. '...English this is not good if we playing the class'
         -> '...English? It is not good if we are playing in class.')
        """
        # Evaluative adjectives
        EVAL_ADJS = r'good|bad|right|wrong|okay|acceptable|fair|safe|proper|normal|healthy|nice|fine|cool'

        # Pattern:
        # Clause 1: (Why|How|Where|What|When) ... [topic/noun]
        # Clause 2: (this|it|that) is (not )?(good|bad|right...) ...
        pat = r'^(why|what|where|when|how)\s+([^?!\n]+?)\s+(this|that|it)\s+(is|was)\s+(not\s+)?(' + EVAL_ADJS + r')(\s+[^?!\n]+)?$'
        m = re.match(pat, text.strip(), flags=re.IGNORECASE)
        if m:
            wh = m.group(1).capitalize()
            c1_body = m.group(2).strip()
            # If demonstrative 'this/that' is used as evaluation of a situation, standard English uses 'It is'
            eval_subj = "It"
            aux = m.group(4)
            neg = m.group(5) or ""
            adj = m.group(6)
            rest = m.group(7) or ""

            c1 = f"{wh} {c1_body}?"
            c2 = f"{eval_subj} {aux} {neg}{adj}{rest}".strip()
            if not c2.endswith(('.', '!', '?')):
                c2 += '.'
            return f"{c1} {c2}", True

        # Secondary: Declarative clause + Evaluative appraisal run-on
        # e.g. "we are playing the class this is not good"
        pat2 = r'^([^?!\n]+?)\s+(this|that)\s+(is|was)\s+(not\s+)?(' + EVAL_ADJS + r')(\s+[^?!\n]+)?$'
        m2 = re.match(pat2, text.strip(), flags=re.IGNORECASE)
        if m2:
            c1 = m2.group(1).strip()
            if c1 and c1[0].islower():
                c1 = c1[0].upper() + c1[1:]
            c1 += '.'
            aux = m2.group(3)
            neg = m2.group(4) or ""
            adj = m2.group(5)
            rest = m2.group(6) or ""
            c2 = f"It {aux} {neg}{adj}{rest}".strip()
            if not c2.endswith(('.', '!', '?')):
                c2 += '.'
            return f"{c1} {c2}", True

        return text, False

    @classmethod
    def apply_all(cls, text: str) -> str:
        """Applies full syntactic and clausal reasoning pipeline."""
        result = text
        # Step 1: Pre-segment evaluative run-on
        result, _ = cls.split_evaluative_runon(result)
        # Step 2: Question inversion
        result, _ = cls.fix_question_inversion(result)
        # Step 3: Spatial prepositions
        result, _ = cls.fix_spatial_location_prepositions(result)
        # Step 4: Bare participle auxiliaries
        result, _ = cls.fix_bare_participle_auxiliaries(result)
        return result
