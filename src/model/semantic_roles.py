"""
Semantic Role & Frame Semantics Engine for Grammatical Error Correction.
Enforces linguistic frame consistency including:
1. Agent-Action Semantic Roles (e.g. In educational frames: Teachers teach, students study).
2. Motion Destination Prepositions (e.g. 'going to school' vs incorrect 'going in the school').
3. Conversational Interrogative + Perception Imperative Clause Splitting (e.g. WH-questions run-on with 'see / look').
"""
import re
from typing import Tuple, List, Dict, Any, Optional


class SemanticRoleEngine:
    """
    Evaluates and enforces semantic role concord, institution motion prepositions,
    and conversational interrogative clause boundaries.
    """

    # 1. Motion prepositions to functional institutions/destinations
    # In English, movement towards functional institutions uses preposition 'to' (and omits 'the' when referring to primary purpose)
    INSTITUTION_MOTION_RULES = [
        # School
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?school\b', r'\1 to school'),
        (r'\b(head|heads|heading|headed)\s+in\s+(?:the\s+)?school\b', r'\1 to school'),
        (r'\b(walk|walks|walking|walked)\s+in\s+(?:the\s+)?school\b', r'\1 to school'),
        (r'\b(travel|travels|traveling|traveled)\s+in\s+(?:the\s+)?school\b', r'\1 to school'),
        
        # College / University
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?college\b', r'\1 to college'),
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?university\b', r'\1 to university'),
        
        # Church / Mosque / Temple / Synagogue
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?church\b', r'\1 to church'),
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?temple\b', r'\1 to temple'),
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?mosque\b', r'\1 to mosque'),
        
        # Bed (sleeping context)
        (r'\b(go|goes|going|went|gone)\s+in\s+(?:the\s+)?bed\b', r'\1 to bed'),
        
        # Hospital
        (r'\b(go|goes|going|went|gone)\s+in\s+hospital\b', r'\1 to hospital'),
        (r'\b(go|goes|going|went|gone)\s+in\s+the\s+hospital\b', r'\1 to the hospital'),
        
        # Market / Work
        (r'\b(go|goes|going|went|gone)\s+in\s+work\b', r'\1 to work'),
        (r'\b(go|goes|going|went|gone)\s+in\s+the\s+market\b', r'\1 to the market'),
    ]

    # 2. Agent-Action Semantic Role Mismatches (Frame Semantics / VerbNet)
    # Educational Frame:
    # Agent: Teacher / Professor / Instructor -> Action: Teach / Lecture / Instruct / Explain
    # Agent: Student / Pupil / Learner -> Action: Study / Learn / Revise
    EDUCATIONAL_AGENT_RULES = [
        # Teacher + studying -> teaching
        (r'\b(the\s+)?(teacher|prof|professor|instructor|tutor)\s+(?:is|was)\s+(?:not\s+)?studying\b',
         lambda m: m.group(0).replace("studying", "teaching")),
        (r'\b(the\s+)?(teachers|professors|instructors|tutors)\s+(?:are|were)\s+(?:not\s+)?studying\b',
         lambda m: m.group(0).replace("studying", "teaching")),
        (r'\b(the\s+)?(teacher|prof|professor|instructor|tutor)\s+studies\b',
         lambda m: m.group(0).replace("studies", "teaches")),
        (r'\b(the\s+)?(teacher|prof|professor|instructor|tutor)\s+studied\b',
         lambda m: m.group(0).replace("studied", "taught")),
        (r'\b(the\s+)?(teacher|prof|professor|instructor|tutor)\s+does\s+not\s+study\b',
         lambda m: m.group(0).replace("study", "teach")),
        (r'\b(the\s+)?(teacher|prof|professor|instructor|tutor)\s+did\s+not\s+study\b',
         lambda m: m.group(0).replace("study", "teach")),

        # Student + teaching -> studying (or learning)
        (r'\b(the\s+)?(student|pupil|learner)\s+(?:is|was)\s+(?:not\s+)?teaching\b',
         lambda m: m.group(0).replace("teaching", "studying")),
        (r'\b(the\s+)?(students|pupils|learners)\s+(?:are|were)\s+(?:not\s+)?teaching\b',
         lambda m: m.group(0).replace("teaching", "studying")),
        (r'\b(the\s+)?(student|pupil|learner)\s+teaches\b',
         lambda m: m.group(0).replace("teaches", "studies")),
        (r'\b(the\s+)?(student|pupil|learner)\s+taught\b',
         lambda m: m.group(0).replace("taught", "studied")),
    ]

    # Medical Frame: Doctor treats/prescribes; Patient takes medicine/recovers
    MEDICAL_AGENT_RULES = [
        (r'\b(the\s+)?(doctor|surgeon|physician)\s+(?:is|was)\s+(?:not\s+)?taking\s+medicine\s+from\s+the\s+patient\b',
         lambda m: m.group(0).replace("taking medicine from", "giving medicine to")),
    ]

    @classmethod
    def apply_motion_prepositions(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Applies destination motion prepositions for functional institutions.
        """
        corrected = text
        changes = []
        for pat, repl in cls.INSTITUTION_MOTION_RULES:
            matches = list(re.finditer(pat, corrected, flags=re.IGNORECASE))
            if matches:
                for m in reversed(matches):
                    orig_span = m.group(0)
                    new_span = re.sub(pat, repl, orig_span, flags=re.IGNORECASE)
                    if orig_span[0].isupper():
                        new_span = new_span[0].upper() + new_span[1:]
                    corrected = corrected[:m.start()] + new_span + corrected[m.end():]
                    changes.append({"original": orig_span, "replacement": new_span, "type": "preposition"})
        return corrected, changes

    @classmethod
    def apply_agent_action_roles(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Enforces semantic agent-action role consistency (e.g. teacher teaching vs student studying).
        """
        corrected = text
        changes = []
        for pat, repl_fn in cls.EDUCATIONAL_AGENT_RULES + cls.MEDICAL_AGENT_RULES:
            matches = list(re.finditer(pat, corrected, flags=re.IGNORECASE))
            if matches:
                for m in reversed(matches):
                    orig_span = m.group(0)
                    new_span = repl_fn(m)
                    if orig_span[0].isupper():
                        new_span = new_span[0].upper() + new_span[1:]
                    corrected = corrected[:m.start()] + new_span + corrected[m.end():]
                    changes.append({"original": orig_span, "replacement": new_span, "type": "semantic_role"})
        return corrected, changes

    @classmethod
    def split_interrogative_runon(cls, text: str) -> Tuple[str, bool]:
        """
        Detects run-on conversational sentences combining an interrogative question
        with a perception imperative or observation clause without proper punctuation.
        Example:
        'why are you going in the school see the teacher is not studying'
        -> 'Why are you going to school? See, the teacher is not teaching.'
        """
        # Pattern:
        # Clause 1: (Why|Where|When|How|What|Who) (are|is|do|did|will|can) you ...
        # Boundary word: (see|look|notice|listen)
        # Clause 2: the/a ...
        pat = r'^(why|where|when|how|what|who)\s+(are|is|do|did|will|can|could|should|would)\s+(you|we|they|he|she)\s+([^?!\n]+?)\s+(see|look|listen|watch)\s+((?:the|a|an|that|this|my|your|our|their|[a-zA-Z]+)\s+[^?!\n]+)$'
        m = re.match(pat, text.strip(), flags=re.IGNORECASE)
        if m:
            wh_word = m.group(1).capitalize()
            aux = m.group(2)
            subj = m.group(3)
            c1_body = m.group(4).strip()
            imperative = m.group(5).capitalize()
            c2_body = m.group(6).strip()

            c1 = f"{wh_word} {aux} {subj} {c1_body}?"
            c2 = f"{imperative}, {c2_body}"
            if not c2.endswith(('.', '!', '?')):
                c2 += '.'
            return f"{c1} {c2}", True

        # Secondary Pattern: Declarative + Imperative run-on without punctuation
        # e.g. "you are going to school see the teacher is not teaching"
        pat2 = r'^([a-zA-Z\s]+?\b(?:school|college|class|home|work|office|there|here))\s+(see|look|listen)\s+((?:the|a|an|that|this|my|your|our|their|[a-zA-Z]+)\s+[^?!\n]+)$'
        m2 = re.match(pat2, text.strip(), flags=re.IGNORECASE)
        if m2:
            c1 = m2.group(1).strip()
            if c1 and c1[0].islower():
                c1 = c1[0].upper() + c1[1:]
            c1 += '.'
            imperative = m2.group(2).capitalize()
            c2_body = m2.group(3).strip()
            c2 = f"{imperative}, {c2_body}"
            if not c2.endswith(('.', '!', '?')):
                c2 += '.'
            return f"{c1} {c2}", True

        return text, False

    @classmethod
    def apply_all(cls, text: str) -> str:
        """Runs complete semantic role, preposition, and clause boundary pipeline."""
        result = text
        # Step 1: Prepositions
        result, _ = cls.apply_motion_prepositions(result)
        # Step 2: Agent-action roles
        result, _ = cls.apply_agent_action_roles(result)
        # Step 3: Clause run-on split
        result, _ = cls.split_interrogative_runon(result)
        return result
