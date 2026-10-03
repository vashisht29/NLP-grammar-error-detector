"""
Deep Learning Grammatical Error Detection (GED) Inference Engine.
Combines Transformer Seq2Seq Generation with Token-Level Diff Alignment
to pinpoint errors, classify categories, and generate structured error reports.
"""
import time
import re
import difflib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

from .config import ModelConfig
from .classifier import ErrorClassifier
from .spellchecker import SpellChecker
import os
from .homophones import HomophoneEngine
from .gender_collocations import GenderCollocationEngine
from .chat_normalizer import ChatNormalizer
from .lexicon import LexicalSemantics
from .language_detector import HinglishDetector
from .semantic_roles import SemanticRoleEngine
from .syntactic_engine import SyntacticReasoningEngine


@dataclass
class GrammarError:
    """Individual grammatical error located inside a sentence."""
    error_id: int
    original_text: str
    original_span: Tuple[int, int]  # (start_char, end_char)
    original_tokens: List[str]
    suggested_text: str
    suggested_tokens: List[str]
    error_type: str
    explanation: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    """Complete grammatical error detection and correction result."""
    original_sentence: str
    is_grammatically_correct: bool
    error_count: int
    corrected_sentence: str
    errors: List[GrammarError]
    overall_confidence: float
    processing_time_ms: float
    model_backend: str
    is_hinglish: bool = False

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["errors"] = [e.to_dict() for e in self.errors]
        res["is_hinglish"] = self.is_hinglish
        return res


PASSIVE_PARTICIPLES: Dict[str, str] = {
    "wrote": "written", "write": "written",
    "took": "taken", "take": "taken",
    "gave": "given", "give": "given",
    "broke": "broken", "break": "broken",
    "chose": "chosen", "choose": "chosen",
    "drove": "driven", "drive": "driven",
    "stole": "stolen", "steal": "stolen",
    "froze": "frozen", "freeze": "frozen",
    "hid": "hidden", "hide": "hidden",
    "rode": "ridden", "ride": "ridden",
    "spoke": "spoken", "speak": "spoken",
    "woke": "woken", "wake": "woken",
    "did": "done",
    "saw": "seen", "see": "seen",
    "ate": "eaten", "eat": "eaten",
    "fell": "fallen", "fall": "fallen",
    "drew": "drawn", "draw": "drawn",
    "flew": "flown", "fly": "flown",
    "blew": "blown", "blow": "blown",
    "threw": "thrown", "throw": "thrown",
    "knew": "known", "know": "known",
    "grew": "grown", "grow": "grown",
    "tore": "torn", "tear": "torn",
    "wore": "worn", "wear": "worn",
    "swore": "sworn", "swear": "sworn",
    "bit": "bitten", "bite": "bitten",
    "shook": "shaken", "shake": "shaken",
    "rose": "risen", "rise": "risen",
    "sank": "sunk", "sink": "sunk",
    "shrank": "shrunk", "shrink": "shrunk",
    "began": "begun", "begin": "begun",
    "drank": "drunk", "drink": "drunk",
    "sang": "sung", "sing": "sung",
    "swam": "swum", "swim": "swum",
    "beat": "beaten",
    "forgave": "forgiven", "forgive": "forgiven",
    "forgot": "forgotten", "forget": "forgotten",
    "prove": "proven", "prooved": "proven",
    "make": "made",
    "build": "built",
    "send": "sent",
    "spend": "spent",
    "keep": "kept",
    "find": "found",
    "tell": "told",
    "hold": "held",
    "leave": "left",
    "lose": "lost",
    "pay": "paid",
    "bring": "brought",
    "buy": "bought",
    "catch": "caught",
    "fight": "fought",
    "teach": "taught",
    "think": "thought",
}

PERFECT_PARTICIPLES: Dict[str, str] = {
    "went": "gone", "go": "gone",
    "saw": "seen", "see": "seen",
    "ate": "eaten", "eat": "eaten",
    "wrote": "written", "write": "written",
    "took": "taken", "take": "taken",
    "gave": "given", "give": "given",
    "did": "done", "do": "done",
    "ran": "run",
    "came": "come",
    "began": "begun", "begin": "begun",
    "drank": "drunk", "drink": "drunk",
    "fell": "fallen", "fall": "fallen",
    "broke": "broken", "break": "broken",
    "chose": "chosen", "choose": "chosen",
    "stole": "stolen", "steal": "stolen",
    "drove": "driven", "drive": "driven",
    "swore": "sworn", "swear": "sworn",
    "wore": "worn", "wear": "worn",
    "tore": "torn", "tear": "torn",
    "bit": "bitten", "bite": "bitten",
    "flew": "flown", "fly": "flown",
    "blew": "blown", "blow": "blown",
    "drew": "drawn", "draw": "drawn",
    "spoke": "spoken", "speak": "spoken",
    "froze": "frozen", "freeze": "frozen",
    "hid": "hidden", "hide": "hidden",
    "rode": "ridden", "ride": "ridden",
    "sang": "sung", "sing": "sung",
    "swam": "swum", "swim": "swum",
    "sank": "sunk", "sink": "sunk",
    "shook": "shaken", "shake": "shaken",
    "rose": "risen", "rise": "risen",
    "shrank": "shrunk", "shrink": "shrunk",
    "forgave": "forgiven", "forgive": "forgiven",
    "forgot": "forgotten", "forget": "forgotten",
}


class DeepGrammarDetector:
    """
    Deep Learning Grammatical Error Detection & Correction Pipeline.
    Supports PyTorch + Hugging Face Transformers with automatic device placement (MPS/CUDA/CPU).
    Includes an embedded offline heuristic engine so the system works even before weights are fetched.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = self.config.resolve_device()
        self.model = None
        self.tokenizer = None
        self.backend = "uninitialized"
        self._cache: Dict[str, DetectionResult] = {}
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Attempts to load the Transformer model; gracefully falls back if dependencies/weights are not present."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            print(f"[DeepGrammarDetector] Loading Transformer model '{self.config.model_name}' on device '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            
            if self.device in ("mps", "cuda"):
                self.model = self.model.to(self.device)
            elif self.device == "cpu":
                torch.set_num_threads(4)
                
            self.model.eval()
            self.backend = f"HuggingFace Transformer ({self.config.model_name} on {self.device})"
            print(f"[DeepGrammarDetector] Successfully loaded {self.backend}.")

            # Pre-warm neural inference kernels to eliminate first-query latency spike
            try:
                _ = self._generate_correction("Hello world")
                print("[DeepGrammarDetector] Inference kernels pre-warmed and ready.")
            except Exception:
                pass
        except Exception as e:
            print(f"[DeepGrammarDetector] Note: Transformer model load skipped ({e}). Activating internal rule-augmented engine.")
            self.backend = "Rule-Augmented Neural Heuristic Engine"

    def detect(self, sentence: str) -> DetectionResult:
        """
        Detects grammatical errors in an input sentence.
        
        Args:
            sentence: Input English sentence.
            
        Returns:
            DetectionResult with error tokens, categories, explanations, and corrected text.
        """
        start_time = time.perf_counter()
        clean_sentence = sentence.strip()

        if not clean_sentence:
            return DetectionResult(
                original_sentence=sentence,
                is_grammatically_correct=True,
                error_count=0,
                corrected_sentence="",
                errors=[],
                overall_confidence=1.0,
                processing_time_ms=0.0,
                model_backend=self.backend
            )

        # 0. Hinglish / Non-English Language Detection Guardrail
        is_hinglish, hinglish_conf, hinglish_markers = HinglishDetector.check_hinglish(clean_sentence)
        if is_hinglish:
            return DetectionResult(
                original_sentence=clean_sentence,
                is_grammatically_correct=False,
                error_count=0,
                corrected_sentence="",
                errors=[],
                overall_confidence=round(hinglish_conf, 2),
                processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                model_backend=self.backend,
                is_hinglish=True
            )

        # Check if an Independent Remote/Local LLM Backend Server is active
        llm_backend_url = os.environ.get("LLM_BACKEND_URL", "").strip().rstrip("/")
        if llm_backend_url:
            try:
                import urllib.request
                import json
                req = urllib.request.Request(
                    f"{llm_backend_url}/api/detect",
                    data=json.dumps({"sentence": clean_sentence}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "GED-Client"}
                )
                with urllib.request.urlopen(req, timeout=4.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        err_objs = []
                        for idx, err in enumerate(data.get("errors", []), 1):
                            err_objs.append(GrammarError(
                                error_id=err.get("error_id", idx),
                                original_text=err.get("original_text", ""),
                                original_span=tuple(err.get("original_span", [0, 0])),
                                original_tokens=err.get("original_tokens", []),
                                suggested_text=err.get("suggested_text", ""),
                                suggested_tokens=err.get("suggested_tokens", []),
                                error_type=err.get("error_type", "Grammar Error"),
                                explanation=err.get("explanation", ""),
                                confidence=err.get("confidence", 0.95)
                            ))
                        return DetectionResult(
                            original_sentence=data.get("original_sentence", clean_sentence),
                            is_grammatically_correct=data.get("is_grammatically_correct", len(err_objs) == 0),
                            error_count=data.get("error_count", len(err_objs)),
                            corrected_sentence=data.get("corrected_sentence", clean_sentence),
                            errors=err_objs,
                            overall_confidence=data.get("overall_confidence", 0.95),
                            processing_time_ms=data.get("processing_time_ms", 15.0),
                            model_backend=data.get("model_backend", "Independent LLM Backend (Remote/Local GPU)"),
                            is_hinglish=data.get("is_hinglish", False)
                        )
            except Exception:
                pass

        if clean_sentence in self._cache:
            cached = self._cache[clean_sentence]
            return DetectionResult(
                original_sentence=cached.original_sentence,
                is_grammatically_correct=cached.is_grammatically_correct,
                error_count=cached.error_count,
                corrected_sentence=cached.corrected_sentence,
                errors=cached.errors,
                overall_confidence=cached.overall_confidence,
                processing_time_ms=0.5,
                model_backend=cached.model_backend
            )

        # 1. Generate corrected sentence using Deep Learning model (or heuristic fallback)
        has_double_quotes = clean_sentence.startswith('"') and clean_sentence.endswith('"') and len(clean_sentence) >= 2
        has_single_quotes = clean_sentence.startswith("'") and clean_sentence.endswith("'") and len(clean_sentence) >= 2

        if has_double_quotes:
            inner_sentence = clean_sentence[1:-1].strip()
            inner_corrected = self._generate_correction(inner_sentence)
            inner_clean = inner_corrected.rstrip('.!?')
            corrected_sentence = f'"{inner_clean}."'
        elif has_single_quotes:
            inner_sentence = clean_sentence[1:-1].strip()
            inner_corrected = self._generate_correction(inner_sentence)
            inner_clean = inner_corrected.rstrip('.!?')
            corrected_sentence = f"'{inner_clean}.'"
        else:
            corrected_sentence = self._generate_correction(clean_sentence)

        # 2. Align tokens between original and corrected sentence to locate exact error spans
        errors = self._align_and_extract_errors(clean_sentence, corrected_sentence)

        is_correct = len(errors) == 0
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Calculate average confidence
        if errors:
            avg_conf = sum(e.confidence for e in errors) / len(errors)
        else:
            avg_conf = 0.98 if is_correct else 0.85

        result = DetectionResult(
            original_sentence=clean_sentence,
            is_grammatically_correct=is_correct,
            error_count=len(errors),
            corrected_sentence=corrected_sentence,
            errors=errors,
            overall_confidence=round(avg_conf, 3),
            processing_time_ms=round(elapsed_ms, 2),
            model_backend=self.backend
        )
        if len(self._cache) < 2000:
            self._cache[clean_sentence] = result
        return result

    def _split_into_sentences(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Splits text into individual sentences while preserving whitespace and punctuation delimiters.
        Handles common abbreviations (e.g., Dr., Mr., Inc., e.g., i.e.) without incorrect fragmentation.
        """
        ABBREVS = {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'inc.', 'corp.', 'ltd.', 'e.g.', 'i.e.', 'etc.', 'vs.', 'no.', 'fig.', 'al.', 'dept.', 'u.s.', 'u.k.'
        }
        raw_parts = re.split(r'((?:(?<=[.!?])|(?<=[.!?][\"\']))\s+)', text)
        sentences = []
        delimiters = []

        current_sent = ""
        for i in range(0, len(raw_parts), 2):
            chunk = raw_parts[i]
            delim = raw_parts[i + 1] if i + 1 < len(raw_parts) else ""

            test_chunk = (current_sent + chunk).strip()
            last_word = test_chunk.split()[-1].lower() if test_chunk.split() else ""

            if last_word in ABBREVS and delim:
                current_sent += chunk + delim
            else:
                sentences.append((current_sent + chunk).strip())
                delimiters.append(delim)
                current_sent = ""
        return sentences, delimiters

    def _generate_correction(self, text: str) -> str:
        """Generates grammatical correction via Neural-Hybrid Pipeline with paragraph-level sentence chunking."""
        # 0. Pre-segment conversational run-on questions and evaluative appraisals
        segmented_text, was_split1 = SemanticRoleEngine.split_interrogative_runon(text)
        segmented_text, was_split2 = SyntacticReasoningEngine.split_evaluative_runon(segmented_text)
        was_split = was_split1 or was_split2

        sentences, delimiters = self._split_into_sentences(segmented_text)
        if len(sentences) <= 1 and not was_split:
            return self._generate_single_sentence_correction(text)

        # Multi-sentence paragraph or segmented run-on: process each sentence individually
        corrected_sentences = []
        for s in sentences:
            if s:
                corr_s = self._generate_single_sentence_correction(s)
                corrected_sentences.append(corr_s)
            else:
                corrected_sentences.append("")

        return "".join(c + d for c, d in zip(corrected_sentences, delimiters))

    def _generate_single_sentence_correction(self, sentence: str) -> str:
        """Corrects an individual sentence via Neural-Hybrid Pipeline (Transformer Seq2Seq + NLP Linguistic Engine)."""
        clean_sentence = sentence.strip()
        if not clean_sentence:
            return sentence

        has_double_quotes = clean_sentence.startswith('"') and clean_sentence.endswith('"') and len(clean_sentence) >= 2
        has_single_quotes = clean_sentence.startswith("'") and clean_sentence.endswith("'") and len(clean_sentence) >= 2

        if has_double_quotes:
            inner_sentence = clean_sentence[1:-1].strip()
            inner_corrected = self._run_single_sentence_pipeline(inner_sentence)
            inner_clean = inner_corrected.rstrip('.!?')
            return f'"{inner_clean}."'
        elif has_single_quotes:
            inner_sentence = clean_sentence[1:-1].strip()
            inner_corrected = self._run_single_sentence_pipeline(inner_sentence)
            inner_clean = inner_corrected.rstrip('.!?')
            return f"'{inner_clean}.'"
        else:
            return self._run_single_sentence_pipeline(clean_sentence)

    def _run_single_sentence_pipeline(self, sentence: str) -> str:
        # Step 1: Pre-process with NLP Linguistic & Spell-Checking Engine
        normalized = self._heuristic_correction(sentence)

        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                # Prefix prompt for T5 grammar models
                input_text = f"gec: {normalized}" if "t5" in self.config.model_name.lower() else normalized
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.config.max_length, truncation=True)

                if self.device in ("mps", "cuda"):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                token_count = inputs["input_ids"].shape[1]
                gen_max_len = min(self.config.max_length, max(token_count + 14, 24))

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=gen_max_len,
                        num_beams=2,
                        early_stopping=True
                    )
                neural_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                neural_output = re.sub(r"\b(what|that|it|there|who|how|here)'s(up|a|an|the|good|ready|nice|coming)\b", r"\1's \2", neural_output, flags=re.I)

                # Neural Hallucination Guard: Ensure Transformer never invents non-existent English words
                words = re.findall(r'\b[A-Za-z]+\b', neural_output)
                for w in words:
                    if not SpellChecker.is_valid_word(w):
                        cand = SpellChecker.correct_word(w)
                        if cand:
                            neural_output = re.sub(r'\b' + re.escape(w) + r'\b', cand, neural_output)

                # Protect subject pronoun consistency (prevent neural model from distorting 'you' to 'we' / 'I')
                if re.search(r'\byou\b', sentence, re.I) and not re.search(r'\bwe\b', sentence, re.I):
                    neural_output = re.sub(r'\b(are|were|do|did|can|could|will|would|should)\s+we\b', r'\1 you', neural_output, flags=re.I)
                    neural_output = re.sub(r'\bwe\s+(are|were|have|can|could|will|would|should)\b', r'you \1', neural_output, flags=re.I)

                # Protect relative clause structure (prevent neural model from dropping relative pronoun clause 'who was/is')
                if re.search(r'\b(which|who)\s+(was|were|is|are)\b', sentence, re.I):
                    aux_tense = 'was' if 'was' in sentence.lower() else ('were' if 'were' in sentence.lower() else ('is' if 'is' in sentence.lower() else 'are'))
                    neural_output = re.sub(
                        r'\b((?:the\s+|a\s+|an\s+|this\s+|that\s+)?[a-zA-Z\-]+\s+(?:student|teacher|person|people|man|men|woman|women|boy|boys|girl|girls|child|children|friend|friends))\s+([a-zA-Z]+ing)\b',
                        rf'\1 who {aux_tense} \2',
                        neural_output,
                        flags=re.I
                    )

                # Step 2: Post-process to ensure all linguistic constraints are preserved
                return self._heuristic_correction(neural_output)
            except Exception as e:
                print(f"[DeepGrammarDetector] Transformer inference error ({e}), falling back to heuristic.")

        # Fallback if model weights not active
        return normalized

    def _heuristic_correction(self, text: str) -> str:
        corrected = text
        corrected = re.sub(r"\b(what|that|it|there|who|how|here)'s(up|a|an|the|good|ready|nice|coming)\b", r"\1's \2", corrected, flags=re.I)
        # 0. Conversational Chat & Informal Text Normalization
        corrected, _ = ChatNormalizer.normalize(corrected)

        # 1. Confused Words / Contextual Homophones (accept vs except, affect vs effect, it's vs its, etc.)
        corrected, _ = HomophoneEngine.apply(corrected)

        # 2. Gender Collocations & Semantic Concord ('she is handsome' -> 'she is beautiful')
        corrected, _ = GenderCollocationEngine.apply(corrected)

        # 3. Spelling correction across words
        def fix_spelling(m):
            w = m.group()
            fixed = SpellChecker.correct_word(w)
            return fixed if fixed else w
        corrected = re.sub(r'\b[A-Za-z]+\b', fix_spelling, corrected)

        # 3b. Frame Semantics, Agent-Action Roles & Motion Destination Prepositions
        corrected = SemanticRoleEngine.apply_all(corrected)

        # 3c. Syntactic Reasoning: Question Inversion, Spatial Prepositions & Bare Participles
        corrected = SyntacticReasoningEngine.apply_all(corrected)

        # 4. Modal Auxiliary Perfects ("could of" -> "could have", "should of" -> "should have")
        modal_perfect_patterns = [
            (r'\b(could|should|would|might|must)\s+of\b', r'\1 have'),
            (r'\b(couldn\'?t|shouldn\'?t|wouldn\'?t|mightn\'?t|mustn\'?t)\s+of\b', r'\1 have'),
            (r'\bought\s+to\s+of\b', r'ought to have'),
        ]
        for pat, repl in modal_perfect_patterns:
            corrected = re.sub(pat, repl, corrected, flags=re.IGNORECASE)

        # 5. Passive Voice Past Participles ("was wrote" -> "was written", "were took" -> "were taken")
        def fix_passive(m):
            aux = m.group(1)
            adv = m.group(2) or ''
            verb = m.group(3)
            target = PASSIVE_PARTICIPLES.get(verb.lower(), verb)
            if verb[0].isupper():
                target = target.capitalize()
            sep = ' ' if adv else ''
            return f'{aux} {adv}{sep}{target}'

        corrected = re.sub(
            r'\b(was|were|is|are|am|been|being|be|got|gets|gotten)\s+(?:(\w+ly)\s+)?([A-Za-z]+)\b',
            fix_passive,
            corrected,
            flags=re.IGNORECASE
        )

        # 6. Perfect Aspect Participles ("had went" -> "had gone", "have saw" -> "have seen")
        def fix_perfect(m):
            aux = m.group(1)
            adv = m.group(2) or ''
            verb = m.group(3)
            target = PERFECT_PARTICIPLES.get(verb.lower(), verb)
            if verb[0].isupper():
                target = target.capitalize()
            sep = ' ' if adv else ''
            return f'{aux} {adv}{sep}{target}'

        corrected = re.sub(
            r'\b(has|have|had|having)\s+(?:(\w+ly|already|never|ever|just)\s+)?([A-Za-z]+)\b',
            fix_perfect,
            corrected,
            flags=re.IGNORECASE
        )

        # 7. Quantifier Agreement: every / each + singular count noun (run before SVA)
        AUXILIARY_VERBS_SET = {'was', 'were', 'is', 'are', 'has', 'have', 'had', 'does', 'do', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must'}
        def fix_every_each_noun(m):
            quant = m.group(1)
            noun = m.group(2)
            if noun.lower() in AUXILIARY_VERBS_SET:
                return m.group(0)
            if noun.lower().endswith('ies'):
                singular = noun[:-3] + ('y' if noun[-1].islower() else 'Y')
            elif noun.lower().endswith('es') and noun.lower()[:-2] in ('branch', 'dish', 'box', 'watch', 'glass', 'class', 'bus'):
                singular = noun[:-2]
            elif noun.lower().endswith('s') and not noun.lower().endswith(('ss', 'us', 'is')):
                singular = noun[:-1]
            else:
                return m.group(0)
            return f'{quant} {singular}'

        def fix_every_each_adj_noun(m):
            quant = m.group(1)
            adj = m.group(2)
            noun = m.group(3)
            if noun.lower() in AUXILIARY_VERBS_SET:
                return m.group(0)
            if noun.lower().endswith('ies'):
                singular = noun[:-3] + ('y' if noun[-1].islower() else 'Y')
            elif noun.lower().endswith('es') and noun.lower()[:-2] in ('branch', 'dish', 'box', 'watch', 'glass', 'class', 'bus'):
                singular = noun[:-2]
            elif noun.lower().endswith('s') and not noun.lower().endswith(('ss', 'us', 'is')):
                singular = noun[:-1]
            else:
                return m.group(0)
            return f'{quant} {adj} {singular}'

        corrected = re.sub(r'\b(every|each)\s+([A-Za-z]+)\s+([A-Za-z]+s)\b', fix_every_each_adj_noun, corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b(every|each)\s+([A-Za-z]+s)\b', fix_every_each_noun, corrected, flags=re.IGNORECASE)

        # Cardinal Quantifier Agreement: two/three/many/several + plural count noun (e.g. "two friend" -> "two friends")
        def fix_cardinal_plural(txt):
            cardinals = r'two|three|four|five|six|seven|eight|nine|ten|several|many|few|both'
            NON_PLURAL_TARGETS = {
                'people', 'children', 'men', 'women', 'sheep', 'deer', 'fish', 'aircraft', 'species', 'series',
                'news', 'information', 'advice', 'equipment', 'furniture', 'luggage', 'baggage', 'homework',
                'money', 'time', 'years', 'months', 'days', 'hours', 'minutes', 'seconds', 'is', 'are', 'was',
                'were', 'has', 'have', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should',
                'may', 'might', 'must', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or', 'but'
            }
            COMMON_ADJECTIVES = {
                'alternative', 'different', 'other', 'new', 'old', 'good', 'bad', 'great', 'small', 'large',
                'high', 'low', 'young', 'important', 'major', 'political', 'social', 'economic', 'financial',
                'recent', 'possible', 'special', 'clear', 'main', 'true', 'certain', 'free', 'strong', 'best',
                'better', 'hard', 'real', 'short', 'single', 'private', 'wrong', 'fine', 'common', 'poor', 'rich',
                'close'
            }
            boundary_tokens = (
                r'is|are|was|were|has|have|had|do|does|did|will|would|can|could|should|may|might|must|'
                r'lead|led|went|came|said|told|thought|seen|found|made|took|gave|know|knew|tests|'
                r'of|in|on|at|to|for|with|by|from|about|into|through|during|before|after|above|below|between|under|since|without|against|among|'
                r'that|which|who|whom|whose|where|when|why|how|because|although|though|if|unless|while|'
                r'and|or|but|nor|yet|so'
            )
            def pluralize_word(noun):
                n_low = noun.lower()
                if n_low in NON_PLURAL_TARGETS or n_low in AUXILIARY_VERBS_SET or n_low in COMMON_ADJECTIVES:
                    return noun
                if n_low.endswith(('s', 'ss', 'us', 'is', 'ics')):
                    return noun
                if n_low.endswith(('sh', 'ch', 'x', 'z')):
                    return noun + 'es'
                elif n_low.endswith('y') and len(noun) > 1 and n_low[-2] not in 'aeiou':
                    return noun[:-1] + ('ies' if noun[-1].islower() else 'IES')
                else:
                    return noun + ('s' if noun[-1].islower() else 'S')

            def repl(m):
                card = m.group(1)
                phrase = m.group(2).strip()
                words = phrase.split()
                if not words:
                    return m.group(0)
                last_word = words[-1]
                adjectives = words[:-1]
                pl = pluralize_word(last_word)
                joined = ' '.join(adjectives + [pl])
                return f'{card} {joined}'

            pat = r'\b(' + cardinals + r')\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b(?=\s+(?:' + boundary_tokens + r')|[.,;!?\"\']|$)'
            return re.sub(pat, repl, txt, flags=re.IGNORECASE)

        corrected = fix_cardinal_plural(corrected)

        # 8. Subject-Verb Agreement corrections
        # Plural subjects followed by singular auxiliary:
        SINGULAR_S_NOUNS = {
            'news', 'series', 'species', 'gas', 'bus', 'lens',
            'physics', 'mathematics', 'economics', 'politics', 'statistics', 'optics', 'genetics', 'mechanics', 'ethics',
            'analysis', 'crisis', 'thesis', 'basis', 'hypothesis', 'synthesis', 'diagnosis', 'prognosis', 'parenthesis',
            'status', 'focus', 'campus', 'census', 'apparatus', 'syllabus', 'radius',
            'business', 'process', 'success', 'address', 'witness', 'class', 'glass', 'grass', 'mass', 'pass', 'boss', 'loss'
        }
        def fix_plural_sva(txt):
            def repl_plural_aux(m):
                subj = m.group(1)
                aux = m.group(2).lower()
                tokens = subj.strip().lower().split()
                w_low = tokens[-1] if tokens else ''

                # Prepositional phrase guard: If the noun is preceded by 'of',
                # the verb agrees with the primary head noun, NOT this prepositional object!
                start_idx = m.start()
                pre_text = txt[:start_idx].rstrip()
                if re.search(r'\bof\s+(?:the\s+|these\s+|those\s+|my\s+|our\s+|their\s+)?$', pre_text, re.IGNORECASE):
                    return m.group(0)

                is_plural = False
                if w_low in ('people', 'children', 'men', 'women', 'they', 'we', 'both', 'several', 'many', 'few'):
                    is_plural = True
                elif w_low not in SINGULAR_S_NOUNS and not w_low.endswith(('ss', 'us', 'is', 'ics')) and w_low.endswith('s') and len(w_low) > 2:
                    is_plural = True

                if not is_plural:
                    return m.group(0)

                aux_map = {'is': 'are', 'was': 'were', 'has': 'have', 'doesn\'t': 'don\'t', 'doesnt': 'dont'}
                new_aux = aux_map.get(aux, aux)
                if m.group(2)[0].isupper():
                    new_aux = new_aux.capitalize()
                return f'{subj}{new_aux}'

            pat = r'\b((?:two|three|four|five|six|seven|eight|nine|ten|several|many|both|few|\w+)\s+)(is|was|has|doesn\'?t)\b'
            return re.sub(pat, repl_plural_aux, txt, flags=re.IGNORECASE)

        corrected = fix_plural_sva(corrected)

        # Distributive Pronoun Subject-Verb Agreement: "Every one of the [plural] was", "Each of the [plural] was"
        distributive_sva = [
            (r'\b(every\s+one|each\s+one|each|one|neither|either)\s+of\s+((?:the\s+|these\s+|those\s+|my\s+|our\s+|their\s+)?[A-Za-z]+s)\s+were\b', r'\1 of \2 was'),
            (r'\b(every\s+one|each\s+one|each|one|neither|either)\s+of\s+((?:the\s+|these\s+|those\s+|my\s+|our\s+|their\s+)?[A-Za-z]+s)\s+are\b', r'\1 of \2 is'),
            (r'\b(every\s+one|each\s+one|each|one|neither|either)\s+of\s+((?:the\s+|these\s+|those\s+|my\s+|our\s+|their\s+)?[A-Za-z]+s)\s+have\b', r'\1 of \2 has'),
        ]
        for pat, repl in distributive_sva:
            corrected = re.sub(pat, repl, corrected, flags=re.IGNORECASE)

        # Distributive Pronoun Concord: "Every one of [X] ... because they [V]" -> "because he or she [V]"
        corrected = re.sub(
            r'\b(every\s+one|each\s+one|each)\s+of\s+([^,;\n]+?),\s+because\s+they\s+([A-Za-z]+)\b',
            r'\1 of \2, because he or she \3',
            corrected,
            flags=re.IGNORECASE
        )


        # Singular 3rd person subjects followed by plural auxiliary or base verb:
        singular_sva = [
            (r'\b(he|she|it|this|that|everyone|everybody|someone|nobody)\s+are\b', r'\1 is'),
            (r'\b(he|she|it|this|that|everyone|everybody|someone|nobody)\s+were\b', r'\1 was'),
            (r'\b(he|she|it|this|that|everyone|everybody|someone|nobody)\s+have\b', r'\1 has'),
            (r'\b(he|she|it|this|that|everyone|everybody|someone|nobody)\s+don\'?t\b', r'\1 doesn\'t'),
            (r'\b(he|she|it|everyone|someone|everybody)\s+go\b', r'\1 goes'),
            (r'\b(he|she|it|everyone|someone|everybody)\s+do\b', r'\1 does'),
            (r'\b(i)\s+is\b', r'I am'),
            (r'\b(i)\s+are\b', r'I am'),
        ]
        for pattern, repl in singular_sva:
            corrected = re.sub(pattern, repl, corrected, flags=re.IGNORECASE)

        # Collective Noun Agreement: "the board of directors was", "the engineering group was", "a consortium of journalists was", etc.
        collective_nouns = (
            r"board|committee|panel|team|group|crowd|jury|council|cabinet|faculty|"
            r"staff|commission|delegation|orchestra|choir|consortium|assembly|class|corps|"
            r"family|generation|tribe|army|fleet|series|collection|pair|bunch|government|organization|company|university|firm|agency|corporation|administration|"
            r"laboratory|institute|institution|foundation|association|center|division|department|bank|union|party|club|league|office|bureau|ministry|court|hospital|school|college"
        )
        collective_patterns = [
            (r'\b((?:the\s+|a\s+|an\s+)?(?:[\w\-]+\s+)?(?:' + collective_nouns + r')(?:\s+of\s+[\w\s]+?)?)\s+were\b', r'\1 was'),
            (r'\b((?:the\s+|a\s+|an\s+)?(?:[\w\-]+\s+)?(?:' + collective_nouns + r')(?:\s+of\s+[\w\s]+?)?)\s+are\b', r'\1 is'),
            (r'\b((?:the\s+|a\s+|an\s+)?(?:[\w\-]+\s+)?(?:' + collective_nouns + r')(?:\s+of\s+[\w\s]+?)?)\s+have\b', r'\1 has'),
        ]
        for pattern, repl in collective_patterns:
            corrected = re.sub(pattern, repl, corrected, flags=re.IGNORECASE)

        # Collective Noun Pronoun-Antecedent Agreement: "The research team ... their report" -> "its report"
        def fix_collective_pronoun(txt):
            pat = (
                r'\b((?:the\s+|a\s+|an\s+)?(?:[\w\-]+\s+)?(?:' + collective_nouns + r'))'
                r'(\b(?:\s+(?:which|that|who))?[^.;]*?\b)their\s+([A-Za-z]+)\b'
            )
            def repl(m):
                subj = m.group(1)
                mid = m.group(2)
                noun = m.group(3)
                subj_last = subj.strip().split()[-1].lower()
                if subj_last.endswith('s') and not subj_last.endswith(('ss', 'us')):
                    return m.group(0)  # Plural like 'teams', keep 'their'
                return f'{subj}{mid}its {noun}'
            return re.sub(pat, repl, txt, flags=re.IGNORECASE)

        corrected = fix_collective_pronoun(corrected)

        # Correlative Conjunction SVA (Proximity Rule): "neither A nor B were" -> "was" (verb agrees with B)
        def fix_correlative_conjunction(txt):
            def repl_corr(m):
                conn = m.group(1)
                subj1 = m.group(2)
                sep = m.group(3)
                subj2 = m.group(4)
                verb = m.group(5)

                s2_tokens = subj2.strip().split()
                last_word = s2_tokens[-1].lower() if s2_tokens else ''

                is_plural = (last_word.endswith('s') and not last_word.endswith(('ss', 'us', 'is'))) or last_word in ('people', 'children', 'men', 'women', 'they')

                verb_lower = verb.lower()
                if not is_plural:
                    if verb_lower == 'were': new_verb = 'was'
                    elif verb_lower == 'are': new_verb = 'is'
                    elif verb_lower == 'have': new_verb = 'has'
                    elif verb_lower == 'do': new_verb = 'does'
                    elif verb_lower == 'don\'t': new_verb = 'doesn\'t'
                    else: new_verb = verb
                else:
                    if verb_lower == 'was': new_verb = 'were'
                    elif verb_lower == 'is': new_verb = 'are'
                    elif verb_lower == 'has': new_verb = 'have'
                    elif verb_lower == 'does': new_verb = 'do'
                    elif verb_lower == 'doesn\'t': new_verb = 'don\'t'
                    else: new_verb = verb

                if verb[0].isupper():
                    new_verb = new_verb.capitalize()

                return f'{conn} {subj1} {sep} {subj2} {new_verb}'

            pat = r'\b(neither|either|not\s+only)\s+([^\n,;]+?)\s+(nor|or|but\s+also)\s+((?:the\s+|a\s+|an\s+|this\s+|that\s+|his\s+|her\s+|my\s+|our\s+)?[a-zA-Z\s\-]+?)\s+(were|was|are|is|have|has|do|does|don\'t|doesn\'t)\b'
            return re.sub(pat, repl_corr, txt, flags=re.IGNORECASE)

        corrected = fix_correlative_conjunction(corrected)

        # Compound Subject Coordination: 'Both A and B was/is/has' -> always plural 'were/are/have'
        def fix_both_and(txt):
            pat = r'\b(both\s+[a-zA-Z\s\-]+?\s+and\s+(?:the\s+|a\s+|an\s+|this\s+|that\s+|these\s+|those\s+|his\s+|her\s+|my\s+|our\s+|their\s+)?[a-zA-Z\-]+(?:\s+[a-zA-Z\-]+)?)\s+\b(was|is|has|does|doesn\'t)\b'
            def repl(m):
                prefix = m.group(1)
                verb = m.group(2)
                v_low = verb.lower()
                mapping = {'was': 'were', 'is': 'are', 'has': 'have', 'does': 'do', 'doesn\'t': 'don\'t'}
                new_v = mapping.get(v_low, verb)
                if verb[0].isupper():
                    new_v = new_v.capitalize()
                return f'{prefix} {new_v}'
            return re.sub(pat, repl, txt, flags=re.IGNORECASE)

        corrected = fix_both_and(corrected)

        # Parenthetical Coordination SVA: 'Subject1 (as well as|along with|together with|in addition to) Subject2 Verb' (Verb agrees with Subject1)
        def fix_parenthetical_coordination(txt):
            parentheticals = r'as\s+well\s+as|along\s+with|together\s+with|in\s+addition\s+to|accompanied\s+by|including'
            pat = r'\b((?:the\s+|a\s+|an\s+|this\s+|that\s+|these\s+|those\s+|his\s+|her\s+|my\s+|our\s+|their\s+)?[a-zA-Z\-]+(?:\s+[a-zA-Z\-]+)?)\s*,?\s*(' + parentheticals + r')\s+((?:the\s+|a\s+|an\s+|this\s+|that\s+|these\s+|those\s+|his\s+|her\s+|my\s+|our\s+|their\s+)?[a-zA-Z\-]+(?:\s+[a-zA-Z\-]+)?)\s*,?\s*\b(were|was|are|is|have|has|do|does|don\'t|doesn\'t)\b'
            def repl(m):
                subj1 = m.group(1).strip()
                conn = m.group(2)
                subj2 = m.group(3).strip()
                verb = m.group(4)

                s1_tokens = subj1.split()
                last_word = s1_tokens[-1].lower() if s1_tokens else ''
                is_s1_plural = (last_word.endswith('s') and last_word not in SINGULAR_S_NOUNS and not last_word.endswith(('ss', 'us', 'is', 'ics'))) or last_word in ('people', 'children', 'men', 'women', 'they', 'we', 'both', 'several', 'many')

                verb_lower = verb.lower()
                if not is_s1_plural:
                    if verb_lower == 'were': new_verb = 'was'
                    elif verb_lower == 'are': new_verb = 'is'
                    elif verb_lower == 'have': new_verb = 'has'
                    elif verb_lower == 'do': new_verb = 'does'
                    elif verb_lower == 'don\'t': new_verb = 'doesn\'t'
                    else: new_verb = verb
                else:
                    if verb_lower == 'was': new_verb = 'were'
                    elif verb_lower == 'is': new_verb = 'are'
                    elif verb_lower == 'has': new_verb = 'have'
                    elif verb_lower == 'does': new_verb = 'do'
                    elif verb_lower == 'doesn\'t': new_verb = 'don\'t'
                    else: new_verb = verb

                if verb[0].isupper():
                    new_verb = new_verb.capitalize()
                return f'{subj1} {conn} {subj2} {new_verb}'
            return re.sub(pat, repl, txt, flags=re.IGNORECASE)

        corrected = fix_parenthetical_coordination(corrected)

        # Academic / Latin / Greek Irregular Plural Concord (data were, criteria were, etc.)
        latin_plurals = r'data|criteria|phenomena|strata|bacteria|media|analyses|hypotheses|theses|crises|parentheses'
        corrected = re.sub(r'\b((?:the\s+|these\s+|those\s+|all\s+|such\s+|our\s+|their\s+|its\s+|experimental\s+|clinical\s+|empirical\s+)?(?:' + latin_plurals + r'))\s+was\b', r'\1 were', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b((?:the\s+|these\s+|those\s+|all\s+|such\s+|our\s+|their\s+|its\s+|experimental\s+|clinical\s+|empirical\s+)?(?:' + latin_plurals + r'))\s+is\b', r'\1 are', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b((?:the\s+|these\s+|those\s+|all\s+|such\s+|our\s+|their\s+|its\s+|experimental\s+|clinical\s+|empirical\s+)?(?:' + latin_plurals + r'))\s+has\b', r'\1 have', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b((?:the\s+|these\s+|those\s+|all\s+|such\s+|our\s+|their\s+|its\s+|experimental\s+|clinical\s+|empirical\s+)?(?:' + latin_plurals + r'))\s+shows\b', r'\1 show', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b((?:the\s+|these\s+|those\s+|all\s+|such\s+|our\s+|their\s+|its\s+|experimental\s+|clinical\s+|empirical\s+)?(?:' + latin_plurals + r'))\s+indicates\b', r'\1 indicate', corrected, flags=re.IGNORECASE)

        # 8b. Advanced Conditionals, Temporal Clause Linking, and Subjunctive
        participle_pattern = (
            r'[A-Za-z]+ed|[A-Za-z]+en|been|gone|seen|taken|done|written|made|come|found|'
            r'chosen|given|broken|known|heard|told|left|lost|paid|met|built|sent|spent|'
            r'run|held|bought|brought|thought|caught|fought|taught|sold|read|cut|hit|put|set'
        )

        # Third Conditional If-clause: "If X would have/would of/had have [V3]" -> "If X had [V3]"
        def repl_third_cond(m):
            if_clause = m.group(1)
            adv = m.group(2) or ''
            participle = m.group(3)
            sep = ' ' if adv else ''
            return f'{if_clause} had {adv}{sep}{participle}'

        corrected = re.sub(
            r'\b(if\s+[^,;\n]+?)\s+(?:would\s+have|would\s+of|had\s+have)\s+(?:(\w+ly)\s+)?(' + participle_pattern + r')\b',
            repl_third_cond,
            corrected,
            flags=re.IGNORECASE
        )

        # Second Conditional If-clause: "If X would be" -> "If X were", "If X would have a/an/the/more" -> "If X had"
        corrected = re.sub(r'\b(if\s+[^,;\n]+?)\s+would\s+be\b', r'\1 were', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b(if\s+[^,;\n]+?)\s+would\s+have\s+(a|an|the|more|enough|sufficient|some|any|no)\b', r'\1 had \2', corrected, flags=re.IGNORECASE)

        # Subjunctive 'were' in hypothetical / wish clauses: "If I was you" -> "If I were you", "wish I was" -> "wish I were"
        subjunctive_targets = r'i|he|she|it|this|that|[A-Za-z]+'
        corrected = re.sub(
            r'\b(if|as\s+if|as\s+though|wish\s+(?:that\s+)?)\s+(' + subjunctive_targets + r')\s+was\s+(you|here|there|alive|rich|younger|older|present|able|ready|possible|true)\b',
            r'\1 \2 were \3',
            corrected,
            flags=re.IGNORECASE
        )

        # First Conditional / Time clauses: No future auxiliary 'will' in time/condition clauses
        time_conjunctions = r'if|when|as\s+soon\s+as|until|before|after|unless|in\s+case'
        def repl_time_will(m):
            conj = m.group(1)
            subj = m.group(2)
            adv = m.group(3) or ''
            verb = m.group(4)
            s_tokens = subj.strip().lower().split()
            last_tok = s_tokens[-1] if s_tokens else ''
            is_singular_3rd = last_tok in ('he', 'she', 'it', 'this', 'that', 'someone', 'everyone', 'nobody') or (not last_tok.endswith('s') and last_tok not in ('i', 'you', 'we', 'they'))
            conjugated = verb
            if is_singular_3rd:
                if verb.endswith(('sh', 'ch', 'ss', 'x', 'o')):
                    conjugated = verb + 'es'
                elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
                    conjugated = verb[:-1] + 'ies'
                elif not verb.endswith('s'):
                    conjugated = verb + 's'
            sep = ' ' if adv else ''
            return f'{conj} {subj} {adv}{sep}{conjugated}'

        corrected = re.sub(
            r'\b(' + time_conjunctions + r')\s+([A-Za-z\s]+?)\s+will\s+(?:(\w+ly)\s+)?([A-Za-z]+)\b',
            repl_time_will,
            corrected,
            flags=re.IGNORECASE
        )

        # Correlative Conjunction Pairs: 'no sooner ... when' -> 'no sooner ... than', 'hardly ... than' -> 'hardly ... when'
        corrected = re.sub(r'\b(no\s+sooner\s+[^,;\n]+?)\s+when\b', r'\1 than', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b(hardly|scarcely)\s+([^,;\n]+?)\s+than\b', r'\1 \2 when', corrected, flags=re.IGNORECASE)

        # Negative Inversion: 'No sooner he had arrived' -> 'No sooner had he arrived'
        def repl_inversion(m):
            adv = m.group(1)
            subj = m.group(2)
            aux = m.group(3)
            rest = m.group(4)
            return f'{adv} {aux} {subj} {rest}'
        corrected = re.sub(r'\b(no\s+sooner|hardly|scarcely|seldom|rarely)\s+([A-Za-z]+)\s+(had|did|have|has)\s+([A-Za-z]+)\b', repl_inversion, corrected, flags=re.IGNORECASE)

        # Mandative Subjunctive: 'recommend that he goes' -> 'recommend that he go'
        def repl_mandative(m):
            verb = m.group(1)
            subj = m.group(2)
            target = m.group(3).lower()
            if target in ('is', 'was', 'are', 'were'):
                return f'{verb} that {subj} be'
            if target.endswith('ies'):
                base = target[:-3] + 'y'
            elif target.endswith('es') and target[:-2] in ('wash', 'watch', 'pass', 'fix', 'go', 'do'):
                base = target[:-2]
            elif target.endswith('s') and not target.endswith(('ss', 'us', 'is')):
                base = target[:-1]
            else:
                base = target
            return f'{verb} that {subj} {base}'

        corrected = re.sub(r'\b(recommend|recommends|recommended|suggest|suggests|suggested|insist|insists|insisted|demand|demands|demanded|essential|vital|imperative)\s+that\s+([A-Za-z]+)\s+([A-Za-z]+)\b', repl_mandative, corrected, flags=re.IGNORECASE)

        # Comparative & Idiomatic Connectors: 'prefer X than Y' -> 'prefer X to Y', 'different than' -> 'different from'
        corrected = re.sub(r'\bprefer\s+([A-Za-z\s]+?)\s+than\s+([A-Za-z\s]+?)\b', r'prefer \1 to \2', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bdifferent\s+than\b', 'different from', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b(superior|inferior|senior|junior)\s+than\b', r'\1 to', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bbetween\s+([A-Za-z\s]+?)\s+or\s+([A-Za-z\s]+?)\b', r'between \1 and \2', corrected, flags=re.IGNORECASE)

        # Gerund complements: 'look forward to hear' -> 'hearing', 'capable to do' -> 'capable of doing'
        corrected = re.sub(r'\b(look|looking|looked)\s+forward\s+to\s+([A-Za-z]+)\b', lambda m: f'{m.group(1)} forward to {m.group(2)}ing' if not m.group(2).endswith('ing') else m.group(0), corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bcapable\s+to\s+([A-Za-z]+)\b', r'capable of \1ing', corrected, flags=re.IGNORECASE)

        # Double Negatives: 'can\'t hardly' -> 'can hardly'
        corrected = re.sub(r'\b(can\'?t|couldn\'?t)\s+hardly\b', 'can hardly', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b(can\'?t|couldn\'?t)\s+scarcely\b', 'can scarcely', corrected, flags=re.IGNORECASE)



        # 9. Verb Tense & Past Context
        has_past_keyword = bool(re.search(r'\b(yesterday|last\s+(?:week|month|year)|ago)\b', corrected, re.IGNORECASE))
        if has_past_keyword:
            past_mappings = [
                (r'\bgo\b', 'went'),
                (r'\beat\b', 'ate'),
                (r'\bsee\b', 'saw'),
                (r'\bcome\b', 'came'),
                (r'\brun\b', 'ran'),
                (r'\btake\b', 'took'),
                (r'\bwalk\b', 'walked'),
                (r'\bplay\b', 'played'),
                (r'\bwatch\b', 'watched'),
            ]
            for pat, repl in past_mappings:
                corrected = re.sub(pat, repl, corrected, flags=re.IGNORECASE)

        # 10. Existential SVA (there was -> there were before plural noun phrases)
        corrected = re.sub(
            r'\bthere\s+was\s+(no\s+other\s+(?:viable\s+)?|many\s+|several\s+|a\s+few\s+|two\s+|three\s+|\w+s\b)',
            r'there were \1',
            corrected,
            flags=re.IGNORECASE
        )

        # 11. Narrative past tense context: 'which lead to' -> 'which led to'
        corrected = re.sub(r'\bwhich\s+lead\s+to\b', 'which led to', corrected, flags=re.IGNORECASE)

        # 12. Article corrections (a before vowel sound / an before consonant)
        vowel_sound_words = r'\b(apple|orange|egg|elephant|hour|honest|umbrella|idea|idiom|author)\b'
        consonant_sound_words = r'\b(book|car|dog|university|European|pen|cat|table|house|man|woman)\b'
        corrected = re.sub(r'\ba\s+(' + vowel_sound_words + r')\b', r'an \1', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\ban\s+(' + consonant_sound_words + r')\b', r'a \1', corrected, flags=re.IGNORECASE)

        # 13. Common Preposition idioms & Redundant Prepositions
        prep_mappings = [
            (r'\binterested\s+on\b', 'interested in'),
            (r'\bgood\s+in\s+playing\b', 'good at playing'),
            (r'\blisten\s+at\b', 'listen to'),
            (r'\bdepend\s+of\b', 'depend on'),
            (r'\bmarried\s+with\b', 'married to'),
            (r'\bcongratulate\s+for\b', 'congratulate on'),
            (r'\b(welcome|come|came|comes|coming|go|goes|went|going|gone|arrive|arrived|arrives|arriving|walk|walked|walking|drive|drove|driving|return|returned|returning|head|headed|heading|reach|reached|reaching|back)\s+to\s+home\b', r'\1 home'),
            (r'\b(discuss|discussed|discussing|discusses)\s+about\b', r'\1'),
            (r'\b(order|ordered|ordering|orders)\s+for\b(?=\s+(?:a|an|the|food|pizza|drinks|dinner|lunch|breakfast|[a-z]+))', r'\1'),
            (r'\b(enter|entered|entering|enters)\s+into\b(?=\s+(?:the\s+)?(?:room|building|hall|house|office|class))', r'\1'),
            (r'\b(comprise|comprises|comprised)\s+of\b', r'\1'),
            (r'\bcope\s+up\s+with\b', 'cope with'),
        ]
        for pat, repl in prep_mappings:
            corrected = re.sub(pat, repl, corrected, flags=re.IGNORECASE)

        # 14. Capitalize first letter and ensure ending punctuation
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]

        corrected = re.sub(r'\."\s*\.', '."', corrected)
        corrected = re.sub(r'\.\s*\.', '.', corrected)
        if corrected and not corrected.endswith(('.', '!', '?', '."', '!"', '?"', ".'", "!'", "?'")):
            if corrected.endswith(('"', "'")):
                corrected = corrected[:-1] + '.' + corrected[-1]
            else:
                corrected += '.'

        return corrected

    def _tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenizes text preserving word tokens and character spans (token, start_char, end_char)."""
        tokens = []
        for match in re.finditer(r"[\w']+|[^\w\s]", text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

    def _align_and_extract_errors(self, original: str, corrected: str) -> List[GrammarError]:
        """Aligns original and corrected tokens via SequenceMatcher to detect error spans."""
        orig_tokens_meta = self._tokenize(original)
        corr_tokens_meta = self._tokenize(corrected)

        orig_words = [t[0] for t in orig_tokens_meta]
        corr_words = [t[0] for t in corr_tokens_meta]

        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        opcodes = matcher.get_opcodes()

        errors = []
        error_id = 1

        def emit_error(start_i, end_i, start_j, end_j):
            nonlocal error_id
            sub_orig_tokens = orig_words[start_i:end_i]
            sub_corr_tokens = corr_words[start_j:end_j]
            if sub_orig_tokens == sub_corr_tokens:
                return

            if start_i < len(orig_tokens_meta) and end_i <= len(orig_tokens_meta) and start_i < end_i:
                start_char = orig_tokens_meta[start_i][1]
                end_char = orig_tokens_meta[end_i - 1][2]
                orig_span_text = original[start_char:end_char]
            elif start_i < len(orig_tokens_meta):
                start_char = orig_tokens_meta[start_i][1]
                end_char = start_char
                orig_span_text = ""
            else:
                start_char = len(original)
                end_char = len(original)
                orig_span_text = ""

            corr_span_text = " ".join(sub_corr_tokens)
            corr_span_text = re.sub(r"\s+([,.:;?!'’])", r"\1", corr_span_text)
            corr_span_text = re.sub(r"\b([a-zA-Z]+)\s+('\w+)\b", r"\1\2", corr_span_text)

            context_before = orig_words[max(0, start_i - 7):start_i]
            context_after = orig_words[end_i:min(len(orig_words), end_i + 5)]

            category, explanation, confidence = ErrorClassifier.classify(
                orig_tokens=sub_orig_tokens,
                corr_tokens=sub_corr_tokens,
                context_before=context_before,
                context_after=context_after
            )

            if orig_span_text == corr_span_text:
                return

            errors.append(GrammarError(
                error_id=error_id,
                original_text=orig_span_text,
                original_span=(start_char, end_char),
                original_tokens=sub_orig_tokens,
                suggested_text=corr_span_text,
                suggested_tokens=sub_corr_tokens,
                error_type=category,
                explanation=explanation,
                confidence=confidence
            ))
            error_id += 1

        KNOWN_MULTIWORD_SLANG = {
            'what sup', 'wat sup', 'whats up', 'wassup', 'wazzup', 'how r u', 'how are u', 'how ar you', 'how ar u',
            'gud mrng', 'gud nyt', 'gm', 'gn', 'cousin brother', 'cousin sister',
            'revert back', 'do the needful', 'out of station', 'cope up with'
        }

        def decompose_and_process(start_i, end_i, start_j, end_j):
            o_toks = orig_words[start_i:end_i]
            c_toks = corr_words[start_j:end_j]

            # 1. Known multi-word slang phrases: keep unified as one error
            if " ".join(o_toks).lower() in KNOWN_MULTIWORD_SLANG:
                emit_error(start_i, end_i, start_j, end_j)
                return

            # 1b. Multi-word slang sub-slice decomposition (e.g. 'hey what sup' -> 'hey' + 'what sup')
            if len(o_toks) > 1:
                for sz in (3, 2):
                    if len(o_toks) >= sz:
                        for a in range(len(o_toks) - sz + 1):
                            combo_phrase = " ".join(o_toks[a:a + sz]).lower()
                            if combo_phrase in KNOWN_MULTIWORD_SLANG:
                                for b in range(len(c_toks)):
                                    for c_sz in (1, 2, 3):
                                        if b + c_sz <= len(c_toks):
                                            c_phrase = " ".join(c_toks[b:b + c_sz]).lower().rstrip(',.!?')
                                            if c_phrase in ("what's up", "how are you", "good morning", "good night", "cousin", "reply", "welcome", "without"):
                                                if a > 0 or b > 0:
                                                    decompose_and_process(start_i, start_i + a, start_j, start_j + b)
                                                emit_error(start_i + a, start_i + a + sz, start_j + b, start_j + b + c_sz)
                                                if (a + sz < len(o_toks)) or (b + c_sz < len(c_toks)):
                                                    decompose_and_process(start_i + a + sz, end_i, start_j + b + c_sz, end_j)
                                                return

            # 2. Compound word merger decomposition (e.g. 'wel come' -> 'welcome', 'gan g' -> 'gang')
            if len(o_toks) > 1:
                for a in range(len(o_toks) - 1):
                    combo = (o_toks[a] + o_toks[a + 1]).lower()
                    for b in range(len(c_toks)):
                        if c_toks[b].lower().rstrip(',.!?') == combo:
                            if a > 0 or b > 0:
                                decompose_and_process(start_i, start_i + a, start_j, start_j + b)
                            emit_error(start_i + a, start_i + a + 2, start_j + b, start_j + b + 1)
                            if (a + 2 < len(o_toks)) or (b + 1 < len(c_toks)):
                                decompose_and_process(start_i + a + 2, end_i, start_j + b + 1, end_j)
                            return

            # 3. Compound word split decomposition (e.g. 'alot' -> 'a lot')
            if len(c_toks) > 1:
                for a in range(len(o_toks)):
                    for b in range(len(c_toks) - 1):
                        combo = (c_toks[b] + c_toks[b + 1]).lower()
                        if o_toks[a].lower().rstrip(',.!?') == combo:
                            if a > 0 or b > 0:
                                decompose_and_process(start_i, start_i + a, start_j, start_j + b)
                            emit_error(start_i + a, start_i + a + 1, start_j + b, start_j + b + 2)
                            if (a + 1 < len(o_toks)) or (b + 2 < len(c_toks)):
                                decompose_and_process(start_i + a + 1, end_i, start_j + b + 2, end_j)
                            return

            # 3d. Leading word case-match with trailing multi-word substitution (e.g. 'why this is' -> 'Why are they')
            if len(o_toks) > 1 and len(c_toks) > 1 and o_toks[0].lower() == c_toks[0].lower() and o_toks[0] != c_toks[0]:
                emit_error(start_i, start_i + 1, start_j, start_j + 1)
                decompose_and_process(start_i + 1, end_i, start_j + 1, end_j)
                return

            # 3e. Inversion / Transposition (e.g. 'this is' -> 'are they', 'you are' -> 'are you')
            if len(o_toks) == 2 and len(c_toks) == 2 and ("are" in [w.lower() for w in c_toks] or "is" in [w.lower() for w in c_toks]):
                emit_error(start_i, end_i, start_j, end_j)
                return

            # 4. Parallel token replacements
            if len(o_toks) > 1 and len(o_toks) == len(c_toks):
                for k in range(len(o_toks)):
                    emit_error(start_i + k, start_i + k + 1, start_j + k, start_j + k + 1)
                return

            emit_error(start_i, end_i, start_j, end_j)

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            decompose_and_process(i1, i2, j1, j2)

        return errors
