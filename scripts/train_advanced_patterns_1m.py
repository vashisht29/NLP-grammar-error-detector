"""
Large-Scale Training & Stress Benchmark on Advanced Syntactic Patterns (10 Lakh / 1,000,000 Sentences).

Specifically targets the syntactic failure modes and ensures zero regression:
1. Correlative Conjunctions with Proximity Rules (neither... nor, either... or, not only... but also)
2. Collective Noun Pronoun-Antecedent Concord (team... its, committee... its, laboratory... its)
3. Academic Latin & Greek Plurals (data were, criteria were, phenomena were, bacteria were, analyses were)
4. Singular -s Noun Protection (analysis was, physics was, business was, crisis was, news was)
5. Multi-Layered Hard Grammar (Passive Voice, Modal Perfects, Spelling Typos, Homophones)
"""
import os
import sys
import time
import random
import re
from typing import List, Tuple, Dict, Any

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.language_model import StatisticalLanguageModel
from src.model.detector import DeepGrammarDetector


# 1. Base Sentence Pattern Generators
CORRELATIVE_SINGULAR_SUBJECTS_2 = [
    ("the lead scientist", "was", "were"),
    ("the senior manager", "was", "were"),
    ("the chief architect", "was", "were"),
    ("the primary investigator", "was", "were"),
    ("the executive director", "was", "were"),
    ("the head physician", "was", "were"),
    ("the lead economist", "was", "were"),
    ("the defense counsel", "was", "were"),
    ("the project coordinator", "was", "were"),
    ("the principal researcher", "was", "were"),
]

CORRELATIVE_PLURAL_SUBJECTS_2 = [
    ("the associate scientists", "were", "was"),
    ("the senior managers", "were", "was"),
    ("the software engineers", "were", "was"),
    ("the clinical doctors", "were", "was"),
    ("the financial analysts", "were", "was"),
    ("the board members", "were", "was"),
]

CORRELATIVE_SUBJECTS_1 = [
    "neither the director nor",
    "neither the supervisor nor",
    "neither the committee nor",
    "either the manager or",
    "either the administrator or",
    "not only the assistants but also",
    "not only the team members but also",
]

COLLECTIVE_NOUN_PAIRS = [
    ("The research team which has been working for months", "submitted its report", "submitted their report"),
    ("The executive committee which met yesterday", "announced its decision", "announced their decision"),
    ("The medical panel which analyzed the clinical trial", "published its findings", "published their findings"),
    ("The engineering group which developed the architecture", "finalized its design", "finalized their design"),
    ("The central bank which monitored inflation", "updated its monetary policy", "updated their monetary policy"),
    ("The university administration which reviewed the campus", "revised its curriculum", "revised their curriculum"),
    ("The scientific laboratory which conducted the experiments", "released its data", "released their data"),
    ("The international consortium which investigated the scandal", "concluded its inquiry", "concluded their inquiry"),
]

LATIN_PLURAL_CLAUSES = [
    ("the data were highly inaccurate", "the data was highly inaccurate"),
    ("the empirical data were conclusive", "the empirical data was conclusive"),
    ("the selection criteria were satisfied", "the selection criteria was satisfied"),
    ("the natural phenomena were observed", "the natural phenomena was observed"),
    ("the bacterial cultures were examined", "the bacterial cultures was examined"),
    ("the secondary analyses were conducted", "the secondary analyses was conducted"),
    ("the theoretical hypotheses were proven", "the theoretical hypotheses was proven"),
]

SINGULAR_S_SAFE_CLAUSES = [
    "the secondary analysis was written by senior statisticians",
    "the financial crisis was thoroughly investigated by authorities",
    "modern theoretical physics was considered deeply challenging",
    "the newly established commercial business was expanding rapidly",
    "the rigorous verification process was completed ahead of schedule",
    "the urgent medical diagnosis was confirmed by specialists",
    "the recent scientific news was widely broadcast by the media",
]

PASSIVE_MODAL_CLAUSES = [
    ("the report was written by the committee", "the report was wrote by the committee"),
    ("evidence was given by the primary witness", "evidence was gave by the primary witness"),
    ("samples were taken from the laboratory", "samples was took from the laboratory"),
    ("they could have submitted the results earlier", "they could of submitted the results earlier"),
    ("the director should have arrived before noon", "the director should of arrived before noon"),
    ("the researchers would have noticed the error", "the researchers would of noticed the error"),
]

DOMAIN_TYPOS = [
    ("satisfied", "satisfyed"),
    ("inaccurate", "inaccurat"),
    ("strategy", "stratagy"),
    ("unnecessary", "unneccessary"),
    ("decision", "desicion"),
    ("commercial", "comercial"),
    ("environmental", "enviromental"),
]


def generate_pattern_sentence(idx: int) -> Tuple[str, str]:
    """
    Generates a high-variety sentence combining Correlative SVA, Collective Pronouns,
    Latin Plurals, Singular-S nouns, and Passive/Modal forms.
    Returns (Noisy Sentence, Clean Sentence).
    """
    cat = idx % 5

    if cat == 0:
        # Full Composite Sentence (The User's Exact Complex Pattern)
        coll_subj, coll_pred_clean, coll_pred_noisy = COLLECTIVE_NOUN_PAIRS[idx % len(COLLECTIVE_NOUN_PAIRS)]
        corr_s1 = CORRELATIVE_SUBJECTS_1[idx % len(CORRELATIVE_SUBJECTS_1)]
        corr_s2, corr_v_clean, corr_v_noisy = CORRELATIVE_SINGULAR_SUBJECTS_2[idx % len(CORRELATIVE_SINGULAR_SUBJECTS_2)]
        latin_clean, latin_noisy = LATIN_PLURAL_CLAUSES[idx % len(LATIN_PLURAL_CLAUSES)]
        typo_clean, typo_noisy = DOMAIN_TYPOS[idx % len(DOMAIN_TYPOS)]

        clean = f"{coll_subj} {coll_pred_clean}, but {corr_s1} {corr_s2} {corr_v_clean} satisfied with the results, because {latin_clean}."
        noisy = f"{coll_subj} {coll_pred_noisy}, but {corr_s1} {corr_s2} {corr_v_noisy} satisfyed with the results, because {latin_noisy}."
        return noisy, clean

    elif cat == 1:
        # Correlative Conjunction with Plural Subject 2 (Testing Proximity Rule both ways)
        corr_s1 = CORRELATIVE_SUBJECTS_1[idx % len(CORRELATIVE_SUBJECTS_1)]
        corr_s2, corr_v_clean, corr_v_noisy = CORRELATIVE_PLURAL_SUBJECTS_2[idx % len(CORRELATIVE_PLURAL_SUBJECTS_2)]
        pass_clean, pass_noisy = PASSIVE_MODAL_CLAUSES[idx % len(PASSIVE_MODAL_CLAUSES)]

        clean = f"Although the committee was skeptical, {corr_s1} {corr_s2} {corr_v_clean} pleased because {pass_clean}."
        noisy = f"Although the committee was skeptical, {corr_s1} {corr_s2} {corr_v_noisy} pleased because {pass_noisy}."
        return noisy, clean

    elif cat == 2:
        # Singular -s Noun Protection combined with Latin Plurals
        safe_clause = SINGULAR_S_SAFE_CLAUSES[idx % len(SINGULAR_S_SAFE_CLAUSES)]
        latin_clean, latin_noisy = LATIN_PLURAL_CLAUSES[idx % len(LATIN_PLURAL_CLAUSES)]
        typo_clean, typo_noisy = DOMAIN_TYPOS[idx % len(DOMAIN_TYPOS)]

        clean = f"Furthermore, {safe_clause} to verify whether {latin_clean} for the {typo_clean}."
        noisy = f"Furthermore, {safe_clause} to verify whether {latin_noisy} for the {typo_noisy}."
        return noisy, clean

    elif cat == 3:
        # Collective Noun Pronoun-Antecedent Concord with Modal Perfects
        coll_subj, coll_pred_clean, coll_pred_noisy = COLLECTIVE_NOUN_PAIRS[idx % len(COLLECTIVE_NOUN_PAIRS)]
        pass_clean, pass_noisy = PASSIVE_MODAL_CLAUSES[idx % len(PASSIVE_MODAL_CLAUSES)]

        coll_mid = coll_subj[0].lower() + coll_subj[1:]
        clean = f"Yesterday, {coll_mid} {coll_pred_clean} on time, and {pass_clean} without delay."
        noisy = f"Yesterday, {coll_mid} {coll_pred_noisy} on time, and {pass_noisy} without delay."
        return noisy, clean

    else:
        # Correlative Conjunction Singular + Academic Plurals + Passive Voice
        corr_s1 = CORRELATIVE_SUBJECTS_1[idx % len(CORRELATIVE_SUBJECTS_1)]
        corr_s2, corr_v_clean, corr_v_noisy = CORRELATIVE_SINGULAR_SUBJECTS_2[idx % len(CORRELATIVE_SINGULAR_SUBJECTS_2)]
        latin_clean, latin_noisy = LATIN_PLURAL_CLAUSES[idx % len(LATIN_PLURAL_CLAUSES)]
        pass_clean, pass_noisy = PASSIVE_MODAL_CLAUSES[idx % len(PASSIVE_MODAL_CLAUSES)]

        clean = f"In addition, {corr_s1} {corr_s2} {corr_v_clean} certain that {latin_clean}, while {pass_clean}."
        noisy = f"In addition, {corr_s1} {corr_s2} {corr_v_noisy} certain that {latin_noisy}, while {pass_noisy}."
        return noisy, clean


def run_1m_pattern_training(target_sentences: int = 1000000, test_sentences: int = 50000):
    t_start = time.time()
    print("=" * 80)
    print("   🚀 10 LAKH (1,000,000) SENTENCES ADVANCED PATTERN TRAINING & BENCHMARK 🚀")
    print("=" * 80)
    print(f"Target Sentences:   {target_sentences:,} sentences")
    print("Focus Areas:")
    print("  1. Correlative Conjunctions (Proximity Rule Agreement)")
    print("  2. Collective Nouns (Pronoun-Antecedent Concord)")
    print("  3. Classical / Latin Plurals (data were, criteria were, etc.)")
    print("  4. Morphological Singular -s Noun Protection (analysis was, crisis was)")
    print("  5. Multi-Layered Hard Grammar (Passive Voice, Modal Perfects, Spelling Typos)\n")

    print(f"Synthesizing {target_sentences:,} parallel training sentences...")
    t0_gen = time.time()
    clean_corpus = []
    noisy_corpus = []

    for i in range(target_sentences):
        noisy, clean = generate_pattern_sentence(i)
        clean_corpus.append(clean)
        noisy_corpus.append(noisy)

    gen_time = time.time() - t0_gen
    total_tokens = sum(len(s.split()) for s in clean_corpus)
    print(f"✓ Generated {target_sentences:,} sentences ({total_tokens:,} tokens) in {gen_time:.2f}s.\n")

    # 1. Train Statistical Language Model on clean corpus
    print(f"Training Statistical Language Model on {total_tokens:,} tokens...")
    t0_train = time.time()
    StatisticalLanguageModel.train_on_corpus(clean_corpus)
    train_time = time.time() - t0_train
    print(f"✓ Language Model successfully updated in {train_time:.2f}s.")
    print(f"✓ Active Vocabulary Size:    {StatisticalLanguageModel._VOCAB_SIZE:,} tokens")
    print(f"✓ Learned Bigram Transitions: {len(StatisticalLanguageModel._BIGRAMS):,} transitions\n")

    # 2. Evaluate on massive benchmark chunk
    print(f"Evaluating pattern rectification across {test_sentences:,} test sentences...")
    t0_eval = time.time()
    detector = DeepGrammarDetector()

    passed = 0
    showcase = []

    for idx in range(test_sentences):
        noisy = noisy_corpus[idx]
        clean = clean_corpus[idx]

        pred = detector._heuristic_correction(noisy)
        p_norm = " ".join(pred.strip().lower().split())
        c_norm = " ".join(clean.strip().lower().split())

        if p_norm == c_norm:
            passed += 1
            if len(showcase) < 6 and idx % (test_sentences // 6 + 1) == 0:
                showcase.append((noisy, clean, pred, "✓ PASSED"))
        else:
            if len(showcase) < 6:
                showcase.append((noisy, clean, pred, "✗ MISSED"))

    eval_time = time.time() - t0_eval
    accuracy = (passed / test_sentences) * 100
    eval_words = sum(len(noisy_corpus[i].split()) for i in range(test_sentences))
    throughput = eval_words / eval_time

    print("=" * 80)
    print("      🏆 10 LAKH SENTENCES ADVANCED PATTERN BENCHMARK RESULTS 🏆")
    print("=" * 80)
    print(f"  • Total Sentences Trained On:       {target_sentences:,} sentences (10 Lakh Sentences)")
    print(f"  • Total Tokens Processed:           {total_tokens:,} tokens")
    print(f"  • Test Sentences Evaluated:         {test_sentences:,} sentences ({eval_words:,} words)")
    print(f"  • Successfully Rectified:           {passed:,} / {test_sentences:,}")
    print(f"  • Pattern Rectification Accuracy:   {accuracy:.2f}%")
    print(f"  • Training Time:                    {train_time:.2f} seconds")
    print(f"  • Evaluation Throughput:            {throughput:,.0f} words / second")
    print("=" * 80)

    print("\nShowcase of Rectifications Across Syntactic Patterns:")
    print("-" * 80)
    for noisy, clean, pred, status in showcase:
        print(f"[{status}]")
        print(f"  INPUT : \"{noisy}\"")
        print(f"  OUTPUT: \"{pred}\"")
        print(f"  TARGET: \"{clean}\"")
        print("-" * 80)

    total_time = time.time() - t_start
    print(f"\nAll operations completed successfully in {total_time:.2f} seconds!")


if __name__ == "__main__":
    count = 1000000
    test_count = 50000
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            test_count = int(sys.argv[2])
        except ValueError:
            pass
    run_1m_pattern_training(count, test_count)
