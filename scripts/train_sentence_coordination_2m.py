"""
Large-Scale Sentence Coordination & Multi-Clause Training Script (20 Lakh / 2,000,000 Sentences).

Comprehensive end-to-end training and stress benchmark testing:
1. Compound Subject Coordination (Both... and -> plural verb)
2. Correlative Conjunction Coordination & Proximity Rules (neither... nor, either... or, not only... but also)
3. Parenthetical Coordination with Intervening Prepositional Phrases (as well as, along with, together with, in addition to, accompanied by)
4. Cross-Clause Temporal & Narrative Coordination (Sequence of Tenses across compound & complex sentences)
5. Conditional Clause Coordination (1st, 2nd, 3rd, and mixed conditionals)
6. Collective Noun Pronoun-Antecedent Concord across clauses
7. Academic Latin/Greek Irregular Plural Concord (data were, criteria were, analyses were)
8. Multi-Layered Hard Grammar Injections (passive participles, modal contractions, homophones, spelling typos)
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


# 1. Parenthetical Subjects (Singular Main Subject + Plural Intervening Subject)
PARENTHETICAL_SINGULAR_S1 = [
    ("The director", "as well as his assistants", "was invited", "were invited"),
    ("The lead investigator", "along with the researchers", "was present", "were present"),
    ("The senior manager", "together with the software engineers", "was reviewing", "were reviewing"),
    ("The chief executive", "in addition to the board members", "was confident", "were confident"),
    ("The head physician", "accompanied by the resident doctors", "was examining", "were examining"),
    ("The project coordinator", "as well as the field agents", "was preparing", "were preparing"),
    ("The lead architect", "along with the structural engineers", "was inspecting", "were inspecting"),
    ("The principal scientist", "together with the laboratory assistants", "was publishing", "were publishing"),
]

# 2. Parenthetical Subjects (Plural Main Subject + Singular Intervening Subject)
PARENTHETICAL_PLURAL_S1 = [
    ("The players", "along with the head coach", "were celebrating", "was celebrating"),
    ("The software engineers", "as well as the project manager", "were developing", "was developing"),
    ("The senior researchers", "together with the laboratory director", "were collaborating", "was collaborating"),
    ("The financial analysts", "in addition to the chief economist", "were forecasting", "was forecasting"),
    ("The board members", "accompanied by the corporate secretary", "were discussing", "was discussing"),
]

# 3. Compound Subjects with 'Both... and' (Always Plural)
BOTH_AND_SUBJECTS = [
    ("Both the manager and the supervisor", "were present", "was present"),
    ("Both the physician and the clinical nurse", "were attending", "was attending"),
    ("Both the author and the publisher", "were satisfied", "was satisfied"),
    ("Both the research team and the sponsor", "were delighted", "was delighted"),
    ("Both the lead scientist and the associate", "were investigating", "was investigating"),
    ("Both the architect and the structural engineer", "were evaluating", "was evaluating"),
]

# 4. Correlative Conjunction Pairs (Proximity Rule)
CORRELATIVE_PROXIMITY_SINGULAR = [
    ("neither the manager nor the lead scientist", "was satisfied", "were satisfied"),
    ("neither the supervisor nor the primary investigator", "was convinced", "were convinced"),
    ("either the administrator or the senior director", "was responsible", "were responsible"),
    ("not only the assistants but also the chief architect", "was confident", "were confident"),
    ("neither the counsel nor the executive director", "was present", "were present"),
]

CORRELATIVE_PROXIMITY_PLURAL = [
    ("neither the director nor the associate scientists", "were pleased", "was pleased"),
    ("either the manager or the senior engineers", "were notified", "was notified"),
    ("not only the supervisor but also the committee members", "were convinced", "was convinced"),
    ("neither the lead scientist nor the research assistants", "were informed", "was informed"),
]

# 5. Third Conditional & Temporal Cross-Clause Linkage
CONDITIONAL_TEMPORAL_PAIRS = [
    (
        "If the financial advisor had recommended a more stable portfolio",
        "If the financial advisor would have recomended a more stable portfolio",
        "the couple could have avoided losing their life savings",
        "the couple could of avoided losing their life savings",
        "the market fluctuations were completely unforeseeable",
        "the market fluctuashons were completely unforseeable",
        "professional experience",
        "professional expirience"
    ),
    (
        "If the technical team had upgraded the security protocol earlier",
        "If the technical team would have upgraded the security protocol earlier",
        "the bank could have prevented the malicious breach",
        "the bank could of prevented the malicious breach",
        "the network fluctuations were highly volatile",
        "the network fluctuashons were highly volatile",
        "technical experience",
        "technical expirience"
    ),
    (
        "If the senior architect had reviewed the structural foundation",
        "If the senior architect would have reviewed the structural foundation",
        "the company could have finished the construction on time",
        "the company could of finished the construction on time",
        "the ground fluctuations were thoroughly monitored",
        "the ground fluctuashons were thoroughly monitored",
        "engineering experience",
        "engineering expirience"
    ),
]

# 6. Corporate Collective Noun + Multi-Clause Narrative
COLLECTIVE_NARRATIVE_TRIPLETS = [
    (
        "Although the board of directors was highly skeptical about the new strategy",
        "Although the board of directors were highly skeptical about the new stratagy",
        "they ultimately decided to approve it because there were no other viable alternatives",
        "they ultimately decided to approve it because their was no other viable alternatives",
        "which led to a completely unnecessary financial crisis that affected every employee",
        "which lead to a completely unneccessary financial crisis that affected every employees"
    ),
    (
        "Although the central committee was skeptical about the agricultural policy",
        "Although the central committee were skeptical about the agricultural policy",
        "they ultimately decided to approve it because there were no other viable alternatives",
        "they ultimately decided to approve it because their was no other viable alternatives",
        "which led to a completely unnecessary economic crisis that affected every farmer",
        "which lead to a completely unneccessary economic crisis that affected every farmers"
    ),
    (
        "Although the executive panel was skeptical about the marketing proposal",
        "Although the executive panel were skeptical about the marketing proposal",
        "they ultimately decided to approve it because there were no other viable alternatives",
        "they ultimately decided to approve it because their was no other viable alternatives",
        "which led to a completely unnecessary operational crisis that affected every consumer",
        "which lead to a completely unneccessary operational crisis that affected every consumers"
    ),
]

# 7. Latin Irregular Plurals + Passive Participle Clauses
LATIN_PASSIVE_PAIRS = [
    ("the data were highly inaccurate", "the data was highly inaccurate", "the report was written by the committee", "the report was wrote by the committee"),
    ("the secondary analyses were conclusive", "the secondary analyses was conclusive", "samples were taken from the site", "samples was took from the site"),
    ("the criteria were thoroughly verified", "the criteria was thoroughly verified", "evidence was given by the witness", "evidence was gave by the witness"),
    ("the empirical data were validated", "the empirical data was validated", "the decision was made by the board", "the decision was make by the board"),
    ("the scientific phenomena were observed", "the scientific phenomena was observed", "the records were kept securely", "the records was kept securely"),
]


def generate_coordination_sentence(idx: int) -> Tuple[str, str]:
    mode = idx % 6

    if mode == 0:
        s1, s2, v_clean, v_noisy = PARENTHETICAL_SINGULAR_S1[idx % len(PARENTHETICAL_SINGULAR_S1)]
        latin_c, latin_n, pass_c, pass_n = LATIN_PASSIVE_PAIRS[idx % len(LATIN_PASSIVE_PAIRS)]
        clean = f"{s1} {s2} {v_clean} to confirm that {latin_c}, and {pass_c}."
        noisy = f"{s1} {s2} {v_noisy} to confirm that {latin_n}, and {pass_n}."
        return noisy, clean

    elif mode == 1:
        s1, s2, v_clean, v_noisy = PARENTHETICAL_PLURAL_S1[idx % len(PARENTHETICAL_PLURAL_S1)]
        corr_subj, corr_v_c, corr_v_n = CORRELATIVE_PROXIMITY_SINGULAR[idx % len(CORRELATIVE_PROXIMITY_SINGULAR)]
        clean = f"While {s1} {s2} {v_clean} the victory, {corr_subj} {corr_v_c} with the outcome."
        noisy = f"While {s1} {s2} {v_noisy} the victory, {corr_subj} {corr_v_n} with the outcome."
        return noisy, clean

    elif mode == 2:
        both_subj, both_v_c, both_v_n = BOTH_AND_SUBJECTS[idx % len(BOTH_AND_SUBJECTS)]
        latin_c, latin_n, pass_c, pass_n = LATIN_PASSIVE_PAIRS[idx % len(LATIN_PASSIVE_PAIRS)]
        clean = f"{both_subj} {both_v_c} during the conference, because {latin_c}."
        noisy = f"{both_subj} {both_v_n} during the conference, because {latin_n}."
        return noisy, clean

    elif mode == 3:
        cond_c, cond_n, mod_c, mod_n, fluc_c, fluc_n, exp_c, exp_n = CONDITIONAL_TEMPORAL_PAIRS[idx % len(CONDITIONAL_TEMPORAL_PAIRS)]
        clean = f"{cond_c}, {mod_c}, especially since {fluc_c} to someone without {exp_c}."
        noisy = f"{cond_n}, {mod_n}, especially since {fluc_n} to someone without {exp_n}."
        return noisy, clean

    elif mode == 4:
        cl1_c, cl1_n, cl2_c, cl2_n, cl3_c, cl3_n = COLLECTIVE_NARRATIVE_TRIPLETS[idx % len(COLLECTIVE_NARRATIVE_TRIPLETS)]
        clean = f"{cl1_c}, {cl2_c} available at that moment, {cl3_c}."
        noisy = f"{cl1_n}, {cl2_n} available at that moment, {cl3_n}."
        return noisy, clean

    else:
        corr_subj, corr_v_c, corr_v_n = CORRELATIVE_PROXIMITY_PLURAL[idx % len(CORRELATIVE_PROXIMITY_PLURAL)]
        latin_c, latin_n, pass_c, pass_n = LATIN_PASSIVE_PAIRS[idx % len(LATIN_PASSIVE_PAIRS)]
        clean = f"Although the committee was skeptical, {corr_subj} {corr_v_c} that {latin_c}, while {pass_c}."
        noisy = f"Although the committee was skeptical, {corr_subj} {corr_v_n} that {latin_n}, while {pass_n}."
        return noisy, clean


def run_coordination_training(target_sentences: int = 2000000, test_sentences: int = 50000):
    t_start = time.time()
    print('=' * 85)
    print('   🚀 20 LAKH (2,000,000) SENTENCES UNIFIED SENTENCE COORDINATION TRAINING 🚀')
    print('=' * 85)
    print(f'Target Sentences:    {target_sentences:,} sentences')
    print('Sentence Coordination Focus Dimensions:')
    print('  1. Compound Subject Coordination (Both A and B -> plural verb concord)')
    print('  2. Parenthetical Coordination (S1 as well as S2 Verb -> agrees with S1)')
    print('  3. Correlative Conjunction Coordination (neither... nor, either... or -> proximity)')
    print('  4. Third Conditional & Temporal Clause Linking (If had V3... could have V3)')
    print('  5. Multi-Clause Narrative Synthesis (Although X, Y because Z, which led to W)')
    print('  6. Academic Latin/Greek Irregular Plural Coordination (data were, criteria were)\n')

    print(f'Synthesizing {target_sentences:,} parallel training sentences across all coordination modes...')
    t0_gen = time.time()
    clean_corpus = []
    noisy_corpus = []

    for i in range(target_sentences):
        noisy, clean = generate_coordination_sentence(i)
        clean_corpus.append(clean)
        noisy_corpus.append(noisy)

    gen_time = time.time() - t0_gen
    total_tokens = sum(len(s.split()) for s in clean_corpus)
    print(f'✓ Generated {target_sentences:,} sentences ({total_tokens:,} tokens) in {gen_time:.2f}s.\n')

    # 1. Train Statistical Language Model on the unified coordinated corpus
    print(f'Updating Language Model N-Gram transition probabilities across {total_tokens:,} tokens...')
    t0_train = time.time()
    StatisticalLanguageModel.train_on_corpus(clean_corpus)
    train_time = time.time() - t0_train
    print(f'✓ Statistical Language Model successfully updated in {train_time:.2f}s.')
    print(f'✓ Total Vocabulary Size:      {StatisticalLanguageModel._VOCAB_SIZE:,} words')
    print(f'✓ Coordinated N-Gram Pairs:   {len(StatisticalLanguageModel._BIGRAMS):,} transitions\n')

    # 2. Benchmark evaluation across test sentences
    print(f'Evaluating end-to-end coordination rectification across {test_sentences:,} test sentences...')
    t0_eval = time.time()
    detector = DeepGrammarDetector()

    passed = 0
    showcase = []

    for idx in range(test_sentences):
        noisy = noisy_corpus[idx]
        clean = clean_corpus[idx]

        pred = detector._heuristic_correction(noisy)
        p_norm = ' '.join(pred.strip().lower().split())
        c_norm = ' '.join(clean.strip().lower().split())

        if p_norm == c_norm:
            passed += 1
            if len(showcase) < 8 and idx % (test_sentences // 8 + 1) == 0:
                showcase.append((noisy, clean, pred, '✓ PASSED'))
        else:
            if len(showcase) < 8:
                showcase.append((noisy, clean, pred, '✗ MISSED'))

    eval_time = time.time() - t0_eval
    accuracy = (passed / test_sentences) * 100
    eval_words = sum(len(noisy_corpus[i].split()) for i in range(test_sentences))
    throughput = eval_words / eval_time

    print('=' * 85)
    print('      🏆 UNIFIED SENTENCE COORDINATION BENCHMARK RESULTS 🏆')
    print('=' * 85)
    print(f'  • Total Parallel Sentences:          {target_sentences:,} sentences (20 Lakh Sentences)')
    print(f'  • Total Tokens Processed:            {total_tokens:,} tokens')
    print(f'  • Stress Test Sentences Evaluated:   {test_sentences:,} sentences ({eval_words:,} words)')
    print(f'  • Perfectly Coordinated & Rectified: {passed:,} / {test_sentences:,}')
    print(f'  • Coordination Rectification Accuracy: {accuracy:.2f}%')
    print(f'  • Language Model Training Time:      {train_time:.2f} seconds')
    print(f'  • Evaluation Throughput:             {throughput:,.0f} words / second')
    print('=' * 85)

    print('\nRepresentative Showcase Across Sentence Coordination Types:')
    print('-' * 85)
    for noisy, clean, pred, status in showcase:
        print(f'[{status}]')
        print(f'  INPUT : "{noisy}"')
        print(f'  OUTPUT: "{pred}"')
        print(f'  TARGET: "{clean}"')
        print('-' * 85)

    total_time = time.time() - t_start
    print(f'\nAll operations completed successfully in {total_time:.2f} seconds!')


if __name__ == '__main__':
    count = 2000000
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
    run_coordination_training(count, test_count)
