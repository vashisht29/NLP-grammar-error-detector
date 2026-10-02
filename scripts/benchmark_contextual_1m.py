"""
Large-Scale Contextual NLP Benchmark: 1,000,000 Words Evaluated in Sentences.
Tests contextual disambiguation, real-word spelling errors (malapropisms/confusables),
ambiguous typo resolution, and false-positive resistance across 10 Lakh words.
"""
import os
import sys
import time
import random
from typing import List, Tuple, Dict, Any

# Ensure project root in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.homophones import HomophoneEngine
from src.model.gender_collocations import GenderCollocationEngine
from src.model.spellchecker import SpellChecker


# Representative sentence templates covering real-word confusions and ambiguous typos
# Format: (Corrupted Sentence with error, Target Correct Sentence, Confused Pair/Word)
REAL_WORD_SENTENCE_TEMPLATES = [
    # 1. form vs from
    ("I received a special gift form my best friend.", "I received a special gift from my best friend.", "form -> from"),
    ("She will travel form London to Paris tomorrow morning.", "She will travel from London to Paris tomorrow morning.", "form -> from"),
    ("Please fill out this registration from completely.", "Please fill out this registration form completely.", "from -> form"),
    ("We are far away form our sweet home.", "We are far away from our sweet home.", "form -> from"),

    # 2. quiet vs quite
    ("The student was quiet sure about the math exam answers.", "The student was quite sure about the math exam answers.", "quiet -> quite"),
    ("This new laptop model is quiet good and remarkably fast.", "This new laptop model is quite good and remarkably fast.", "quiet -> quite"),
    ("Please keep quite while everyone is reading in the library.", "Please keep quiet while everyone is reading in the library.", "quite -> quiet"),
    ("They stayed inside a very quite room during the evening.", "They stayed inside a very quiet room during the evening.", "quite -> quiet"),

    # 3. peace vs piece
    ("Would you like another peace of chocolate birthday cake?", "Would you like another piece of chocolate birthday cake?", "peace -> piece"),
    ("He carefully wrote his notes on a peace of paper.", "He carefully wrote his notes on a piece of paper.", "peace -> piece"),
    ("The brave leaders prayed together for world piece today.", "The brave leaders prayed together for world peace today.", "piece -> peace"),
    ("May the noble soul rest in piece forever and ever.", "May the noble soul rest in peace forever and ever.", "piece -> peace"),

    # 4. break vs brake
    ("The speeding car breaks suddenly stopped working on the highway.", "The speeding car brakes suddenly stopped working on the highway.", "breaks -> brakes"),
    ("The driver stepped on the breaks just in time yesterday.", "The driver stepped on the brakes just in time yesterday.", "breaks -> brakes"),
    ("Let us take a short coffee brake after the lecture.", "Let us take a short coffee break after the lecture.", "brake -> break"),
    ("The employees enjoyed a peaceful lunch brake in the garden.", "The employees enjoyed a peaceful lunch break in the garden.", "brake -> break"),

    # 5. desert vs dessert
    ("We ordered a delicious vanilla ice cream for desert.", "We ordered a delicious vanilla ice cream for dessert.", "desert -> dessert"),
    ("She baked a wonderful sweet strawberry desert for the guests.", "She baked a wonderful sweet strawberry dessert for the guests.", "desert -> dessert"),
    ("The Sahara dessert is famously hot, dry, and sandy.", "The Sahara desert is famously hot, dry, and sandy.", "dessert -> desert"),

    # 6. hear vs here
    ("Can you here the beautiful violin music in the background?", "Can you hear the beautiful violin music in the background?", "here -> hear"),
    ("I could not here anything over the loud crowd noise.", "I could not hear anything over the loud crowd noise.", "here -> hear"),
    ("Please come hear and sit right next to your friends.", "Please come here and sit right next to your friends.", "hear -> here"),

    # 7. bare vs bear
    ("The patient could not bare the extreme physical pain anymore.", "The patient could not bear the extreme physical pain anymore.", "bare -> bear"),
    ("Please bare with me while the software system loads.", "Please bear with me while the software system loads.", "bare -> bear"),
    ("The children were playing happily on the beach with bare feet.", "The children were playing happily on the beach with bare feet.", "bare (valid)"),

    # 8. dairy vs diary
    ("She wrote all her private personal secrets in her dairy.", "She wrote all her private personal secrets in her diary.", "dairy -> diary"),
    ("He keeps a daily secret dairy about his adventurous travels.", "He keeps a daily secret diary about his adventurous travels.", "dairy -> diary"),
    ("The local dairy farm produces fresh milk, cheese, and butter.", "The local dairy farm produces fresh milk, cheese, and butter.", "dairy (valid)"),

    # 9. angle vs angel
    ("The kind nurse looked like a gentle guardian angle to him.", "The kind nurse looked like a gentle guardian angel to him.", "angle -> angel"),
    ("The artist carefully measured an acute right angle with a ruler.", "The artist carefully measured an acute right angle with a ruler.", "angle (valid)"),

    # 10. trial vs trail
    ("The hikers followed a scenic mountain trial through the tall forest.", "The hikers followed a scenic mountain trail through the tall forest.", "trial -> trail"),
    ("The high court murder trial continued for three long weeks.", "The high court murder trial continued for three long weeks.", "trial (valid)"),

    # 11. loose vs lose
    ("Be careful not to loose your keys or wallet outside.", "Be careful not to lose your keys or wallet outside.", "loose -> lose"),
    ("The team did not want to loose the championship game.", "The team did not want to lose the championship game.", "loose -> lose"),
    ("He wore a comfortable loose cotton shirt during the hot summer.", "He wore a comfortable loose cotton shirt during the hot summer.", "loose (valid)"),

    # 12. weather vs whether (and ambiguous typo wether)
    ("I really wonder weather it will rain heavily this afternoon.", "I really wonder whether it will rain heavily this afternoon.", "weather -> whether"),
    ("We must decide wether or not we should proceed further.", "We must decide whether or not we should proceed further.", "wether -> whether"),
    ("The sunny weather in California was bright and warm all day.", "The sunny weather in California was bright and warm all day.", "weather (valid)"),

    # 13. its vs it's
    ("The joyful dog wagged it's tail when seeing the delicious food.", "The joyful dog wagged its tail when seeing the delicious food.", "it's -> its"),
    ("Look outside the window because its raining heavily right now.", "Look outside the window because it's raining heavily right now.", "its -> it's"),

    # 14. accept vs except
    ("She decided to accept the prestigious scholarship offer with honor.", "She decided to accept the prestigious scholarship offer with honor.", "accept (valid)"),
    ("All the students passed the exam accept for two students.", "All the students passed the exam except for two students.", "accept -> except"),

    # 15. their vs there vs they're
    ("The tourists parked their rented car over their by the fence.", "The tourists parked their rented car over there by the fence.", "their -> there"),
    ("Listen carefully because their going to announce the grand winner soon.", "Listen carefully because they're going to announce the grand winner soon.", "their -> they're"),

    # 16. your vs you're
    ("I am delighted to say that your welcome to join us.", "I am delighted to say that you're welcome to join us.", "your -> you're"),
    ("Do not forget to pick up you're heavy backpack from school.", "Do not forget to pick up your heavy backpack from school.", "you're -> your"),

    # 17. Ambiguous typos in sentences
    ("I had a deep thoght about future technology and artificial intelligence.", "I had a deep thought about future technology and artificial intelligence.", "thoght -> thought"),
    ("The engineering team created a great strategic paln for next quarter.", "The engineering team created a great strategic plan for next quarter.", "paln -> plan"),
    ("The student said I am redy to take the examination.", "The student said I am ready to take the examination.", "redy -> ready"),
    ("We wished our colleague a very happy bithday with sweets.", "We wished our colleague a very happy birthday with sweets.", "bithday -> birthday"),
    ("I love going for a walk early in the mornig.", "I love going for a walk early in the morning.", "mornig -> morning"),
    ("The two freinds walked togther to the university campus.", "The two friends walked together to the university campus.", "freinds/togther"),
    ("I noticed several speeling mistaks in that English sentance.", "I noticed several spelling mistakes in that English sentence.", "speeling/mistaks"),
    ("The little puppy wagged its tail happily near the river bank.", "The little puppy wagged its tail happily near the river bank.", "wagged (valid)"),
]

# Filler sentence variants to ensure natural English variety and reach exactly 1,000,000 words
MODIFIERS = [
    "recently", "certainly", "clearly", "undoubtedly", "definitely",
    "indeed", "always", "often", "frequently", "naturally"
]
EXTENSIONS = [
    "without any hesitation.",
    "during the annual conference.",
    "according to the latest official report.",
    "under the supervision of senior researchers.",
    "across all international regions.",
    "with great dedication and enthusiasm.",
    "before the deadline expired yesterday.",
    "after considering all relevant factors."
]


def run_sentence_through_nlp(sentence: str) -> str:
    """Fast execution through NLP Linguistic, Homophone, and SpellChecking layers."""
    # 1. Homophone & Confused Word Disambiguation
    corrected, _ = HomophoneEngine.apply(sentence)

    # 2. Gender Collocation
    corrected, _ = GenderCollocationEngine.apply(corrected)

    # 3. SpellChecker for remaining typos
    import re
    def fix_sp(m):
        w = m.group()
        c = SpellChecker.correct_word(w)
        return c if c else w

    corrected = re.sub(r'\b[A-Za-z]+\b', fix_sp, corrected)
    return corrected


def benchmark_1_million_words(target_word_count: int = 1000000):
    print("=" * 70)
    print(f"  CONTEXTUAL NLP SPELLING & REAL-WORD BENCHMARK ({target_word_count:,} WORDS)")
    print("=" * 70)

    print("Initializing NLP Engine, Homophone Dictionaries & Vocabulary...")
    t0 = time.time()
    SpellChecker._initialize()
    print(f"NLP Engine initialized in {time.time() - t0:.2f}s.\n")

    # Generate sentences until word count reaches target_word_count
    print(f"Generating sentences to reach {target_word_count:,} words...")
    sentences_data: List[Tuple[str, str, str, bool]] = []
    # (input_sentence, target_sentence, label, is_error_case)

    total_words = 0
    t_gen_start = time.time()

    template_idx = 0
    num_templates = len(REAL_WORD_SENTENCE_TEMPLATES)

    while total_words < target_word_count:
        corrupt, target, label = REAL_WORD_SENTENCE_TEMPLATES[template_idx % num_templates]
        template_idx += 1

        is_error = corrupt != target

        # Add modifier or extension to create realistic variety
        if random.random() < 0.4:
            ext = random.choice(EXTENSIONS)
            corrupt_mod = corrupt[:-1] + " " + ext
            target_mod = target[:-1] + " " + ext
            words_in_sent = len(corrupt_mod.split())
            sentences_data.append((corrupt_mod, target_mod, label, is_error))
            total_words += words_in_sent
        else:
            words_in_sent = len(corrupt.split())
            sentences_data.append((corrupt, target, label, is_error))
            total_words += words_in_sent

    print(f"Generated {len(sentences_data):,} sentences containing {total_words:,} words in {time.time() - t_gen_start:.2f}s.")
    print("Beginning Contextual Disambiguation & Sentence Verification...\n")

    t_eval_start = time.time()
    correct_count = 0
    real_word_correct = 0
    real_word_total = 0
    clean_correct = 0
    clean_total = 0

    showcase = []

    for idx, (inp, tgt, label, is_err) in enumerate(sentences_data):
        pred = run_sentence_through_nlp(inp)

        # Sentence match check
        is_match = pred.strip().lower() == tgt.strip().lower()

        if is_err:
            real_word_total += 1
            if is_match:
                real_word_correct += 1
                correct_count += 1
        else:
            clean_total += 1
            if is_match:
                clean_correct += 1
                correct_count += 1

        # Collect diverse showcase samples for reporting
        if len(showcase) < 14 and idx % (len(sentences_data) // 14 + 1) == 0:
            showcase.append({
                "input": inp,
                "target": tgt,
                "pred": pred,
                "label": label,
                "status": "✓ PASSED" if is_match else "✗ MISSED"
            })

        if (idx + 1) % 25000 == 0 or (idx + 1) == len(sentences_data):
            elapsed = time.time() - t_eval_start
            words_done = sum(len(s[0].split()) for s in sentences_data[:idx + 1])
            rate = words_done / elapsed if elapsed > 0 else 0
            acc = (correct_count / (idx + 1)) * 100
            print(f"  Processed {idx + 1:,} sentences ({words_done:,} words) | Accuracy: {acc:.2f}% | Rate: {rate:,.0f} words/sec")

    total_eval_time = time.time() - t_eval_start
    overall_acc = (correct_count / len(sentences_data)) * 100
    real_word_acc = (real_word_correct / real_word_total * 100) if real_word_total else 100.0
    clean_acc = (clean_correct / clean_total * 100) if clean_total else 100.0
    throughput = total_words / total_eval_time

    print("\n" + "=" * 70)
    print("  10 LAKH (1,000,000) WORDS BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  • Total Words Evaluated:         {total_words:,} words")
    print(f"  • Total Sentences Evaluated:     {len(sentences_data):,} sentences")
    print(f"  • Real-Word Error Rectification: {real_word_correct:,}/{real_word_total:,} ({real_word_acc:.2f}%)")
    print(f"  • False-Positive Resistance:     {clean_correct:,}/{clean_total:,} ({clean_acc:.2f}%)")
    print(f"  • Overall Sentence Accuracy:     {correct_count:,}/{len(sentences_data):,} ({overall_acc:.2f}%)")
    print(f"  • Total Inference Time:          {total_eval_time:.2f} seconds")
    print(f"  • Processing Throughput:         {throughput:,.0f} words / second")
    print("=" * 70)

    print("\nRepresentative Sentence Verification Showcase:")
    print("-" * 70)
    for s in showcase:
        print(f"[{s['status']}] {s['label']}")
        print(f"  Input:     \"{s['input']}\"")
        print(f"  Corrected: \"{s['pred']}\"")
        if s['status'] == "✗ MISSED":
            print(f"  Expected:  \"{s['target']}\"")
        print("-" * 70)


if __name__ == "__main__":
    words = 1000000
    if len(sys.argv) > 1:
        try:
            words = int(sys.argv[1])
        except ValueError:
            pass
    benchmark_1_million_words(words)
