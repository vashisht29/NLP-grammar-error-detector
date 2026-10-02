"""
Large-Scale NLP Spelling Stress-Test & Benchmark (100,000 Hard Spelling Mistakes).
Tests accuracy, error rectification, and throughput on:
1. Real-world human spelling errors from Birkbeck & Wikipedia (spell-errors.txt)
2. Hard multi-character synthetic typos (phonetic swaps, consonant drops, transpositions)
"""
import os
import sys
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.spellchecker import SpellChecker


def generate_hard_typo(word: str) -> str:
    """Generates a hard spelling mistake for a word."""
    if len(word) < 4:
        return word

    w = list(word)
    choice = random.choice([1, 2, 3, 4, 5])

    # 1. Transposition (swap adjacent)
    if choice == 1 and len(w) >= 4:
        idx = random.randint(1, len(w) - 2)
        w[idx], w[idx + 1] = w[idx + 1], w[idx]

    # 2. Consonant omission (e.g. drop double letter or middle consonant)
    elif choice == 2:
        double_idx = -1
        for i in range(len(w) - 1):
            if w[i] == w[i + 1]:
                double_idx = i
                break
        if double_idx != -1:
            w.pop(double_idx)
        else:
            w.pop(random.randint(1, len(w) - 2))

    # 3. Vowel confusion (swap e <-> a, i <-> e, o <-> u)
    elif choice == 3:
        vowels = {'a': 'e', 'e': 'i', 'i': 'e', 'o': 'u', 'u': 'o'}
        for i in range(len(w)):
            if w[i] in vowels and random.random() < 0.6:
                w[i] = vowels[w[i]]
                break

    # 4. Phonetic corruption (ph -> f, c -> k, tion -> shun)
    elif choice == 4:
        joined = "".join(w)
        if "ph" in joined:
            joined = joined.replace("ph", "f", 1)
        elif "tion" in joined:
            joined = joined.replace("tion", "shun", 1)
        elif "c" in joined and not joined.endswith("c"):
            joined = joined.replace("c", "k", 1)
        else:
            # Drop silent letter
            idx = random.randint(1, len(w) - 2)
            joined = joined[:idx] + joined[idx+1:]
        return joined

    # 5. Letter duplication (e.g. add extra consonant)
    else:
        idx = random.randint(1, len(w) - 2)
        w.insert(idx, w[idx])

    typo = "".join(w)
    return typo if typo != word else word + "s"


def run_benchmark(target_count: int = 100000):
    print("=" * 65)
    print(f"  NLP SPELLING STRESS-TEST BENCHMARK ({target_count:,} WORDS)")
    print("=" * 65)

    print("Initializing NLP Statistical SpellChecker...")
    t0_init = time.time()
    SpellChecker._initialize()
    print(f"Model initialized in {time.time() - t0_init:.2f}s.")

    # 1. Gather real-world human error pairs
    test_cases = []  # List of (typo, target_correct_word)
    errors_file = os.path.join("src", "data", "spell-errors.txt")
    if os.path.exists(errors_file):
        with open(errors_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" in line:
                    correct, typos = line.strip().split(":", 1)
                    correct = correct.strip().lower()
                    for typo in typos.split(","):
                        typo = typo.strip().lower()
                        if "*" in typo:
                            typo = typo.split("*")[0].strip()
                        if typo and typo != correct and typo.isalpha() and correct.isalpha():
                            test_cases.append((typo, correct))

    print(f"Loaded {len(test_cases):,} real-world human error pairs from Birkbeck corpus.")

    # 2. Synthesize hard spelling mistakes from top vocabulary until target_count
    vocab_file = os.path.join("src", "data", "count_1w.txt")
    vocab = []
    if os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2 and parts[0].isalpha() and len(parts[0]) >= 4:
                    vocab.append(parts[0].lower())
                if len(vocab) >= 30000:
                    break

    random.seed(42)
    needed = target_count - len(test_cases)
    print(f"Generating {needed:,} hard synthetic test cases from {len(vocab):,} vocabulary words...")
    
    for i in range(needed):
        target = vocab[i % len(vocab)]
        typo = generate_hard_typo(target)
        test_cases.append((typo, target))

    # Shuffle test cases
    random.shuffle(test_cases)
    total_to_run = len(test_cases)

    print(f"\nStarting benchmark on {total_to_run:,} spelling mistake test cases...")
    t_start = time.time()
    
    correct_count = 0
    showcase_examples = []

    for idx, (typo, target) in enumerate(test_cases):
        pred = SpellChecker.correct_word(typo)
        
        # Check match (case insensitive)
        if pred and pred.lower() == target.lower():
            correct_count += 1
            if len(showcase_examples) < 12 and idx % 2000 == 0:
                showcase_examples.append((typo, target, pred, "✓ PASSED"))
        elif len(showcase_examples) < 12 and idx % 4000 == 0:
            showcase_examples.append((typo, target, pred, "✗ MISSED"))

        if (idx + 1) % 25000 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            acc = (correct_count / (idx + 1)) * 100
            print(f"  Progress: {idx + 1:,}/{total_to_run:,} | Acc: {acc:.2f}% | Rate: {rate:,.0f} words/sec")

    total_time = time.time() - t_start
    overall_accuracy = (correct_count / total_to_run) * 100
    throughput = total_to_run / total_time

    print("\n" + "=" * 65)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 65)
    print(f"  • Total Words Tested:      {total_to_run:,}")
    print(f"  • Correctly Rectified:     {correct_count:,}")
    print(f"  • Accuracy:                {overall_accuracy:.2f}%")
    print(f"  • Total Execution Time:    {total_time:.2f} seconds")
    print(f"  • Throughput:              {throughput:,.0f} words / second")
    print("=" * 65)

    print("\nRepresentative Sample of Evaluated Words:")
    print(f"{'Input Typo':18} {'Target Word':18} {'Model Output':18} {'Status'}")
    print("-" * 65)
    for typo, target, pred, status in showcase_examples:
        pred_disp = pred if pred else "None"
        print(f"{typo:18} {target:18} {pred_disp:18} {status}")
    print("=" * 65)


if __name__ == "__main__":
    count = 100000
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(count)
