"""
Hard Robustness Training & Comprehensive Evaluation Suite.
Tests and verifies:
1. Subword Word Segmentation (glued words)
2. Typoglycemia Anagram Signature Matcher (scrambled inner letters)
3. Missing Punctuation Spacing
4. Semantic Intent & Keyboard Fat-Finger Locking
5. Whitelist Compound Word Protection
6. Hard Multi-Layered Paragraphs (simultaneous grammar + spelling + formatting)
7. All 8 Foundational Root Benchmarks (Zero Regressions)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.detector import DeepGrammarDetector


def run_hard_suite():
    print("=" * 75)
    print("  🚀 HARD STRESS-TEST & ROBUSTNESS EVALUATION SUITE")
    print("=" * 75)
    
    t0 = time.time()
    detector = DeepGrammarDetector()
    print(f"Backend: {detector.backend}")
    print(f"Device:  {detector.device.upper()}\n")

    passed_count = 0
    total_count = 0

    # -------------------------------------------------------------
    # Category 1: Feature 1 - Subword Glued / Run-on Word Splitting
    # -------------------------------------------------------------
    print("--- [TEST SECTION 1: Subword Glued Words] ---")
    subword_tests = [
        ("this is a speelingmistake in the essay", "This is a spelling mistake in the essay."),
        ("he is verygood at playing chess", "He is very good at playing chess."),
        ("i am working on myproject today", "I am working on my project today."),
        ("goodmorning to everyone in the room", "Good morning to everyone in the room."),
        ("we studied englishgrammar together", "We studied English grammar together."),
        ("i am going tothe market infrontof eachother", "I am going to the market in front of each other."),
        ("we have alotof questions and we needto talkto them", "We have a lot of questions and we need to talk to them."),
        ("this is verygood and wasa great accomplishment", "This is very good and was a great accomplishment."),
        ("i havea speelingmistake in myproject", "I have a spelling mistake in my project."),
        ("we studied englishgrammar together atschool", "We studied English grammar together at school."),
    ]
    for inp, exp in subword_tests:
        total_count += 1
        res = detector.detect(inp)
        ok = res.corrected_sentence.rstrip('.') == exp.rstrip('.')
        if ok:
            passed_count += 1
            print(f"  ✓ PASSED: '{inp}' ➔ '{res.corrected_sentence}'")
        else:
            print(f"  ✗ FAILED: '{inp}'\n     Expected: '{exp}'\n     Got:      '{res.corrected_sentence}'")

    # -------------------------------------------------------------
    # Category 2: Feature 2 - Typoglycemia / Scrambled Letters
    # -------------------------------------------------------------
    print("\n--- [TEST SECTION 2: Typoglycemia Jumbled Words] ---")
    typoglycemia_tests = [
        ("this was a terrible mtiaske by the team", "This was a terrible mistake by the team."),
        ("she finished the difficult porject on time", "She finished the difficult project on time."),
        ("we had a wodnerful vacation in italy", "We had a wonderful vacation in Italy."),
        ("read the whole setneces carefully", "Read the whole sentence carefully."),
    ]
    for inp, exp in typoglycemia_tests:
        total_count += 1
        res = detector.detect(inp)
        ok = res.corrected_sentence.rstrip('.') == exp.rstrip('.')
        if ok:
            passed_count += 1
            print(f"  ✓ PASSED: '{inp}' ➔ '{res.corrected_sentence}'")
        else:
            print(f"  ✗ FAILED: '{inp}'\n     Expected: '{exp}'\n     Got:      '{res.corrected_sentence}'")

    # -------------------------------------------------------------
    # Category 3: Feature 3 - Missing Punctuation Spacing
    # -------------------------------------------------------------
    print("\n--- [TEST SECTION 3: Missing Punctuation Spacing] ---")
    punct_tests = [
        ("hello,world", "Hello, world."),
        ("the meeting ended.Then we had lunch", "The meeting ended. Then we had lunch."),
        ("are you ready?yes I am", "Are you ready? Yes I am."),
        ("look!there is a bird", "Look! There is a bird."),
    ]
    for inp, exp in punct_tests:
        total_count += 1
        res = detector.detect(inp)
        ok = res.corrected_sentence.rstrip('.') == exp.rstrip('.')
        if ok:
            passed_count += 1
            print(f"  ✓ PASSED: '{inp}' ➔ '{res.corrected_sentence}'")
        else:
            print(f"  ✗ FAILED: '{inp}'\n     Expected: '{exp}'\n     Got:      '{res.corrected_sentence}'")

    # -------------------------------------------------------------
    # Category 4: Feature 4 - Semantic Intent & Fat-Finger Locking
    # -------------------------------------------------------------
    print("\n--- [TEST SECTION 4: Semantic Intent & Fat-Finger Locking] ---")
    intent_tests = [
        ("two frnd", "Two friends."),
        ("he is handone", "He is handsome."),
        ("he is strng", "He is strong."),
        ("she is handsome", "She is beautiful."),
    ]
    for inp, exp in intent_tests:
        total_count += 1
        res = detector.detect(inp)
        ok = res.corrected_sentence.rstrip('.') == exp.rstrip('.')
        if ok:
            passed_count += 1
            print(f"  ✓ PASSED: '{inp}' ➔ '{res.corrected_sentence}'")
        else:
            print(f"  ✗ FAILED: '{inp}'\n     Expected: '{exp}'\n     Got:      '{res.corrected_sentence}'")

    # -------------------------------------------------------------
    # Category 5: Authentic Compound Whitelist (NEVER SPLIT)
    # -------------------------------------------------------------
    print("\n--- [TEST SECTION 5: Compound Words Whitelist Preservation] ---")
    whitelist_sentences = [
        ("understand", "We understand the rules carefully."),
        ("sometimes", "Sometimes we make unexpected mistakes."),
        ("therefore", "Therefore we must remain vigilant."),
        ("everything", "We reviewed everything in the report."),
        ("without", "We completed the project without any delay."),
        ("football", "They played football in the evening."),
        ("throughout", "The team worked hard throughout the month."),
        ("sophisticated", "She has a sophisticated strategy.")
    ]
    for comp, sent in whitelist_sentences:
        total_count += 1
        res = detector.detect(sent)
        ok = comp in res.corrected_sentence.lower()
        if ok:
            passed_count += 1
            print(f"  ✓ PASSED: Word '{comp}' preserved (never erroneously split).")
        else:
            print(f"  ✗ FAILED: Word '{comp}' was incorrectly split in: '{res.corrected_sentence}'")

    # -------------------------------------------------------------
    # Category 6: All 8 Foundational Root Benchmarks (Zero Regressions)
    # -------------------------------------------------------------
    print("\n--- [TEST SECTION 6: All 8 Foundational Root Benchmarks] ---")
    root_benchmarks = [
        ("The board of directors were divided over the new stratagy, but there was several alternative proposals that lead to a unanimous consensus among each employee.",
         "The board of directors was divided over the new strategy, but there were several alternative proposals that led to a unanimous consensus among each employee."),
        
        ("Neither the lead researcher nor his assistants was satisfied with the initial outcome; however, the research team published their report once the data was fully verified.",
         "Neither the lead researcher nor his assistants were satisfied with the initial outcome; however, the research team published its report once the data were fully verified."),
        
        ("If the financial advisor would have recomended a more conservative portfolio, the client could of avoided severe losses during market fluctuashons, although such unforseeable events tests the expirience of even seasoned investors.",
         "If the financial advisor had recommended a more conservative portfolio, the client could have avoided severe losses during market fluctuations, although such unforeseeable events test the experience of even seasoned investors."),
        
        ("two frnd", "Two friends."),
        ("he is handone", "He is handsome."),
        ("he is strng", "He is strong."),
        ("she is handsome", "She is beautiful."),
        
        ("Every one of the archeologists were astounded by the discovery, because they found a ancient manuscript burried deep within the tomb, which proved that the ancient civilization possessed a highly sophisticated knowledge of astronomie.",
         "Every one of the archaeologists was astounded by the discovery, because he or she found an ancient manuscript buried deep within the tomb, which proved that the ancient civilization possessed a highly sophisticated knowledge of astronomy."),

        ("Neither of the finalistas were able to complete the exam on time, because the questions was highly complex and required a thorough understanding of advanced mathematical principal, which completely overwhelmed their consensus.",
         "Neither of the finalists was able to complete the exam on time, because the questions were highly complex and required a thorough understanding of advanced mathematical principles, which completely overwhelmed their concentration.")
    ]
    for i, (inp, exp) in enumerate(root_benchmarks, 1):
        total_count += 1
        res = detector.detect(inp)
        ok = res.corrected_sentence == exp
        if ok:
            passed_count += 1
            print(f"  ✓ PASSED Root Benchmark {i}: 100% Exact Match")
        else:
            print(f"  ✗ FAILED Root Benchmark {i}:")
            print(f"     Expected: '{exp}'")
            print(f"     Got:      '{res.corrected_sentence}'")

    # -------------------------------------------------------------
    # Category 7: Hard Combined Multi-Layer Complex Sentence
    # -------------------------------------------------------------
    print("\n--- [TEST SECTION 7: Hard Multi-Layer Stress Sentence] ---")
    hard_combined = "Although the committee were skeptical,the two frnd submitted their porject on time."
    res_comb = detector.detect(hard_combined)
    total_count += 1
    # Check key components:
    # 1. Punctuation spacing: ',the' -> ', the'
    # 2. Collective noun: 'committee was'
    # 3. Slang + plural: 'two friends'
    # 4. Typoglycemia: 'project'
    has_spacing = ", the" in res_comb.corrected_sentence
    has_was = "committee was" in res_comb.corrected_sentence
    has_friends = "two friends" in res_comb.corrected_sentence
    has_project = "project" in res_comb.corrected_sentence
    comb_ok = has_spacing and has_was and has_friends and has_project
    if comb_ok:
        passed_count += 1
        print(f"  ✓ PASSED: '{hard_combined}'")
        print(f"    Output: '{res_comb.corrected_sentence}'")
        print(f"    Caught: Spacing ({has_spacing}), SVA ({has_was}), Cardinal Plural ({has_friends}), Typoglycemia ({has_project})")
    else:
        print(f"  ✗ FAILED: '{res_comb.corrected_sentence}'")

    # -------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------
    elapsed = time.time() - t0
    acc = (passed_count / total_count) * 100.0
    print("\n" + "=" * 75)
    print(f"  FINAL SUMMARY: {passed_count}/{total_count} Passed ({acc:.2f}% Accuracy)")
    print(f"  Total Duration: {elapsed:.2f} seconds")
    print("=" * 75)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_hard_suite()
    sys.exit(0 if success else 1)
