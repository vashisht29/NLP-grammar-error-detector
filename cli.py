#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for Deep Learning English Grammatical Error Detection (GED).
Run with single sentences or as an interactive terminal REPL.
"""
import sys
import argparse
from src.model.detector import DeepGrammarDetector


# ANSI terminal colors for beautiful output
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def format_sentence_with_highlights(original: str, errors) -> str:
    """Highlights error spans in bold red inside the original sentence."""
    if not errors:
        return f"{TermColors.GREEN}{original}{TermColors.END}"

    # Sort errors by start_char in reverse to insert ANSI codes cleanly
    chars = list(original)
    for err in sorted(errors, key=lambda e: e.original_span[0], reverse=True):
        start, end = err.original_span
        if start < len(chars) and end <= len(chars) and start < end:
            chars.insert(end, TermColors.END)
            chars.insert(start, f"{TermColors.RED}{TermColors.BOLD}{TermColors.UNDERLINE}")

    return "".join(chars)


def display_result(result):
    """Prints a structured, formatted report of the detection result."""
    print("\n" + "=" * 65)
    if result.is_grammatically_correct:
        print(f"  {TermColors.GREEN}{TermColors.BOLD}✓ NO GRAMMATICAL ERRORS DETECTED{TermColors.END}")
    else:
        print(f"  {TermColors.RED}{TermColors.BOLD}✗ {result.error_count} GRAMMATICAL ERROR(S) DETECTED{TermColors.END}")
    print("=" * 65)

    highlighted_original = format_sentence_with_highlights(result.original_sentence, result.errors)
    print(f"\n{TermColors.BOLD}Original Sentence:{TermColors.END}")
    print(f"  {highlighted_original}")

    if not result.is_grammatically_correct:
        print(f"\n{TermColors.BOLD}Suggested Correction:{TermColors.END}")
        print(f"  {TermColors.GREEN}{TermColors.BOLD}{result.corrected_sentence}{TermColors.END}")

        print(f"\n{TermColors.BOLD}Error Analysis Breakdown:{TermColors.END}")
        for err in result.errors:
            print(f"  [{err.error_id}] {TermColors.YELLOW}{TermColors.BOLD}{err.error_type}{TermColors.END}")
            print(f"      • Original Token: '{TermColors.RED}{err.original_text}{TermColors.END}'")
            print(f"      • Suggestion:     '{TermColors.GREEN}{err.suggested_text}{TermColors.END}'")
            print(f"      • Explanation:    {err.explanation}")
            print(f"      • Confidence:     {err.confidence * 100:.1f}%\n")
    else:
        print(f"\n  {TermColors.CYAN}Sentence is well-formed and grammatically sound.{TermColors.END}")

    print(f"{TermColors.BLUE}─────────────────────────────────────────────────────────────────{TermColors.END}")
    print(f"Backend: {result.model_backend} | Latency: {result.processing_time_ms} ms | Overall Conf: {result.overall_confidence * 100:.1f}%\n")


def interactive_mode(detector: DeepGrammarDetector):
    """Interactive REPL mode for continuous sentence testing."""
    print(f"\n{TermColors.HEADER}{TermColors.BOLD}=== English Grammatical Error Detection (GED) REPL ==={TermColors.END}")
    print("Type any English sentence to detect grammatical errors.")
    print("Type 'exit' or 'quit' (or Ctrl+C) to quit.\n")

    while True:
        try:
            sentence = input(f"{TermColors.BOLD}Input Sentence > {TermColors.END}").strip()
            if not sentence:
                continue
            if sentence.lower() in ("exit", "quit", "q"):
                print("Exiting. Goodbye!")
                break

            result = detector.detect(sentence)
            display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Deep Learning English Grammatical Error Detection CLI"
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        default=None,
        help="Input English sentence to analyze (omit to enter interactive REPL mode)"
    )
    args = parser.parse_args()

    detector = DeepGrammarDetector()

    if args.sentence:
        result = detector.detect(args.sentence)
        display_result(result)
    else:
        interactive_mode(detector)


if __name__ == "__main__":
    main()
