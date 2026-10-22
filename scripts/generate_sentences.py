#!/usr/bin/env python3
"""
CLI Runner for Constrained Statistical Text Generation via Length-Normalised A* Search
with Dynamic N-Gram Backoff Costs and Reverse-BFS Distance Heuristics (Tasks 6.5A & 6.5B).

Usage:
    python scripts/generate_sentences.py
    python scripts/generate_sentences.py --completion-prompt "please believe my eyes <CONTENT>."
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from graph_search_nlp import (
    build_ngram_counts,
    complete_sentence,
    process_text_network,
    start_sentence,
)

DEFAULT_GENERATION_CORPUS = """
Please believe my eyes when I tell you the truth.
My eyes saw the truth inside the Ministry of Love.
When I saw the records, my eyes could not believe what happened.
Two people who know the truth can ask for a solution.
Two citizens of Oceania can ask for a better future.
Two members of the party can ask for a solution to the problem.
It was a bright cold day in April, and the clocks were striking thirteen.
Perhaps that was less true when you were actually in their hands.
One did not know what happened inside the Ministry of Love, but it was possible.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrained sentence completion and prefix-conditioned sentence generation via Length-Normalised A* Search."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path or URL to a text corpus (defaults to built-in demonstration corpus).",
    )
    parser.add_argument(
        "--completion-prompt",
        type=str,
        default="please believe my eyes <CONTENT>.",
        help="Prompt for Task 6.5A sentence completion.",
    )
    parser.add_argument(
        "--start-prompt",
        type=str,
        default="two <CONTENT> can ask for a solution.",
        help="Prompt for Task 6.5B sentence starting.",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=0.3,
        help="Convex weight w on h_avg(n) in f_avg(n) = (1-w)*g_avg(n) + w*h_avg(n) (default: 0.3).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=3,
        help="Maximum N-gram backoff order (default: 3).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print step-by-step A* frontier expansion and N-gram backoff traces.",
    )
    args = parser.parse_args()

    temp_file = None
    try:
        if args.corpus is None:
            fd, temp_file = tempfile.mkstemp(suffix=".txt", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(DEFAULT_GENERATION_CORPUS)
            source_path = temp_file
        else:
            source_path = args.corpus

        net = process_text_network(
            source_path,
            rare_threshold=0.001,
            distance_mode="inverted",
            verbose=False,
        )
        net["ngram_counts"] = build_ngram_counts(net["original_tokens"], max_n=args.max_n)

        print("=" * 78)
        print("LENGTH-NORMALISED A* N-GRAM SENTENCE GENERATION (TASKS 6.5A & 6.5B)")
        print("=" * 78)

        completed_tokens = complete_sentence(
            net,
            prompt=args.completion_prompt,
            max_n=args.max_n,
            heuristic_weight=args.heuristic_weight,
            debug=args.debug,
        )
        print(f"\n[Task 6.5A — Sentence Completion]")
        print(f"  Input Prompt : {args.completion_prompt}")
        print(f"  Tokens Out   : {completed_tokens}")
        print(f"  Reconstructed: {' '.join(completed_tokens)}")
        print(f"  Unique Tokens: {len(completed_tokens) == len(set(completed_tokens))}")

        started_tokens = start_sentence(
            net,
            prompt=args.start_prompt,
            max_n=args.max_n,
            heuristic_weight=args.heuristic_weight,
            debug=args.debug,
        )
        print(f"\n[Task 6.5B — Sentence Starting]")
        print(f"  Input Prompt : {args.start_prompt}")
        print(f"  Tokens Out   : {started_tokens}")
        print(f"  Reconstructed: {' '.join(started_tokens)}")
        print(f"  Unique Tokens: {len(started_tokens) == len(set(started_tokens))}")
        print("=" * 78)

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    main()
