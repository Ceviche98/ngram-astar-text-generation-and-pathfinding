#!/usr/bin/env python3
"""
CLI Runner for Topological Graph Pathfinding & Literal Quote Extraction (Tasks 6.1–6.4).

Usage:
    python scripts/run_pathfinding.py
    python scripts/run_pathfinding.py --corpus path/to/1984.txt --sa-iter 10000
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Ensure package root is importable when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from graph_search_nlp import (
    breadth_first_search,
    calculate_path_cost,
    is_simple_path,
    print_expensive_path,
    print_expensive_quote,
    print_long_path,
    print_long_quote,
    process_text_network,
    uniform_cost_search,
)

DEFAULT_SAMPLE_CORPUS = """
It was a bright cold day in April, and the clocks were striking thirteen.
Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,
slipped quickly through the glass doors of Victory Mansions, though not quickly enough
to prevent a swirl of gritty dust from entering along with him.
Perhaps that was less true when you were actually in their hands. One did not know
what happened inside the Ministry of Love, but it was possible to imagine.
Every record has been destroyed or falsified, every book rewritten, every picture
has been repainted, every statue and street building has been renamed, every date
has been altered. And the process is continuing day by day and minute by minute.
History has stopped. Nothing exists except an endless present in which the Party
is always right. People wanted to believe the truth, and the Ministry of Truth
controlled the past and the present in the same way.
Kind words were rare inside the Ministry, yet people still wanted the same truth.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run topological graph search, Simulated Annealing simple-path optimization, and prefix-pruned quote extraction."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path or URL to a text corpus (defaults to built-in Orwellian sample corpus).",
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=0.005,
        help="Relative frequency threshold for <RARE> replacement (default: 0.005).",
    )
    parser.add_argument(
        "--start-word",
        type=str,
        default="kind",
        help="Start node for longest path search (default: 'kind').",
    )
    parser.add_argument(
        "--end-word",
        type=str,
        default="ministry",
        help="End node for longest path search (default: 'ministry').",
    )
    parser.add_argument(
        "--quote-start",
        type=str,
        default="perhaps",
        help="Start node for literal quote extraction (default: 'perhaps').",
    )
    parser.add_argument(
        "--quote-end",
        type=str,
        default="it",
        help="End node for literal quote extraction (default: 'it').",
    )
    parser.add_argument(
        "--sa-iter",
        type=int,
        default=3000,
        help="Simulated annealing iterations (default: 3000).",
    )
    args = parser.parse_args()

    temp_file = None
    try:
        if args.corpus is None:
            fd, temp_file = tempfile.mkstemp(suffix=".txt", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(DEFAULT_SAMPLE_CORPUS)
            source_path = temp_file
        else:
            source_path = args.corpus

        net = process_text_network(
            source_path,
            rare_threshold=args.rare_threshold,
            distance_mode="inverted",
            verbose=False,
        )
        G = net["graph"]
        nodes = net["nodes"]
        dist = net["distance_matrix"]
        node_index = {n: i for i, n in enumerate(nodes)}

        print("=" * 78)
        print("TRANSITION NETWORK TOPOLOGY SUMMARY")
        print("=" * 78)
        print(f"Nodes (active vocabulary): {G.number_of_nodes()}")
        print(f"Edges (unique transitions): {G.number_of_edges()}")
        print(f"Total corpus tokens       : {len(net['original_tokens'])}")

        # 1. Unweighted Shortest Path (BFS) & Minimum-Cost Path (UCS)
        bfs_path, _, _, _ = breadth_first_search(G, args.start_word, args.end_word)
        ucs_path, _, _, _, ucs_cost = uniform_cost_search(
            G, args.start_word, args.end_word, distance_matrix=dist, nodes=nodes
        )
        print("\n[Baseline Exact Search]")
        if bfs_path:
            print(f"  BFS Shortest Path ({len(bfs_path)-1} hops): {' -> '.join(bfs_path)}")
        if ucs_path and ucs_cost is not None:
            print(f"  UCS Min-Cost Path (cost={ucs_cost:.1f}): {' -> '.join(ucs_path)}")

        # 2. Task 6.1: Longest Simple Path via Fixed-Endpoint SA
        long_path = print_long_path(
            net,
            start_word=args.start_word,
            end_word=args.end_word,
            sa_iterations=args.sa_iter,
        )
        print("\n[Task 6.1 — Longest Simple Path (Simulated Annealing)]")
        print(f"  Endpoints     : '{args.start_word}' -> '{args.end_word}'")
        print(f"  Node Count    : {len(long_path)} nodes ({max(len(long_path)-1, 0)} edges)")
        print(f"  Cycle-Free    : {is_simple_path(G, long_path)}")
        print(f"  Path Preview  : {' -> '.join(long_path[:15])}{' ...' if len(long_path) > 15 else ''}")

        # 3. Task 6.2: Longest Contiguous Literal Quote via Prefix-Pruned DFS
        long_quote = print_long_quote(
            net, start_word=args.quote_start, end_word=args.quote_end
        )
        print("\n[Task 6.2 — Longest Contiguous Literal Quote (Prefix-Pruned DFS)]")
        print(f"  Endpoints     : '{args.quote_start}' -> '{args.quote_end}'")
        print(f"  Quote Length  : {len(long_quote)} tokens")
        print(f"  Quote         : {' '.join(long_quote)}")

        # 4. Task 6.3: Most Expensive Simple Path via Cost-Guided SA
        exp_start = "wanted" if "wanted" in G else args.start_word
        exp_end = "same" if "same" in G else args.end_word
        exp_path, exp_cost = print_expensive_path(
            net,
            start_word=exp_start,
            end_word=exp_end,
            sa_iterations=args.sa_iter,
        )
        print("\n[Task 6.3 — Most Expensive Simple Path (Inverted-Cost SA)]")
        print(f"  Endpoints     : '{exp_start}' -> '{exp_end}'")
        print(f"  Total Cost    : {exp_cost:,.1f} ( across {len(exp_path)} nodes )")
        print(f"  Cycle-Free    : {is_simple_path(G, exp_path)}")

        # 5. Task 6.4: Most Expensive Literal Quote
        exp_quote, exp_quote_cost = print_expensive_quote(
            net, start_word=args.quote_start, end_word=args.quote_end
        )
        print("\n[Task 6.4 — Most Expensive Contiguous Literal Quote]")
        print(f"  Endpoints     : '{args.quote_start}' -> '{args.quote_end}'")
        print(f"  Total Cost    : {exp_quote_cost:,.1f} ({len(exp_quote)} tokens)")
        print(f"  Quote         : {' '.join(exp_quote)}")
        print("=" * 78)

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    main()
