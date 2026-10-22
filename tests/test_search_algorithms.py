"""
Unit Tests for Transition Network Construction, Graph Pathfinding, Simulated Annealing,
Prefix-Pruned Quote Search, and Length-Normalised A* Sentence Generation.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from graph_search_nlp import (
    astar_search,
    breadth_first_search,
    build_neighbor_map,
    build_ngram_counts,
    build_quote_prefix_set,
    build_reverse_bfs_heuristic,
    build_unweighted_graph,
    calculate_ngram_cost,
    calculate_path_cost,
    complete_sentence,
    compute_distance_matrix,
    depth_first_search,
    fast_approximate_dfs,
    find_longest_unique_non_rare,
    find_valid_quotes_from_start,
    get_text_adjacencies,
    greedy_search,
    h_shared,
    is_simple_path,
    parse_prompt,
    print_expensive_path,
    print_expensive_quote,
    print_long_path,
    print_long_quote,
    process_text_network,
    replace_rare_tokens,
    run_length_normalised_astar,
    simulated_annealing_cost_fixed_endpoints,
    simulated_annealing_fixed_endpoints,
    start_sentence,
    tokenize_text,
    try_merge_paths,
    uniform_cost_search,
)


SAMPLE_LITERARY_TEXT = """
Please believe my eyes when I tell you the truth.
My eyes saw the truth inside the Ministry of Love.
Two people who know the truth can ask for a solution.
Two members of the party can ask for a solution to the problem.
Perhaps that was less true when you were actually in their hands.
One did not know what happened inside the Ministry of Love, but it was possible.
Kind words were rare inside the Ministry, yet people wanted the same truth.
Wanted by the party, the Ministry remained the same.
"""


class TestNetworkAndNGrams(unittest.TestCase):
    """Tests for tokenization, graph construction, distance matrices, and N-gram backoff."""

    def test_tokenize_and_rare_replacement(self) -> None:
        tokens = tokenize_text("Hello, world! Don't stop, world.")
        self.assertEqual(tokens, ["hello", ",", "world", "don't", "stop", ",", "world", "."])

        new_tokens, rare_set, counts = replace_rare_tokens(tokens, rare_threshold=0.35)
        self.assertIn("hello", rare_set)
        self.assertNotIn("world", rare_set)
        self.assertNotIn(",", rare_set)
        self.assertNotIn(".", rare_set)
        self.assertEqual(counts["world"], 2)

    def test_graph_and_inverted_distance_matrix(self) -> None:
        tokens = ["a", "b", "a", "b", "c", "b"]
        adj = get_text_adjacencies(tokens)
        nodes = ["a", "b", "c"]
        G = build_unweighted_graph(nodes, adj)
        self.assertTrue(G.has_edge("a", "b"))
        self.assertTrue(G.has_edge("b", "c"))
        self.assertFalse(G.has_edge("a", "c"))

        dist_inv, count_sym = compute_distance_matrix(nodes, adj, distance_mode="inverted")
        self.assertEqual(dist_inv.shape, (3, 3))
        np.testing.assert_array_equal(np.diag(dist_inv), np.zeros(3))
        # High co-occurrence ('a', 'b') has strictly lower distance than low co-occurrence ('b', 'c')
        self.assertLess(dist_inv[0, 1], dist_inv[1, 2])

    def test_ngram_counts_and_backoff_cost(self) -> None:
        tokens = ["please", "believe", "my", "eyes", "when", "my", "eyes", "saw", "."]
        counts = build_ngram_counts(tokens, max_n=3)
        self.assertEqual(counts[1]["eyes"], 2)
        self.assertEqual(counts[2][("my", "eyes")], 2)
        self.assertEqual(counts[3][("believe", "my", "eyes")], 1)

        # Observed trigram ("believe", "my", "eyes") has lower cost than unobserved trigram ("when", "my", "eyes")
        cost_trigram, _ = calculate_ngram_cost(["please", "believe", "my"], "eyes", counts, max_n=3)
        cost_backoff, _ = calculate_ngram_cost(["unseen", "my"], "eyes", counts, max_n=3)
        self.assertIsNotNone(cost_trigram)
        self.assertIsNotNone(cost_backoff)
        assert cost_trigram is not None and cost_backoff is not None
        self.assertLess(cost_trigram, cost_backoff)

        # Non-existent bigram returns None for pruning/fallback
        cost_none, _ = calculate_ngram_cost(["please"], "saw", counts, max_n=3)
        self.assertIsNone(cost_none)


class TestClassicalAndHeuristicPathfinding(unittest.TestCase):
    """Tests for BFS, DFS, UCS/Dijkstra, Greedy, A*, and Reverse-BFS Heuristic."""

    def setUp(self) -> None:
        self.G = nx.Graph()
        # Shortest hop path A -> D -> G (2 edges, high cost 10+10=20)
        # Cheapest cost path A -> B -> C -> G (3 edges, low cost 1+1+1=3)
        self.G.add_edge("a", "d", weight=10.0)
        self.G.add_edge("d", "g", weight=10.0)
        self.G.add_edge("a", "b", weight=1.0)
        self.G.add_edge("b", "c", weight=1.0)
        self.G.add_edge("c", "g", weight=1.0)

    def test_bfs_shortest_hop_path(self) -> None:
        path, expanded, _tree, _depth = breadth_first_search(self.G, "a", "g")
        self.assertEqual(path, ["a", "d", "g"])
        self.assertIn("a", expanded)

    def test_dfs_simple_path(self) -> None:
        path, _expanded, _tree, _depth = depth_first_search(self.G, "a", "g")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], "a")
        self.assertEqual(path[-1], "g")
        self.assertTrue(is_simple_path(self.G, path))

    def test_ucs_minimum_cost_path(self) -> None:
        path, _expanded, _tree, _depth, cost = uniform_cost_search(self.G, "a", "g")
        self.assertEqual(path, ["a", "b", "c", "g"])
        self.assertAlmostEqual(cost or 0.0, 3.0)

    def test_shared_neighbor_greedy_and_astar(self) -> None:
        adj = {("a", "b"): 5.0, ("b", "c"): 5.0, ("c", "g"): 5.0, ("a", "d"): 1.0, ("d", "g"): 1.0}
        nodes = list(self.G.nodes())
        nbr_map = build_neighbor_map(nodes, adj)
        self.assertEqual(h_shared("g", "g", nbr_map), 0.0)
        self.assertGreater(h_shared("a", "g", nbr_map), 0.0)

        g_path, _, _, _ = greedy_search(self.G, "a", "g", adj=adj, nodes=nodes)
        self.assertIsNotNone(g_path)

        a_path, _, _, _, a_cost = astar_search(self.G, "a", "g", adj=adj, nodes=nodes)
        self.assertEqual(a_path, ["a", "b", "c", "g"])
        self.assertAlmostEqual(a_cost or 0.0, 3.0)

    def test_reverse_bfs_heuristic_admissibility(self) -> None:
        h_dist = build_reverse_bfs_heuristic(self.G, "g")
        self.assertEqual(h_dist["g"], 0)
        self.assertEqual(h_dist["d"], 1)
        self.assertEqual(h_dist["c"], 1)
        self.assertEqual(h_dist["a"], 2)
        # Triangle inequality / step consistency: |d(u) - d(v)| <= 1 for all edges (u, v)
        for u, v in self.G.edges():
            self.assertLessEqual(abs(h_dist[u] - h_dist[v]), 1)


class TestTask6AlgorithmsOnCorpus(unittest.TestCase):
    """End-to-end tests for Tasks 6.1–6.5 on a literary corpus."""

    def setUp(self) -> None:
        fd, self.temp_path = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE_LITERARY_TEXT)
        self.net = process_text_network(
            self.temp_path, rare_threshold=0.001, distance_mode="inverted", verbose=False
        )

    def tearDown(self) -> None:
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_t6_1_longest_simple_path_and_sa(self) -> None:
        path = print_long_path(
            self.net, start_word="kind", end_word="ministry", sa_iterations=500
        )
        self.assertTrue(len(path) >= 2)
        self.assertEqual(path[0], "kind")
        self.assertEqual(path[-1], "ministry")
        self.assertTrue(is_simple_path(self.net["graph"], path))

    def test_t6_2_longest_literal_quote(self) -> None:
        quote = print_long_quote(self.net, start_word="perhaps", end_word="it")
        expected = [
            "perhaps", "that", "was", "less", "true", "when", "you", "were",
            "actually", "in", "their", "hands", ".", "one", "did", "not",
            "know", "what", "happened", "inside", "the", "ministry", "of",
            "love", ",", "but", "it",
        ]
        self.assertEqual(quote, expected)
        self.assertEqual(len(quote), 27)

    def test_t6_3_most_expensive_simple_path(self) -> None:
        path, cost = print_expensive_path(
            self.net, start_word="wanted", end_word="same", sa_iterations=500
        )
        self.assertTrue(len(path) >= 2)
        self.assertEqual(path[0], "wanted")
        self.assertEqual(path[-1], "same")
        self.assertGreater(cost, 0.0)
        self.assertTrue(is_simple_path(self.net["graph"], path))

    def test_t6_4_most_expensive_literal_quote(self) -> None:
        quote, cost = print_expensive_quote(self.net, start_word="perhaps", end_word="it")
        self.assertEqual(len(quote), 27)
        self.assertGreater(cost, 0.0)

    def test_t6_5a_complete_sentence(self) -> None:
        completed = complete_sentence(
            self.net, prompt="please believe my eyes <CONTENT>."
        )
        self.assertNotIn("<CONTENT>", completed)
        self.assertEqual(completed[:4], ["please", "believe", "my", "eyes"])
        self.assertEqual(completed[-1], ".")
        self.assertGreater(len(completed), 5)
        self.assertEqual(len(completed), len(set(completed)))

    def test_t6_5b_start_sentence_no_repeated_tokens(self) -> None:
        started = start_sentence(
            self.net, prompt="two <CONTENT> can ask for a solution."
        )
        self.assertNotIn("<CONTENT>", started)
        self.assertEqual(started[0], "two")
        self.assertEqual(started[-6:], ["can", "ask", "for", "a", "solution", "."])
        self.assertGreater(len(started), 7)
        # Ensure zero duplicate tokens across the entire reconstructed sentence
        self.assertEqual(len(started), len(set(started)))


if __name__ == "__main__":
    unittest.main()
