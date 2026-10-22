"""
Transition Network Construction & Statistical N-Gram Language Modeling.

This module implements the text-to-graph pipeline that transforms a literary
corpus (such as George Orwell's Nineteen Eighty-Four) into a topological word
transition network and precomputes N-gram frequency distributions and
inverted co-occurrence cost matrices.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests


def load_from_source(source: str) -> bytes:
    """
    Load raw bytes from a local file path or HTTP/HTTPS URL.

    Parameters
    ----------
    source : str
        Local file path or remote URL.

    Returns
    -------
    bytes
        Raw binary content of the resource.
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return response.content
    with open(source, "rb") as f:
        return f.read()


def tokenize_text(text: str, char_like: Optional[List[str]] = None) -> List[str]:
    """
    Tokenize raw text into lowercase words and standalone punctuation tokens (',' and '.').

    Alphabetic characters and any character in ``char_like`` (default: ``["'"]``)
    are grouped into contiguous word tokens. Commas and periods are emitted as
    individual tokens, while whitespace, digits, and other symbols act as
    token delimiters.

    Parameters
    ----------
    text : str
        Input text string to tokenize.
    char_like : list of str, optional
        Characters treated as word-internal letters (defaults to ``["'"]``).

    Returns
    -------
    list of str
        Ordered sequence of tokens.
    """
    if char_like is None:
        char_like = ["'"]

    text = text.lower()
    tokens: List[str] = []
    current_word: List[str] = []
    punctuation = {",", "."}

    for char in text:
        if char.isalpha() or char in char_like:
            current_word.append(char)
        elif char in punctuation:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            tokens.append(char)
        else:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []

    if current_word:
        tokens.append("".join(current_word))

    return tokens


def replace_rare_tokens(
    tokens: List[str],
    rare_threshold: float = 0.01,
    rare_token: str = "<RARE>",
) -> Tuple[List[str], Set[str], Counter]:
    """
    Replace words whose relative frequency falls below ``rare_threshold`` with ``rare_token``.

    Punctuation tokens (``,`` and ``.``) are exempt from replacement regardless of frequency.

    Parameters
    ----------
    tokens : list of str
        Ordered list of tokens produced by :func:`tokenize_text`.
    rare_threshold : float, default=0.01
        Minimum relative word frequency required to retain a word in the active vocabulary.
    rare_token : str, default="<RARE>"
        Sentinel token substituted for low-frequency words.

    Returns
    -------
    tuple of (list of str, set of str, Counter)
        ``(new_tokens, rare_set, final_counts)``
    """
    punctuation = {",", "."}
    word_tokens = [t for t in tokens if t not in punctuation]
    total_words = len(word_tokens)

    if total_words == 0:
        return tokens[:], set(), Counter(tokens)

    word_counts = Counter(word_tokens)
    rare_set = {
        word
        for word, count in word_counts.items()
        if (count / total_words) < rare_threshold
    }
    new_tokens = [rare_token if word in rare_set else word for word in tokens]
    final_counts = Counter(new_tokens)

    return new_tokens, rare_set, final_counts


def get_text_adjacencies(tokens: List[str]) -> Counter:
    """
    Count directional consecutive token transitions ``(tokens[i], tokens[i+1])``,
    excluding immediate self-loops ``tokens[i] == tokens[i+1]``.

    Parameters
    ----------
    tokens : list of str
        Ordered token sequence.

    Returns
    -------
    Counter
        Mapping from ``(token_a, token_b)`` pairs to transition counts.
    """
    adjacency_counts: Counter = Counter()
    for i in range(len(tokens) - 1):
        current_token = tokens[i]
        next_token = tokens[i + 1]
        if current_token != next_token:
            adjacency_counts[(current_token, next_token)] += 1
    return adjacency_counts


def build_unweighted_graph(nodes: Iterable[str], adjacency_counts: Counter) -> nx.Graph:
    """
    Construct an undirected, unweighted NetworkX graph from token adjacency counts.

    Parameters
    ----------
    nodes : iterable of str
        Vocabulary nodes to include in the graph.
    adjacency_counts : Counter
        Mapping of ``(u, v)`` pairs to co-occurrence counts.

    Returns
    -------
    nx.Graph
        Undirected simple graph over ``nodes``.
    """
    node_set = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for (a, b), _count in adjacency_counts.items():
        if a in node_set and b in node_set and not G.has_edge(a, b):
            G.add_edge(a, b)

    return G


def compute_distance_matrix(
    nodes: List[str],
    adjacency_counts: Counter,
    distance_mode: str = "inverted",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the symmetrized transition count matrix and pairwise edge distance matrix.

    For ``distance_mode="inverted"``, the symmetrized count matrix
    :math:`C_{\\text{sym}} = \\frac{1}{2}(C + C^\\top)` with zeroed diagonal is transformed into:

    .. math::
        D_{i,j} = (\\max_{u,v} C_{\\text{sym}}(u,v) + 1) - C_{\\text{sym}}(i,j), \\quad D_{i,i} = 0

    Parameters
    ----------
    nodes : list of str
        Ordered list of node labels defining row/column indices.
    adjacency_counts : Counter
        Directional bigram counts ``(u, v) -> count``.
    distance_mode : {"inverted", "direct"}, default="inverted"
        Transformation mode converting co-occurrence counts into edge costs.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(distance_matrix, count_matrix)`` of shape ``(n, n)``.
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    count_matrix = np.zeros((n, n), dtype=float)

    for (a, b), count in adjacency_counts.items():
        if a in node_index and b in node_index:
            count_matrix[node_index[a], node_index[b]] = count

    count_matrix = (count_matrix + count_matrix.T) / 2.0
    np.fill_diagonal(count_matrix, 0.0)

    if distance_mode == "direct":
        distance_matrix = count_matrix.copy()
    elif distance_mode == "inverted":
        max_count = np.max(count_matrix) if n > 0 else 0.0
        if max_count > 0:
            distance_matrix = (max_count + 1.0) - count_matrix
        else:
            distance_matrix = np.ones_like(count_matrix)
    else:
        raise ValueError("distance_mode must be 'direct' or 'inverted'")

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix, count_matrix


def process_text_network(
    source: str,
    rare_threshold: float = 0.01,
    rare_token: str = "<RARE>",
    distance_mode: str = "inverted",
    verbose: bool = False,
    nsample_tokens: int = 20,
) -> Dict[str, Any]:
    """
    End-to-end text network construction pipeline.

    Parameters
    ----------
    source : str
        File path or URL to the raw corpus text.
    rare_threshold : float, default=0.01
        Relative frequency threshold for ``<RARE>`` token substitution.
    rare_token : str, default="<RARE>"
        Replacement token for infrequent words.
    distance_mode : str, default="inverted"
        Cost mode passed to :func:`compute_distance_matrix`.
    verbose : bool, default=False
        Whether to print corpus statistics during processing.
    nsample_tokens : int, default=20
        Number of sample tokens printed when ``verbose=True``.

    Returns
    -------
    dict
        Dictionary containing:
        - ``graph``: ``nx.Graph``
        - ``nodes``: ``List[str]`` sorted by descending frequency
        - ``adjacency_counts``: ``Counter``
        - ``distance_matrix``: ``np.ndarray``
        - ``count_matrix``: ``np.ndarray``
        - ``token_counts``: ``Counter``
        - ``rare_tokens``: ``Set[str]``
        - ``original_tokens``: ``List[str]`` (pre-replacement tokens)
    """
    content = load_from_source(source)
    text = content.decode("utf-8", errors="ignore")
    tokens = tokenize_text(text)

    processed_tokens, rare_set, token_counts = replace_rare_tokens(
        tokens, rare_threshold=rare_threshold, rare_token=rare_token
    )

    adjacency_counts = get_text_adjacencies(processed_tokens)
    nodes = sorted(token_counts.keys(), key=lambda x: (-token_counts[x], x))
    graph = build_unweighted_graph(nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(
        nodes, adjacency_counts, distance_mode=distance_mode
    )

    if verbose:
        print(f"Loaded text: {len(text)} characters, {len(tokens)} total tokens")
        print(f"Replaced {len(rare_set)} rare words (threshold={rare_threshold})")
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Sample tokens: {list(set(processed_tokens))[:nsample_tokens]}")

    return {
        "graph": graph,
        "nodes": nodes,
        "adjacency_counts": adjacency_counts,
        "distance_matrix": distance_matrix,
        "count_matrix": count_matrix,
        "token_counts": token_counts,
        "rare_tokens": rare_set,
        "original_tokens": tokens,
    }


def build_ngram_counts(tokens: List[str], max_n: int = 3) -> Dict[int, Counter]:
    """
    Precompute contiguous N-gram frequency tables for :math:`N \\in \\{1, \\dots, \\text{max\\_n}\\}`.

    Unigrams (:math:`N=1`) are stored as string keys ``w_i``, whereas higher-order
    N-grams (:math:`N \\ge 2`) are stored as tuples ``(w_i, ..., w_{i+n-1})``.

    Parameters
    ----------
    tokens : list of str
        Ordered list of corpus tokens (typically ``original_tokens``).
    max_n : int, default=3
        Maximum N-gram order to extract.

    Returns
    -------
    dict of int -> Counter
        Mapping from N-gram order ``n`` to its frequency ``Counter``.
    """
    ngram_counts: Dict[int, Counter] = {n: Counter() for n in range(1, max_n + 1)}
    num_tokens = len(tokens)

    for i in range(num_tokens):
        for n in range(1, max_n + 1):
            if i + n <= num_tokens:
                ngram_tuple = tuple(tokens[i : i + n])
                if n == 1:
                    ngram_counts[1][ngram_tuple[0]] += 1
                else:
                    ngram_counts[n][ngram_tuple] += 1
            else:
                break

    return ngram_counts


# Backwards-compatible alias
_build_ngram_counts = build_ngram_counts


def calculate_ngram_cost(
    path: List[str],
    next_node: str,
    ngram_counts: Dict[int, Counter],
    max_n: int = 3,
    missing_ngram_penalty: float = 2.0,
) -> Tuple[Optional[float], str]:
    """
    Compute the dynamic N-gram backoff transition cost for appending ``next_node`` to ``path``.

    Mathematical Formulation (Task 6.5C)
    ------------------------------------
    1. **Base Bigram Surprisal**:
       .. math::
           c_2(w_{t-1}, w_t) = -\\ln\\!\\left(\\frac{C(w_{t-1}, w_t)}{C(w_{t-1})} + 10^{-10}\\right)
       If :math:`C(w_{t-1}, w_t) = 0`, returns ``(None, "Bigram does not exist")`` so the
       caller can either prune or apply a non-literal graph-edge fallback penalty.

    2. **Higher-Order N-Gram Augmentation (:math:`n = 3, \\dots, \\text{max\\_n}`)**:
       For each context length :math:`n-1` available in ``path``:
       - If the :math:`n`-gram occurs in the corpus (:math:`C(w_{t-n+1:t}) > 0`):
         adds :math:`-\\ln(P(w_t \\mid w_{t-n+1:t-1}) + 10^{-10})`.
       - Otherwise, adds a fixed backoff penalty :math:`\\lambda_{\\text{backoff}} = 2.0`.

    Parameters
    ----------
    path : list of str
        Current token sequence ending at :math:`w_{t-1}`.
    next_node : str
        Candidate next token :math:`w_t`.
    ngram_counts : dict of int -> Counter
        Precomputed N-gram frequency tables from :func:`build_ngram_counts`.
    max_n : int, default=3
        Maximum N-gram order evaluated.
    missing_ngram_penalty : float, default=2.0
        Fixed penalty added when a higher-order N-gram (:math:`n \\ge 3`) is unobserved.

    Returns
    -------
    tuple of (float or None, str)
        ``(step_cost, debug_trace)`` where ``step_cost`` is ``None`` if the base bigram
        never appears in the corpus.
    """
    current_node_str = path[-1]
    bigram_tuple = (current_node_str, next_node)
    bigram_count = ngram_counts[2].get(bigram_tuple, 0)
    bigram_context_count = ngram_counts[1].get(current_node_str, 0)

    if bigram_context_count == 0 or bigram_count == 0:
        return None, "Bigram does not exist"

    base_cost = -math.log(bigram_count / bigram_context_count + 1e-10)
    total_cost = base_cost
    debug_str = f"(base P(B|A)={base_cost:.2f}"

    for n in range(3, max_n + 1):
        context_len = n - 1
        if len(path) < context_len:
            debug_str += f" + skip(N={n})"
            continue

        context_tuple = tuple(path[-context_len:])
        ngram_tuple = context_tuple + (next_node,)
        ngram_count = ngram_counts[n].get(ngram_tuple, 0)
        context_count = ngram_counts[n - 1].get(context_tuple, 0)

        if context_count > 0 and ngram_count > 0:
            probability = ngram_count / context_count
            cost_bonus = -math.log(probability + 1e-10)
            total_cost += cost_bonus
            debug_str += f" + P(N={n})={cost_bonus:.2f}"
        else:
            total_cost += missing_ngram_penalty
            debug_str += f" + pen(N={n})"

    debug_str += ")"
    return total_cost, debug_str


# Backwards-compatible alias
_calculate_ngram_cost = calculate_ngram_cost


def is_simple_path(G: nx.Graph, path: List[str]) -> bool:
    """
    Verify that ``path`` visits each node at most once (simple path) and that
    all consecutive edges ``(path[i], path[i+1])`` exist in ``G``.
    """
    if not path:
        return True
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return False
    return True


def calculate_path_cost(
    path: List[str],
    dist_matrix: np.ndarray,
    node_index: Dict[str, int],
) -> float:
    """
    Compute the cumulative edge cost of ``path`` using ``dist_matrix`` and ``node_index``.
    """
    if len(path) < 2:
        return 0.0
    total_cost = 0.0
    for i in range(len(path) - 1):
        token_a = path[i]
        token_b = path[i + 1]
        if token_a in node_index and token_b in node_index:
            total_cost += float(dist_matrix[node_index[token_a], node_index[token_b]])
    return total_cost


# Backwards-compatible alias
_calculate_path_cost = calculate_path_cost


def visualize_network(
    G: nx.Graph,
    distance_matrix: np.ndarray,
    nodes: List[str],
    node_colors: Optional[List[str]] = None,
    node_labels: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 14),
    title: str = "Text Transition Network Graph",
    jitter: float = 0.2,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> None:
    """
    Render the word transition network in a jittered matrix-grid layout with edge costs.
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    side = int(np.ceil(np.sqrt(max(n, 1))))

    pos = {}
    rng = np.random.RandomState(random_state)
    for i, node in enumerate(nodes):
        row = i // side
        col = i % side
        pos[node] = (
            col + rng.uniform(-jitter, jitter),
            -row + rng.uniform(-jitter, jitter),
        )

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title)

    if node_colors is None:
        node_colors = ["lightblue"] * n
    if node_labels is None:
        node_labels = {node: str(node)[:10] for node in nodes}

    nx.draw_networkx_nodes(
        G, pos, node_size=900, node_color=node_colors, edgecolors="black", linewidths=1, ax=ax
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)

    edge_labels = {}
    for u, v in G.edges():
        if u in node_index and v in node_index:
            dist_val = distance_matrix[node_index[u], node_index[v]]
            edge_labels[(u, v)] = f"{dist_val:.1f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax, label_pos=0.5)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
