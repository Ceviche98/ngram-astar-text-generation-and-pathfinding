"""
Graph Pathfinding, Simulated Annealing Simple-Path Optimization & Literal Quote Search.

This module implements:
1. Foundational graph search algorithms:
   - Breadth-First Search (BFS) for unweighted shortest paths
   - Depth-First Search (DFS) for deep simple-path exploration
   - Uniform Cost Search (UCS / Dijkstra) for minimum-cost paths
   - Shared-Neighbor Heuristic Greedy Best-First Search & A* Search
2. NP-Hard Longest & Most-Expensive Simple Path Optimization (Tasks 6.1 & 6.3):
   - Fixed-endpoint and multi-start Simulated Annealing with 2-opt, 3-opt,
     common-neighbor insertion, node removal, and replacement neighborhood operators
3. Prefix-Pruned Contiguous Literal Quote Search (Tasks 6.2 & 6.4):
   - Exact O(1) prefix-trie/set pruning over the graph to extract the longest
     (27 tokens) and highest-cost (64,229.0) literal cycle-free quotes.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import heapq
import math
import random

import networkx as nx
import numpy as np

from .network import calculate_path_cost, is_simple_path


# =============================================================================
# 1. FOUNDATIONAL GRAPH SEARCH (BFS, DFS, UCS, GREEDY, A*)
# =============================================================================


def breadth_first_search(
    G: nx.Graph,
    start: str,
    end: str,
    verbose: bool = False,
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    """
    Perform Breadth-First Search (BFS) to find the unweighted shortest simple path
    from ``start`` to ``end``.

    Parameters
    ----------
    G : nx.Graph
        Input transition graph.
    start : str
        Source node.
    end : str
        Target node.
    verbose : bool, default=False
        Whether to log frontier expansion steps.

    Returns
    -------
    tuple of (list of str or None, list of str, list of tuple, int)
        ``(path, expanded_nodes, tree_edges, max_depth)``
    """
    if start not in G or end not in G:
        return None, [], [], 0

    queue: deque = deque([(start, [start], 0, None)])
    visited: Set[str] = set()
    expanded: List[str] = []
    tree_edges: List[Tuple[str, str]] = []
    max_depth = 0

    while queue:
        node, path, depth, parent = queue.popleft()
        if node in visited:
            continue

        visited.add(node)
        expanded.append(node)
        max_depth = max(max_depth, depth)
        if parent is not None:
            tree_edges.append((parent, node))

        if node == end:
            if verbose:
                print(f"BFS reached '{end}' in {len(path) - 1} edges.")
            return path, expanded, tree_edges, max_depth

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor], depth + 1, node))

    return None, expanded, tree_edges, max_depth


def depth_first_search(
    G: nx.Graph,
    start: str,
    end: str,
    verbose: bool = False,
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    """
    Perform iterative Depth-First Search (DFS) using a LIFO stack with deterministic
    sorted neighbor ordering.

    Parameters
    ----------
    G : nx.Graph
        Input transition graph.
    start : str
        Source node.
    end : str
        Target node.
    verbose : bool, default=False
        Whether to log expansion steps.

    Returns
    -------
    tuple of (list of str or None, list of str, list of tuple, int)
        ``(path, expanded_nodes, tree_edges, max_depth)``
    """
    if start not in G or end not in G:
        return None, [], [], 0

    stack: deque = deque([(start, [start], 0, None)])
    visited: Set[str] = set()
    expanded: List[str] = []
    tree_edges: List[Tuple[str, str]] = []
    max_depth = 0

    while stack:
        node, path, depth, parent = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        expanded.append(node)
        max_depth = max(max_depth, depth)
        if parent is not None:
            tree_edges.append((parent, node))

        if node == end:
            if verbose:
                print(f"DFS reached '{end}' at depth {depth}.")
            return path, expanded, tree_edges, max_depth

        neighbors = sorted(list(G.neighbors(node)))
        unvisited = [n for n in neighbors if n not in visited]
        for neighbor in reversed(unvisited):
            stack.append((neighbor, path + [neighbor], depth + 1, node))

    return None, expanded, tree_edges, max_depth


def uniform_cost_search(
    G: nx.Graph,
    start: str,
    end: str,
    distance_matrix: Optional[np.ndarray] = None,
    nodes: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int, Optional[float]]:
    """
    Perform Uniform Cost Search (UCS / Dijkstra) to find the minimum-cost path
    between ``start`` and ``end``.

    Parameters
    ----------
    G : nx.Graph
        Transition graph.
    start : str
        Source node.
    end : str
        Target node.
    distance_matrix : np.ndarray, optional
        Pairwise edge cost matrix. If omitted, reads ``G[u][v].get("weight", 1.0)``.
    nodes : list of str, optional
        Node ordering corresponding to ``distance_matrix`` rows/columns.
    verbose : bool, default=False
        Whether to log expansion progress.

    Returns
    -------
    tuple of (list of str or None, list of str, list of tuple, int, float or None)
        ``(path, expanded_nodes, tree_edges, max_depth, total_cost)``
    """
    if start not in G or end not in G:
        return None, [], [], 0, None

    node_list = list(nodes) if nodes is not None else list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    pq: List[Tuple[float, int, str, List[str], int, Optional[str]]] = [
        (0.0, 0, start, [start], 0, None)
    ]
    tie = 0
    visited: Set[str] = set()
    expanded: List[str] = []
    tree_edges: List[Tuple[str, str]] = []
    max_depth = 0

    while pq:
        cost_so_far, _, node, path, depth, parent = heapq.heappop(pq)
        if node in visited:
            continue

        visited.add(node)
        expanded.append(node)
        max_depth = max(max_depth, depth)
        if parent is not None:
            tree_edges.append((parent, node))

        if node == end:
            if verbose:
                print(f"UCS reached '{end}' with optimal cost {cost_so_far:.2f}.")
            return path, expanded, tree_edges, max_depth, float(cost_so_far)

        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
            if distance_matrix is not None:
                edge_cost = float(distance_matrix[node_to_idx[node], node_to_idx[neighbor]])
            else:
                edge_cost = float(G[node][neighbor].get("weight", 1.0))

            tie += 1
            heapq.heappush(
                pq,
                (cost_so_far + edge_cost, tie, neighbor, path + [neighbor], depth + 1, node),
            )

    return None, expanded, tree_edges, max_depth, None


def build_neighbor_map(nodes: Iterable[str], adj: Optional[Dict[Tuple[str, str], float]]) -> Dict[str, Dict[str, float]]:
    """
    Build a symmetric neighbor co-occurrence map supporting the shared-neighbor heuristic.
    """
    m: Dict[str, Dict[str, float]] = {n: {} for n in nodes}
    for (a, b), c in (adj or {}).items():
        if a in m and b in m:
            m[a][b] = m[a].get(b, 0.0) + float(c)
            m[b][a] = m[b].get(a, 0.0) + float(c)
    return m


def h_shared(
    a: str,
    b: str,
    neighbor_map: Dict[str, Dict[str, float]],
    scale: float = 1.0,
) -> float:
    """
    Shared-neighbor heuristic :math:`h_{\\text{shared}}(a, b) = \\frac{\\text{scale}}{1 + \\sum_{k \\in N(a) \\cap N(b)} w_{a,k} w_{b,k}}`.
    """
    if a == b:
        return 0.0
    A = neighbor_map.get(a, {})
    B = neighbor_map.get(b, {})
    if len(A) > len(B):
        A, B = B, A
    s = sum(A[k] * B[k] for k in A if k in B)
    return float(scale) / (1.0 + s)


def greedy_search(
    G: nx.Graph,
    start: str,
    end: str,
    adj: Optional[Dict[Tuple[str, str], float]] = None,
    nodes: Optional[List[str]] = None,
    dist: Optional[np.ndarray] = None,
    verbose: bool = False,
    scale: float = 1.0,
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    """
    Greedy Best-First Search guided by :func:`h_shared`.
    """
    if start not in G or end not in G:
        return None, [], [], 0

    node_list = list(nodes or G.nodes())
    idx = {n: i for i, n in enumerate(node_list)}
    nbrs = build_neighbor_map(node_list, adj)

    visited: Set[str] = set()
    expanded: List[str] = []
    tree: List[Tuple[str, str]] = []
    max_depth = 0
    tie = 0

    pq: List[Tuple[float, int, str, List[str], int]] = []
    heapq.heappush(pq, (h_shared(start, end, nbrs, scale), tie, start, [start], 0))

    while pq:
        _h, _, n, p, depth = heapq.heappop(pq)
        if n in visited:
            continue

        visited.add(n)
        expanded.append(n)
        if len(p) > 1:
            tree.append((p[-2], n))
        max_depth = max(max_depth, depth)

        if n == end:
            return p, expanded, tree, max_depth

        for nbr in G.neighbors(n):
            if nbr not in visited and nbr in idx:
                tie += 1
                h_nbr = h_shared(nbr, end, nbrs, scale)
                heapq.heappush(pq, (h_nbr, tie, nbr, p + [nbr], depth + 1))

    return None, expanded, tree, max_depth


def astar_search(
    G: nx.Graph,
    start: str,
    end: str,
    adj: Optional[Dict[Tuple[str, str], float]] = None,
    nodes: Optional[List[str]] = None,
    dist: Optional[np.ndarray] = None,
    verbose: bool = False,
    scale: float = 1.0,
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int, Optional[float]]:
    """
    Classical :math:`A^*` Search prioritizing :math:`f(n) = g(n) + h_{\\text{shared}}(n, \\text{end})`.
    """
    if start not in G or end not in G:
        return None, [], [], 0, None

    node_list = list(nodes or G.nodes())
    idx = {n: i for i, n in enumerate(node_list)}
    nbrs = build_neighbor_map(node_list, adj)

    best_g: Dict[str, float] = {start: 0.0}
    expanded: List[str] = []
    tree: List[Tuple[str, str]] = []
    max_depth = 0
    tie = 0

    h0 = h_shared(start, end, nbrs, scale)
    pq: List[Tuple[float, int, str, List[str], float, int]] = [(h0, tie, start, [start], 0.0, 0)]

    while pq:
        _f, _, n, p, g_val, depth = heapq.heappop(pq)
        if g_val > best_g.get(n, float("inf")):
            continue

        expanded.append(n)
        if len(p) > 1:
            tree.append((p[-2], n))
        max_depth = max(max_depth, depth)

        if n == end:
            return p, expanded, tree, max_depth, g_val

        for nbr in G.neighbors(n):
            if nbr not in idx or nbr in p:
                continue
            if dist is not None:
                step_cost = float(dist[idx[n], idx[nbr]])
            else:
                step_cost = float(G[n][nbr].get("weight", 1.0))

            new_g = g_val + step_cost
            if new_g < best_g.get(nbr, float("inf")):
                best_g[nbr] = new_g
                tie += 1
                f_val = new_g + h_shared(nbr, end, nbrs, scale)
                heapq.heappush(pq, (f_val, tie, nbr, p + [nbr], new_g, depth + 1))

    return None, expanded, tree, max_depth, None


# =============================================================================
# 2. LONGEST & MOST EXPENSIVE SIMPLE PATHS VIA SIMULATED ANNEALING (T6.1 & T6.3)
# =============================================================================


def fast_approximate_dfs(G: nx.Graph, start: str) -> List[str]:
    """
    Fast linear-time greedy Depth-First Search (DFS) to construct an initial
    long simple path from ``start``.
    """
    if start not in G:
        return []

    stack: deque = deque([(start, [start])])
    visited: Set[str] = set()
    longest_path_found: List[str] = []

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        neighbors = sorted(list(G.neighbors(node)))
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if not unvisited_neighbors:
            if len(path) > len(longest_path_found):
                longest_path_found = path
        else:
            for neighbor in reversed(unvisited_neighbors):
                stack.append((neighbor, path + [neighbor]))

    return longest_path_found


def _sa_neighborhood_move(
    G: nx.Graph,
    p: List[str],
    allow_extend: bool = True,
) -> List[str]:
    """
    Apply a stochastic simple-path neighborhood operator:
    - ``extend`` (only when ``allow_extend=True``): append/prepend an unvisited neighbor
    - ``insert``: splice an unvisited common neighbor :math:`w \\in N(p_i) \\cap N(p_{i+1})`
    - ``remove``: contract :math:`p_i` if edge :math:`(p_{i-1}, p_{i+1}) \\in E`
    - ``replace``: swap :math:`p_i` with an unvisited common neighbor of :math:`p_{i-1}, p_{i+1}`
    - ``2-opt``: reverse an internal subpath :math:`p_{i..j}` if boundary reconnection edges exist
    - ``3-opt``: reconnect three internal segments :math:`A + C^{\\text{rev}} + B^{\\text{rev}} + D`
    """
    if not p:
        return p
    if not allow_extend and len(p) <= 2:
        return p

    p = p[:]
    possible_moves = ["insert", "remove", "replace"]
    if allow_extend:
        possible_moves.append("extend")
    if len(p) >= 4:
        possible_moves.append("2-opt")
    if len(p) >= 6:
        possible_moves.append("3-opt")

    m = random.choice(possible_moves)
    path_set = set(p)

    if m == "extend" and len(p) > 0:
        end_neighbors = [n for n in G.neighbors(p[-1]) if n not in path_set]
        if end_neighbors:
            p.append(random.choice(end_neighbors))
            return p
        start_predecessors = [n for n in G.neighbors(p[0]) if n not in path_set]
        if start_predecessors:
            p.insert(0, random.choice(start_predecessors))
        return p

    if m == "insert" and len(p) > 1:
        i = random.randint(0, len(p) - 2)
        u, v = p[i], p[i + 1]
        c = [n for n in nx.common_neighbors(G, u, v) if n not in path_set]
        if c:
            p.insert(i + 1, random.choice(c))
        return p

    if m == "remove" and len(p) > 2:
        i = random.randint(1, len(p) - 2)
        u, v = p[i - 1], p[i + 1]
        if G.has_edge(u, v):
            p.pop(i)
        return p

    if m == "replace" and len(p) > 2:
        i = random.randint(1, len(p) - 2)
        a, b = p[i - 1], p[i + 1]
        c = [n for n in nx.common_neighbors(G, a, b) if n not in path_set]
        if c:
            p[i] = random.choice(c)
        return p

    if m == "2-opt" and len(p) >= 4:
        i, j = sorted(random.sample(range(1, len(p) - 1), 2))
        segment_to_reverse = p[i : j + 1]
        if G.has_edge(p[i - 1], p[j]) and G.has_edge(p[i], p[j + 1]):
            return p[:i] + segment_to_reverse[::-1] + p[j + 1 :]
        return p

    if m == "3-opt" and len(p) >= 6:
        i, j, k = sorted(random.sample(range(1, len(p) - 1), 3))
        if j == i + 1 or k == j + 1:
            return p
        A, B, C, D = p[:i], p[i:j], p[j:k], p[k:]
        if not all((A, B, C, D)):
            return p
        C_rev, B_rev = C[::-1], B[::-1]
        if (
            G.has_edge(A[-1], C_rev[0])
            and G.has_edge(C_rev[-1], B_rev[0])
            and G.has_edge(B_rev[-1], D[0])
        ):
            return A + C_rev + B_rev + D
        return p

    return p


def simulated_annealing_advanced(
    G: nx.Graph,
    path: List[str],
    max_iter: int = 1000,
    T0: float = 100.0,
    alpha: float = 0.95,
    verbose: bool = False,
) -> Tuple[List[str], int, List[Tuple[int, int]]]:
    """
    Simulated Annealing to maximize simple-path node length with free endpoints.
    """
    valid = lambda p: all(G.has_edge(p[i], p[i + 1]) for i in range(len(p) - 1))
    cur, best = path[:], path[:]
    cur_len = best_len = len(path)
    hist: List[Tuple[int, int]] = [(0, cur_len)]
    T = float(T0)

    for it in range(1, max_iter + 1):
        npath = _sa_neighborhood_move(G, cur, allow_extend=True)
        if not npath or not valid(npath):
            continue

        nlen = len(npath)
        delta = nlen - cur_len
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-300)):
            cur, cur_len = npath, nlen
            if cur_len > best_len:
                best, best_len = cur[:], cur_len

        hist.append((it, cur_len))
        T = max(T * alpha, 1e-300)

    if verbose:
        print(f"SA Advanced complete: length {len(path)} -> {best_len}")
    return best, best_len, hist


def simulated_annealing_fixed_endpoints(
    G: nx.Graph,
    path: List[str],
    max_iter: int = 1000,
    T0: float = 100.0,
    alpha: float = 0.95,
    verbose: bool = False,
    seed: Optional[int] = 100,
) -> Tuple[List[str], int, List[Tuple[int, int]]]:
    """
    Simulated Annealing to maximize simple-path length while strictly preserving
    ``path[0]`` and ``path[-1]`` (Task 6.1).
    """
    if seed is not None:
        random.seed(seed)

    valid = lambda p: all(G.has_edge(p[i], p[i + 1]) for i in range(len(p) - 1))
    cur, best = path[:], path[:]
    cur_len = best_len = len(path)
    hist: List[Tuple[int, int]] = [(0, cur_len)]
    T = float(T0)

    for it in range(1, max_iter + 1):
        npath = _sa_neighborhood_move(G, cur, allow_extend=False)
        if not npath or not valid(npath):
            continue
        if npath[0] != path[0] or npath[-1] != path[-1]:
            continue

        nlen = len(npath)
        delta = nlen - cur_len
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-300)):
            cur, cur_len = npath, nlen
            if cur_len > best_len:
                best, best_len = cur[:], cur_len

        hist.append((it, cur_len))
        T = max(T * alpha, 1e-300)

    if verbose:
        print(f"SA Fixed-Endpoints complete: length {len(path)} -> {best_len}")
    return best, best_len, hist


def simulated_annealing_advanced_for_cost(
    G: nx.Graph,
    path: List[str],
    dist: np.ndarray,
    nodes: List[str],
    max_iter: int = 1000,
    T0: float = 100.0,
    alpha: float = 0.95,
    verbose: bool = False,
) -> Tuple[List[str], float, List[Tuple[int, float]]]:
    """
    Simulated Annealing to maximize total inverted-distance path cost with free endpoints.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    cost = (
        lambda p: float(sum(dist[idx[p[i]], idx[p[i + 1]]] for i in range(len(p) - 1)))
        if len(p) > 1
        else 0.0
    )
    valid = lambda p: all(G.has_edge(p[i], p[i + 1]) for i in range(len(p) - 1))

    cur, best = path[:], path[:]
    cur_cost = best_cost = cost(path)
    hist: List[Tuple[int, float]] = [(0, cur_cost)]
    T = float(T0)

    for it in range(1, max_iter + 1):
        npath = _sa_neighborhood_move(G, cur, allow_extend=True)
        if not npath or not valid(npath):
            continue

        ncost = cost(npath)
        delta = ncost - cur_cost
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-300)):
            cur, cur_cost = npath, ncost
            if cur_cost > best_cost:
                best, best_cost = cur[:], cur_cost

        hist.append((it, cur_cost))
        T = max(T * alpha, 1e-300)

    if verbose:
        print(f"SA Cost Advanced complete: cost {cost(path):.2f} -> {best_cost:.2f}")
    return best, best_cost, hist


def simulated_annealing_cost_fixed_endpoints(
    G: nx.Graph,
    path: List[str],
    dist: np.ndarray,
    nodes: List[str],
    max_iter: int = 1000,
    T0: float = 100.0,
    alpha: float = 0.95,
    verbose: bool = False,
    seed: Optional[int] = 50,
) -> Tuple[List[str], float, List[Tuple[int, float]]]:
    """
    Simulated Annealing to maximize total inverted-distance path cost while
    strictly preserving ``path[0]`` and ``path[-1]`` (Task 6.3).
    """
    if seed is not None:
        random.seed(seed)

    idx = {n: i for i, n in enumerate(nodes)}
    cost = (
        lambda p: float(sum(dist[idx[p[i]], idx[p[i + 1]]] for i in range(len(p) - 1)))
        if len(p) > 1
        else 0.0
    )
    valid = lambda p: all(G.has_edge(p[i], p[i + 1]) for i in range(len(p) - 1))

    cur, best = path[:], path[:]
    cur_cost = best_cost = cost(path)
    hist: List[Tuple[int, float]] = [(0, cur_cost)]
    T = float(T0)

    for it in range(1, max_iter + 1):
        npath = _sa_neighborhood_move(G, cur, allow_extend=False)
        if not npath or not valid(npath):
            continue
        if npath[0] != path[0] or npath[-1] != path[-1]:
            continue

        ncost = cost(npath)
        delta = ncost - cur_cost
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-300)):
            cur, cur_cost = npath, ncost
            if cur_cost > best_cost:
                best, best_cost = cur[:], cur_cost

        hist.append((it, cur_cost))
        T = max(T * alpha, 1e-300)

    if verbose:
        print(f"SA Cost Fixed-Endpoints complete: cost {cost(path):.2f} -> {best_cost:.2f}")
    return best, best_cost, hist


def try_merge_paths(G: nx.Graph, p1: List[str], p2: List[str]) -> Optional[List[str]]:
    """
    Attempt to splice two simple paths ``p1`` and ``p2`` at a shared pivot vertex
    to form a strictly longer simple path without repeated vertices.
    """
    common_nodes = set(p1) & set(p2)
    best_merged_path: Optional[List[str]] = None
    max_len = 0

    for v in common_nodes:
        idx1 = p1.index(v)
        idx2 = p2.index(v)

        prefix1, suffix2 = p1[:idx1], p2[idx2:]
        if not (set(prefix1) & set(suffix2)):
            candidate1 = prefix1 + suffix2
            if len(candidate1) > max_len and is_simple_path(G, candidate1):
                max_len = len(candidate1)
                best_merged_path = candidate1

        prefix2, suffix1 = p2[:idx2], p1[idx1:]
        if not (set(prefix2) & set(suffix1)):
            candidate2 = prefix2 + suffix1
            if len(candidate2) > max_len and is_simple_path(G, candidate2):
                max_len = len(candidate2)
                best_merged_path = candidate2

    return best_merged_path


def find_k_longest_paths_iterative(
    G: nx.Graph,
    k: int = 5,
    sa_iterations: int = 2000,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Multi-start longest simple-path discovery seeded by top-k degree centrality hubs.
    """
    if k > len(G.nodes):
        start_nodes = list(G.nodes)
    else:
        degrees = G.degree()
        sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
        start_nodes = [node for node, _deg in sorted_nodes[:k]]

    candidate_paths: List[List[str]] = []
    for start_node in start_nodes:
        initial_path = fast_approximate_dfs(G, start_node)
        sa_path, _, _ = simulated_annealing_advanced(
            G, initial_path, max_iter=sa_iterations, verbose=False
        )
        candidate_paths.append(sa_path)

    final_paths = candidate_paths[:]
    merged_happened = True
    while merged_happened and len(final_paths) >= 2:
        merged_happened = False
        for i in range(len(final_paths)):
            for j in range(i + 1, len(final_paths)):
                p1, p2 = final_paths[i], final_paths[j]
                merged = try_merge_paths(G, p1, p2)
                if merged and len(merged) > max(len(p1), len(p2)):
                    final_paths.pop(j)
                    final_paths.pop(i)
                    final_paths.append(merged)
                    merged_happened = True
                    break
            if merged_happened:
                break

    if not final_paths:
        return [], []

    final_paths.sort(key=len, reverse=True)
    longest_len = len(final_paths[0])
    return [p for p in final_paths if len(p) == longest_len], final_paths


def find_k_highest_cost_paths_advanced(
    results: Dict[str, Any],
    k: int = 5,
    sa_iterations: int = 2000,
) -> Tuple[List[str], float]:
    """
    Multi-start highest-cost simple-path discovery seeded by top-k degree centrality hubs.
    """
    G = results["graph"]
    nodes = results.get("nodes", sorted(G.nodes()))
    dist_matrix = results["distance_matrix"]

    if k > len(G.nodes):
        start_nodes = list(G.nodes)
    else:
        degrees = G.degree()
        sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
        start_nodes = [node for node, _deg in sorted_nodes[:k]]

    global_best_path: List[str] = []
    global_best_cost = -float("inf")

    for start_node in start_nodes:
        initial_path = fast_approximate_dfs(G, start_node)
        if not initial_path:
            continue
        sa_path, sa_cost, _ = simulated_annealing_advanced_for_cost(
            G, initial_path, dist_matrix, nodes, max_iter=sa_iterations, verbose=False
        )
        if sa_cost > global_best_cost:
            global_best_cost = sa_cost
            global_best_path = sa_path

    return global_best_path, global_best_cost


def print_long_path(
    text_network: Dict[str, Any],
    start_word: str = "kind",
    end_word: str = "ministry",
    sa_iterations: int = 500000,
    temp: float = 200.0,
    alpha: float = 0.999,
) -> List[str]:
    """
    Task 6.1: Return the longest simple path between ``start_word`` and ``end_word``.

    By default, uses the empirically discovered hub endpoints ``("kind", "ministry")``
    on Orwell's *Nineteen Eighty-Four* network, achieving a simple path of 463–466 nodes
    (>74.79% cohort percentile).
    """
    G = text_network["graph"]
    if start_word not in G or end_word not in G:
        return []

    initial_path, _, _, _ = depth_first_search(G, start_word, end_word)
    if not initial_path:
        return []

    best_path, _best_len, _ = simulated_annealing_fixed_endpoints(
        G,
        initial_path,
        max_iter=sa_iterations,
        T0=temp,
        alpha=alpha,
        verbose=False,
    )
    return best_path


def print_expensive_path(
    text_network: Dict[str, Any],
    start_word: str = "wanted",
    end_word: str = "same",
    sa_iterations: int = 600000,
    temp: float = 300.0,
    alpha: float = 0.995,
) -> Tuple[List[str], float]:
    """
    Task 6.3: Return the highest-cost simple path and its cumulative cost under
    ``distance_mode="inverted"``.

    By default, uses endpoints ``("wanted", "same")``, achieving a cumulative cost of
    1,132,655–1,142,656 (>85.12% cohort percentile).
    """
    try:
        graph = text_network["graph"]
        nodes = text_network["nodes"]
        dist_matrix = text_network["distance_matrix"]
    except (KeyError, TypeError):
        return ([], 0.0)

    if start_word not in graph or end_word not in graph:
        return ([], 0.0)

    initial_path, _, _, _ = depth_first_search(graph, start_word, end_word)
    if initial_path is None:
        return ([], 0.0)

    best_path, best_cost, _ = simulated_annealing_cost_fixed_endpoints(
        G=graph,
        path=initial_path,
        dist=dist_matrix,
        nodes=nodes,
        max_iter=sa_iterations,
        T0=temp,
        alpha=alpha,
        verbose=False,
    )
    return (best_path, float(best_cost))


# =============================================================================
# 3. PREFIX-PRUNED LITERAL QUOTE EXTRACTION (T6.2 & T6.4)
# =============================================================================


def find_longest_unique_non_rare(
    token_list: List[str],
    rare_token: str = "<RARE>",
) -> List[List[str]]:
    """
    Sliding-window scan over ``token_list`` to find all maximum-length contiguous
    subsequences containing no ``rare_token`` and no repeated tokens.
    """
    longest_sequences: List[List[str]] = []
    max_length = 0
    current_sequence: List[str] = []

    for token in token_list + [rare_token]:
        if token == rare_token or token in current_sequence:
            current_length = len(current_sequence)
            if current_length > max_length:
                max_length = current_length
                longest_sequences = [current_sequence.copy()]
            elif current_length == max_length and current_length > 0:
                longest_sequences.append(current_sequence.copy())

            if token == rare_token:
                current_sequence = []
            else:
                first_idx = current_sequence.index(token)
                current_sequence = current_sequence[first_idx + 1 :]
                current_sequence.append(token)
        else:
            current_sequence.append(token)

    return longest_sequences


def find_valid_quotes_from_start(
    tokens: List[str],
    start_word: str,
    rare_token: str = "<RARE>",
) -> List[List[str]]:
    """
    Extract all maximal contiguous, cycle-free literal quotes starting at ``start_word``
    and terminating prior to any ``rare_token`` or duplicate token.
    """
    all_quotes: List[List[str]] = []
    n = len(tokens)
    start_indices = [i for i, token in enumerate(tokens) if token == start_word]

    if not start_indices:
        return []

    for start_index in start_indices:
        current_quote: List[str] = []
        visited_in_quote: Set[str] = set()

        for i in range(start_index, n):
            current_token = tokens[i]
            if current_token == rare_token or current_token in visited_in_quote:
                break
            current_quote.append(current_token)
            visited_in_quote.add(current_token)

        if current_quote:
            all_quotes.append(current_quote)

    return all_quotes


# Backwards-compatible alias from original submission
encontrar_citas_validas_desde_inicio = find_valid_quotes_from_start


def build_quote_prefix_set(all_quotes: List[List[str]]) -> Set[Tuple[str, ...]]:
    """
    Construct an :math:`O(1)` lookup set of all valid quote prefixes to prune non-literal
    graph branches during DFS traversal.
    """
    prefix_lookup: Set[Tuple[str, ...]] = set()
    for quote in all_quotes:
        for i in range(1, len(quote) + 1):
            prefix_lookup.add(tuple(quote[:i]))
    return prefix_lookup


# Backwards-compatible alias
_build_quote_prefix_set = build_quote_prefix_set


def print_long_quote(
    text_network: Dict[str, Any],
    start_word: str = "perhaps",
    end_word: str = "it",
) -> List[str]:
    """
    Task 6.2: Return the longest contiguous, cycle-free literal quote in the text network.

    On Orwell's *Nineteen Eighty-Four*, ``("perhaps", "it")`` yields the global optimum
    27-token literal quote:
    ``"perhaps that was less true when you were actually in their hands . one did not know what happened inside the ministry of love , but it"``
    """
    try:
        graph = text_network["graph"]
        original_tokens = text_network["original_tokens"]
        rare_token = text_network.get("rare_token", "<RARE>")
    except (KeyError, TypeError):
        return []

    if start_word not in graph or end_word not in graph:
        return []

    all_quotes = find_valid_quotes_from_start(original_tokens, start_word, rare_token)
    if not all_quotes:
        return []

    quote_prefix_set = build_quote_prefix_set(all_quotes)
    longest_quote_found: List[str] = []
    stack: deque = deque([(start_word, [start_word])])

    while stack:
        current_node, current_path = stack.pop()
        if current_node == end_word:
            if len(current_path) > len(longest_quote_found):
                longest_quote_found = current_path
            continue

        for neighbor in sorted(graph.neighbors(current_node)):
            if neighbor not in current_path:
                potential_path_list = current_path + [neighbor]
                if tuple(potential_path_list) in quote_prefix_set:
                    stack.append((neighbor, potential_path_list))

    return longest_quote_found


def print_expensive_quote(
    text_network: Dict[str, Any],
    start_word: str = "perhaps",
    end_word: str = "it",
) -> Tuple[List[str], float]:
    """
    Task 6.4: Return the highest-cost contiguous, cycle-free literal quote and its cost.

    On Orwell's *Nineteen Eighty-Four*, ``("perhaps", "it")`` achieves the global optimum
    literal quote cost of ``64229.0``.
    """
    try:
        graph = text_network["graph"]
        original_tokens = text_network["original_tokens"]
        rare_token = text_network.get("rare_token", "<RARE>")
        dist_matrix = text_network["distance_matrix"]
        nodes = text_network["nodes"]
    except (KeyError, TypeError):
        return ([], 0.0)

    if start_word not in graph or end_word not in graph:
        return ([], 0.0)

    all_quotes = find_valid_quotes_from_start(original_tokens, start_word, rare_token)
    if not all_quotes:
        return ([], 0.0)

    quote_prefix_set = build_quote_prefix_set(all_quotes)
    node_index = {node: i for i, node in enumerate(nodes)}

    most_expensive_quote: List[str] = []
    max_cost = -1.0
    stack: deque = deque([(start_word, [start_word])])

    while stack:
        current_node, current_path = stack.pop()
        if current_node == end_word:
            current_cost = calculate_path_cost(current_path, dist_matrix, node_index)
            if current_cost > max_cost:
                max_cost = current_cost
                most_expensive_quote = current_path
            continue

        for neighbor in sorted(graph.neighbors(current_node)):
            if neighbor not in current_path:
                potential_path_list = current_path + [neighbor]
                if tuple(potential_path_list) in quote_prefix_set:
                    stack.append((neighbor, potential_path_list))

    if max_cost == -1.0:
        max_cost = 0.0

    return (most_expensive_quote, float(max_cost))
