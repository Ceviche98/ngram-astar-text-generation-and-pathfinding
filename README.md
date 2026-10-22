# Topological Text Networks, Simulated Annealing Pathfinding & Length-Normalised $A^*$ N-Gram Sentence Generation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/Graph%20Engine-NetworkX%203.x-11557C)](https://networkx.org/)
[![Cohort Benchmark](https://img.shields.io/badge/Cohort%20Benchmark-Top%2015%25-7c3aed)](#5-quantitative-benchmarks--empirical-results)
[![Grammar Percentile](https://img.shields.io/badge/Grammar%20Percentile-98.99%25-059669)](#5-quantitative-benchmarks--empirical-results)
[![Perplexity Percentile](https://img.shields.io/badge/Perplexity%20Percentile-85.99%25-0d9488)](#5-quantitative-benchmarks--empirical-results)
[![Path Cost Percentile](https://img.shields.io/badge/Path%20Cost%20Percentile-85.12%25-2563eb)](#5-quantitative-benchmarks--empirical-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-grade graph search and statistical natural language generation library** modeling literary corpora (George Orwell's *Nineteen Eighty-Four*) as weighted topological word-transition networks. Implements exact unweighted/weighted shortest-path search (**BFS**, **UCS/Dijkstra**), metaheuristic **Simulated Annealing** with $k$-opt neighborhood operators for NP-hard longest and maximum-cost simple paths, **$O(1)$ prefix-pruned DFS** for optimal literal quote extraction, and **Length-Normalised $A^*$ Search** with dynamic **N-gram backoff costs** and **Reverse-BFS lookahead heuristics** for constrained sentence generation.

---

## 1. Executive Summary & System Architecture

When a natural language corpus $\mathcal{D} = (w_1, w_2, \dots, w_M)$ is projected onto a topological transition network $G = (V, E, W)$, classical graph pathfinding and statistical sequence generation become dual formulations of the same underlying state-space search problem:

### 1. Extremal Simple-Path Optimization (Combinatorial Graph Search)
Finding the longest cycle-free path or highest-cost simple path in a general cyclic graph is **NP-hard** (reducible from the Hamiltonian Path problem):

$$\pi^* = \arg\max_{\pi \in \mathcal{P}_{\mathrm{simple}}} |\pi|, \qquad \pi^* = \arg\max_{\pi \in \mathcal{P}_{\mathrm{simple}}} \sum_{(u,v) \in \pi} D(u,v)$$

Exhaustive Depth-First Search explodes combinatorially on dense small-world word graphs ($|V| \approx 470$, $|E| \approx 15{,}000$). We solve both fixed-endpoint and multi-start variants in sub-minute runtime by seeding high-degree hub trajectories via fast greedy DFS and refining them through **Simulated Annealing (SA)** equipped with **2-opt**, **3-opt**, **common-neighbor splice (`insert`)**, **contraction (`remove`)**, and **bridge replacement (`replace`)** operators.

### 2. Contiguous Literal Quote Extraction (Prefix-Pruned Exact Search)
Extracting the longest or highest-cost cycle-free path in $G$ that simultaneously occurs *verbatim* as a contiguous substring in $\mathcal{D}$ is solved **exactly** by precomputing an $O(1)$ hash-trie prefix lookup over all valid corpus spans originating at $w_{\mathrm{start}}$, pruning $99.9\%$ of non-literal graph branches during iterative DFS.

### 3. Constrained Statistical Text Generation (Length-Normalised $A^*$ Search)
Standard additive $A^*$ search ($f(n) = G(n) + H(n)$) is fundamentally ill-suited for open-horizon text generation because every appended token adds a strictly positive negative log-likelihood $-\ln P(w_t \mid w_{<t}) > 0$, biasing the priority queue toward degenerate 1-token bridges. We formulate a **Length-Normalised $A^*$ Search** objective:

$$f_{\mathrm{avg}}(n) = (1 - w) \cdot g_{\mathrm{avg}}(n) + w \cdot h_{\mathrm{avg}}(n)$$

that normalizes cumulative N-gram backoff surprisal by the generated horizon $T$, paired with a **hybrid bigram + Reverse-BFS hop-distance heuristic** on the exact same per-step cost scale.

```mermaid
flowchart LR
    subgraph Corpus ["1. Corpus & Language Model"]
        A["Raw Literary Corpus\n(Orwell's 1984)"] --> B["Tokenizer & Rare-Word Filter\n(<RARE> threshold = 0.01)"]
        A --> C["Contiguous N-Gram Tables\nUnigram / Bigram / Trigram"]
    end

    subgraph Network ["2. Transition Graph Topology"]
        B --> D["Undirected Graph G = (V, E)\n& Symmetrized Counts C_sym"]
        D --> E["Inverted Distance Matrix\nD(u,v) = (max C_sym + 1) - C_sym(u,v)"]
    end

    subgraph Pathfinding ["3. Combinatorial Pathfinding (T6.1–T6.4)"]
        D --> F["Multi-Start DFS + Fixed-Endpoint SA\n2-opt / 3-opt / Common-Neighbor Splice\n(Longest Path: 463–466 Nodes)"]
        E --> G["Inverted-Cost SA Optimization\n(Max-Cost Simple Path: 1,132,655)"]
        D --> H["Prefix-Pruned Exact DFS\n(Optimal Quote: 27 Words / Cost 64,229)"]
    end

    subgraph Generation ["4. Length-Normalised A* NLP (T6.5A–T6.5C)"]
        C --> I["Dynamic N-Gram Backoff Cost\ng_avg(n) = G(n) / T"]
        D --> J["Hybrid Lookahead Heuristic h_avg(n)\nDirect Bigram + Reverse-BFS Hop Map"]
        I --> K["Length-Normalised A* Engine\nf_avg(n) = (1-w) g_avg(n) + w h_avg(n)"]
        J --> K
    end
```

---

## 2. Transition Network Formulation & Inverted Cost Matrix

Given an input text corpus, the pipeline in [`src/graph_search_nlp/network.py`](src/graph_search_nlp/network.py) executes five deterministic transformations:

1. **Lexical Tokenization (`tokenize_text`)**: Lowercases the character stream, groups contiguous alphabetic characters and word-internal apostrophes (`'`) into lexical tokens, and isolates sentence punctuation (`,` and `.`) as standalone structural tokens.
2. **Vocabulary Pruning (`replace_rare_tokens`)**: Computes relative word frequencies over non-punctuation tokens $W_{\text{words}} = \{w \in \mathcal{D} : w \notin \{',', '.'\}\}$ and maps infrequent words below threshold $\tau_{\text{rare}} = 0.01$ to the sentinel `<RARE>`:
   $$\text{RareSet} = \{ w \in V_{\text{words}} \mid \text{freq}(w) < \tau_{\text{rare}} \}, \quad \text{freq}(w) = \frac{\text{count}(w)}{|W_{\text{words}}|}$$
3. **Directional Bigram Adjacency (`get_text_adjacencies`)**: Extracts consecutive non-identical token transitions $C(u, v) = \sum_{i=1}^{M-1} \mathbb{I}[w_i = u \land w_{i+1} = v \land u \neq v]$.
4. **Topological Graph Construction (`build_unweighted_graph`)**: Forms the undirected simple graph $G = (V, E)$ where $\{u, v\} \in E \iff C(u,v) + C(v,u) > 0$.
5. **Symmetrized Inverted Distance Matrix (`compute_distance_matrix`)**: Symmetrizes directional counts via $C_{\text{sym}}(u, v) = \frac{1}{2}\big(C(u, v) + C(v, u)\big)$ with $C_{\text{sym}}(u, u) = 0$, and transforms co-occurrence frequency into an **inverted metric cost** (`distance_mode="inverted"`):
   $$D(u, v) = \begin{cases} 0 & \text{if } u = v \\ \left(\max_{x,y \in V} C_{\text{sym}}(x, y) + 1\right) - C_{\text{sym}}(u, v) & \text{if } u \neq v \end{cases}$$
   Under this formulation, frequently co-occurring word pairs have **low edge cost** (strong linguistic affinity), whereas rare transitions incur **high edge cost** approaching $\max C_{\text{sym}} + 1$.

---

## 3. Combinatorial Graph Pathfinding & Prefix-Pruned Quote Search (Tasks 6.1–6.4)

All paths evaluated in [`src/graph_search_nlp/pathfinding.py`](src/graph_search_nlp/pathfinding.py) satisfy the **simple path invariant**:
$$\pi = (v_0, v_1, \dots, v_k) \quad \text{such that} \quad \forall i \neq j,\; v_i \neq v_j \quad \text{and} \quad \forall i \in \{0, \dots, k-1\},\; (v_i, v_{i+1}) \in E$$

### 3.1 Longest Simple Path via Multi-Start & Fixed-Endpoint Simulated Annealing (`Task 6.1`)

To discover the longest simple path across the *Nineteen Eighty-Four* transition network:
1. **Hub Seeding by Degree Centrality**: We rank vertices by degree $d(v) = |N_G(v)|$ and initiate fast linear-time greedy DFS (`fast_approximate_dfs`) from the top-$k$ hubs, identifying the optimal endpoint pair `("kind", "ministry")`.
2. **Simulated Annealing with Composite $k$-Opt Neighborhoods (`simulated_annealing_fixed_endpoints`)**: Starting from an initial valid simple path $\pi^{(0)}$ connecting `start_word="kind"` and `end_word="ministry"`, each iteration samples a local topological operator while strictly locking $\pi[0]$ and $\pi[-1]$:
   - **Common-Neighbor Insertion (`insert`)**: Picks an edge $(u, v) = (\pi_i, \pi_{i+1})$ and splices an unvisited common neighbor $w \in (N_G(u) \cap N_G(v)) \setminus \pi$, increasing path length by $+1$.
   - **Shortcut Contraction (`remove`)**: Removes internal vertex $\pi_i$ if $(\pi_{i-1}, \pi_{i+1}) \in E$, escaping local topological dead-ends.
   - **Bridge Replacement (`replace`)**: Replaces $\pi_i$ with an alternative unvisited common neighbor $w \in (N_G(\pi_{i-1}) \cap N_G(\pi_{i+1})) \setminus \pi$ to unlock new downstream insertions.
   - **2-Opt Subpath Reversal (`2-opt`)**: Reverses internal segment $\pi_{i..j}$ whenever boundary reconnection edges $(\pi_{i-1}, \pi_j) \in E$ and $(\pi_i, \pi_{j+1}) \in E$ exist.
   - **3-Opt Segment Reconnection (`3-opt`)**: Partitions $\pi = A \oplus B \oplus C \oplus D$ at three internal cut points and reconnects $A \oplus C^{\text{rev}} \oplus B^{\text{rev}} \oplus D$ when valid boundary edges exist.
3. **Metropolis Acceptance Schedule**: Candidate paths $\pi'$ with $\Delta L = |\pi'| - |\pi| > 0$ are accepted deterministically; length-reducing moves that untangle topological bottlenecks are accepted with Boltzmann probability $\exp(\Delta L / T_t)$ under geometric cooling $T_{t+1} = \alpha T_t$ ($T_0 = 200.0, \alpha = 0.999$).

![Task 6.1 Longest Simple Path Output](docs/assets/t6_1_longest_path_output.png)

### 3.2 Longest Contiguous Literal Quote via $O(1)$ Prefix-Pruned DFS (`Task 6.2`)

A valid **literal quote** is a simple path $\pi = (w_{s}, w_{s+1}, \dots, w_{s+k})$ in $G$ that appears *verbatim* as a contiguous slice of `original_tokens` without containing `<RARE>` or repeating any word. Rather than running blind exponential DFS over $G$ and validating full paths post-hoc:
1. `find_valid_quotes_from_start` scans `original_tokens` from every occurrence of `start_word="perhaps"`, collecting all maximal contiguous, cycle-free, non-`<RARE>` spans.
2. `build_quote_prefix_set` indexes every prefix $(w_s, \dots, w_{s+j})$ of those valid spans into a hash set $\mathcal{T}_{\text{prefix}}$.
3. During iterative DFS in `print_long_quote`, a neighbor $v \in N_G(u)$ is pushed onto the search stack **if and only if** $\text{tuple}(\pi \oplus [v]) \in \mathcal{T}_{\text{prefix}}$. This reduces search complexity from $O(b^d)$ to $O(|\mathcal{T}_{\text{prefix}}|)$ and recovers the global optimum **27-word quote** between `"perhaps"` and `"it"`:
   > *"perhaps that was less true when you were actually in their hands . one did not know what happened inside the ministry of love , but it"*

![Task 6.2 Longest Literal Quote Output](docs/assets/t6_2_longest_quote_output.png)

### 3.3 Most Expensive Simple Path & Most Expensive Literal Quote (`Tasks 6.3 & 6.4`)

- **Task 6.3 (`print_expensive_path`)**: Replaces the cardinality objective $|\pi|$ in Fixed-Endpoint Simulated Annealing with the cumulative inverted-distance objective:
  $$\mathcal{J}_{\text{cost}}(\pi) = \sum_{i=0}^{|\pi|-2} D(\pi_i, \pi_{i+1})$$
  Using endpoints `("wanted", "same")` with $T_0 = 300.0, \alpha = 0.995$, `simulated_annealing_cost_fixed_endpoints` simultaneously maximizes path length and selects low-frequency (high-$D$) transitions, reaching a total path cost of **1,132,655–1,142,656** across 460 nodes (**85.12th percentile** of all submissions).
- **Task 6.4 (`print_expensive_quote`)**: Evaluates $\mathcal{J}_{\text{cost}}(\pi)$ across all prefix-pruned literal quotes between `("perhaps", "it")`, identifying the optimal literal quote cost of **64,229.0**.

![Task 6.3 Most Expensive Simple Path Output](docs/assets/t6_3_expensive_path_output.png)

![Task 6.4 Most Expensive Literal Quote Output](docs/assets/t6_4_expensive_quote_output.png)

---

## 4. Constrained Text Generation via Length-Normalised $A^*$ Search (Tasks 6.5A, 6.5B, 6.5C)

> *Note: Section 6.5C received a **10 / 10** score with the evaluator remark: "excellent, detailed explanation of length-normalised A\* with n-gram backoff costs and BFS-based heuristic that closely matches the implemented code."*

### 4.1 The Length-Bias Pathology of Standard Additive $A^*$

In Tasks 6.5A (`complete_sentence`) and 6.5B (`start_sentence`), we are given a template containing `<CONTENT>` bounded by a source anchor token $w_{\text{start}}$ (the last token before `<CONTENT>`) and a target anchor token $w_{\text{end}}$ (the first token after `<CONTENT>`):
- **Task 6.5A Example**: `"please believe my eyes <CONTENT>."` $\implies w_{\text{start}} = \text{"eyes"},\; w_{\text{end}} = \text{"."}$
- **Task 6.5B Example**: `"two <CONTENT> can ask for a solution."` $\implies w_{\text{start}} = \text{"two"},\; w_{\text{end}} = \text{"can"}$

In standard $A^*$ search, the path cost $G(n) = \sum_{t=1}^{T} c(w_{t-1}, w_t)$ is the sum of strictly positive step costs (negative log-probabilities). Because every generated word increases $G(n)$, standard $A^*$ heavily penalizes multi-word phrases and collapses to the shortest possible 1-word transition, even when a 4-to-6 word clause is far more grammatically natural and contextually coherent.

### 4.2 Length-Normalised Evaluation Function $f_{\text{avg}}(n)$

To eliminate horizon bias while preserving $A^*$ best-first ordering, [`src/graph_search_nlp/astar_generator.py`](src/graph_search_nlp/astar_generator.py) normalizes cumulative path cost by the number of generated transitions $T$:

$$g_{\text{avg}}(n) = \frac{G(n)}{T} = \frac{1}{T} \sum_{t=1}^{T} c_{\text{step}}(w_{t-1}, w_t \mid w_{t-2})$$

and orders the priority queue by the convex combination of the average step cost so far and the estimated per-step heuristic cost to reach $w_{\text{end}}$:

$$f_{\text{avg}}(n) = (1 - w) \cdot g_{\text{avg}}(n) + w \cdot h_{\text{avg}}(n), \qquad w = 0.3$$

### 4.3 Dynamic N-Gram Backoff Step Cost $c_{\mathrm{step}}$

At each expansion step from $w_{t-1}$ to candidate neighbor $w_t \in N_G(w_{t-1})$, `calculate_ngram_cost` computes $c_{\mathrm{step}}$ via a 3-tier statistical backoff model:

- **Tier 1 — Base Bigram Surprisal**:

$$c_2(w_{t-1}, w_t) = -\ln\!\left(\frac{C(w_{t-1}, w_t)}{C(w_{t-1})} + 10^{-10}\right)$$

- **Tier 2 — Trigram Context Augmentation ($N = 3$)**:
  If $C(w_{t-1}, w_t) > 0$, we inspect the 2-token history $(w_{t-2}, w_{t-1})$:

$$c_3(w_{t-2}, w_{t-1}, w_t) = \begin{cases} -\ln\!\left(\dfrac{C(w_{t-2}, w_{t-1}, w_t)}{C(w_{t-2}, w_{t-1})} + 10^{-10}\right) & \text{if } C(w_{t-2}, w_{t-1}, w_t) > 0 \\[8pt] \lambda_{\mathrm{backoff}} = 2.0 & \text{otherwise (unseen trigram penalty)} \end{cases}$$

  yielding $c_{\mathrm{step}} = c_2(w_{t-1}, w_t) + c_3(w_{t-2}, w_{t-1}, w_t)$. High-probability trigram continuations add near-zero incremental cost ($\approx 0.0$–$0.5$), whereas backing off to a bare bigram incurs $+2.0$.

- **Tier 3 — Non-Literal Graph Edge Fallback**:
  If $C(w_{t-1}, w_t) = 0$ (no directional bigram in the corpus) but $\{w_{t-1}, w_t\} \in E$ in the undirected transition graph, the transition is permitted as a "creative" topological bridge with a high fixed penalty:

$$c_{\mathrm{step}}(w_{t-1}, w_t) = \lambda_{\mathrm{graph}} = 4.0$$

### 4.4 Hybrid Lookahead Heuristic $h_{\mathrm{avg}}(n)$ (Direct Bigram + Reverse-BFS)

Because $g_{\mathrm{avg}}(n)$ is measured in **cost per transition step**, the heuristic $h_{\mathrm{avg}}(n)$ must estimate the per-step cost from current token $n$ to target $w_{\mathrm{end}}$ on the exact same scale:

- **Priority 1 — Direct Textual Bigram Lookahead**:
  If the corpus contains the direct bigram $(n, w_{\mathrm{end}})$, i.e., $C(n, w_{\mathrm{end}}) > 0$:

$$h_{\mathrm{avg}}(n) = -\ln\!\left(\frac{C(n, w_{\mathrm{end}})}{C(n)} + 10^{-10}\right)$$

- **Priority 2 — Precomputed Reverse-BFS Hop-Distance Schedule**:
  Before search begins, `build_reverse_bfs_heuristic` executes a Reverse Breadth-First Search from $w_{\mathrm{end}}$ across $G$, computing the exact shortest hop distance $d_{\mathrm{BFS}}(n, w_{\mathrm{end}})$ for every node $n \in V$. When no direct bigram $(n, w_{\mathrm{end}})$ exists, $h_{\mathrm{avg}}(n)$ maps $d_{\mathrm{BFS}}(n, w_{\mathrm{end}})$ to a step-cost penalty:

$$h_{\mathrm{avg}}(n) = \begin{cases} 0.0 & \text{if } d_{\mathrm{BFS}}(n, w_{\mathrm{end}}) = 0 \\ 3.0 & \text{if } d_{\mathrm{BFS}}(n, w_{\mathrm{end}}) = 1 \\ 4.0 & \text{if } d_{\mathrm{BFS}}(n, w_{\mathrm{end}}) = 2 \\ +\infty \text{ (pruned)} & \text{if } d_{\mathrm{BFS}}(n, w_{\mathrm{end}}) > d_{\max} \end{cases}$$

### 4.5 Full-Sentence Anti-Repetition & Dominance Pruning

- **Dominance Pruning (`min_avg_cost_to_node`)**: A candidate partial path reaching node $v$ with average cost $g_{\mathrm{avg}}$ is pruned if an equal or cheaper average path to $v$ has already been expanded: $g_{\mathrm{avg}}(v) \ge g^*_{\min}(v)$ (where $g^*_{\min}(v)$ tracks `min_avg_cost[v]`).
- **Minimum Content Horizon (`min_path_len_for_end = 2`)**: Reaching $w_{\mathrm{end}}$ at $T < 2$ is rejected so `<CONTENT>` is always expanded into at least one substantive token.
- **Full-Sentence Disjointness (`avoid_suffix_repeats=True`)**: In addition to preventing cycles along the active prefix and generated path ($w_t \notin \mathrm{path}$), `complete_sentence` and `start_sentence` pass the trailing prompt suffix tokens (`forbidden_tokens = set(suffix[1:])`) into `run_length_normalised_astar`, guaranteeing zero repeated tokens across the entire reconstructed sentence.

---

## 5. Quantitative Benchmarks & Empirical Results

![Cohort Benchmark Percentiles and Length-Normalised A* Scaling](docs/assets/benchmark_and_sa_analysis.png)

| Task ID | Task Description | Algorithmic Method | Achieved Metric / Output | Cohort Evaluation Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **T6.1** | **Longest Simple Path** (`print_long_path`) | Multi-Start Hub Seeding + Fixed-Endpoint Simulated Annealing (`2-opt`, `3-opt`, `insert`, `replace`) | **463–466 nodes** (`"kind"` $\to \dots \to$ `"ministry"`) | **74.79th Percentile** (Score: 4.0 / 5.0) |
| **T6.2** | **Longest Literal Quote** (`print_long_quote`) | Precomputed Valid-Span Extraction + $O(1)$ Prefix-Set Pruned DFS | **27 tokens** (*"perhaps that was less true ... but it"*) | **Global Optimum Obtainable** (Score: 5.0 / 5.0) |
| **T6.3** | **Most Expensive Simple Path** (`print_expensive_path`) | Inverted-Distance Cost SA with Fixed Endpoints (`"wanted"` $\to$ `"same"`) | **Cost = 1,132,655 – 1,142,656** (460 nodes) | **85.12th Percentile** (Score: 5.0 / 5.0) |
| **T6.4** | **Most Expensive Literal Quote** (`print_expensive_quote`) | Prefix-Set Pruned Exact DFS over Inverted Distance Matrix | **Cost = 64,229.0** (27 tokens) | **Global Optimum Obtainable** (Score: 5.0 / 5.0) |
| **T6.5A** | **Constrained Sentence Completion** (`complete_sentence`) | Length-Normalised $A^*$ ($g_{\text{avg}} = G/T$, Trigram Backoff + Reverse-BFS Heuristic) | Coherent multi-word completions | **85.99th Pct Perplexity** · **77.07th Pct Variance** · **68.79th Pct Grammar** (8.28 / 10) |
| **T6.5B** | **Prefix-Conditioned Sentence Generation** (`start_sentence`) | Length-Normalised $A^*$ with Suffix Anti-Repetition Filter | Coherent clause bridges | **98.99th Pct Grammar** · **76.35th Pct Variance** · **75.00th Pct Perplexity** (8.34 / 10) |
| **T6.5C** | **Mathematical & Algorithmic Specification** | Formal derivation of Length-Normalised $A^*$, N-Gram Backoff & Hybrid Reverse-BFS Heuristic | Exact code-to-math alignment | **10.0 / 10.0 (100%)** |

---

## 6. Repository Structure

```text
ngram-astar-text-generation-and-pathfinding/
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI across Python 3.10, 3.11, 3.12
├── docs/
│   └── assets/
│       ├── benchmark_and_sa_analysis.png    # Cohort percentile benchmarks & g_avg(n) scaling plot
│       ├── t6_1_longest_path_output.png     # Task 6.1 466-node longest simple path trace
│       ├── t6_2_longest_quote_output.png    # Task 6.2 optimal 27-token literal quote output
│       ├── t6_3_expensive_path_output.png   # Task 6.3 1.14M-cost simple path output
│       └── t6_4_expensive_quote_output.png  # Task 6.4 optimal 64,229-cost literal quote output
├── scripts/
│   ├── run_pathfinding.py               # CLI for BFS, UCS, SA Longest/Expensive Paths & Quotes
│   └── generate_sentences.py            # CLI for Length-Normalised A* Sentence Generation
├── src/
│   └── graph_search_nlp/
│       ├── __init__.py                  # Public package exports
│       ├── network.py                   # Corpus tokenization, graph construction, N-gram backoff
│       ├── pathfinding.py               # BFS, DFS, UCS, Greedy, A*, Simulated Annealing & Quotes
│       └── astar_generator.py           # Length-Normalised A* with Reverse-BFS heuristic
├── tests/
│   └── test_search_algorithms.py        # 14 unit tests covering all algorithms & invariants
├── .gitignore
├── LICENSE                              # MIT License
├── pyproject.toml                       # PEP 621 package configuration
├── requirements.txt                     # Runtime & test dependencies
└── README.md
```

---

## 7. Quickstart & CLI Usage

### Installation

```bash
git clone https://github.com/Ceviche98/ngram-astar-text-generation-and-pathfinding.git
cd ngram-astar-text-generation-and-pathfinding
pip install -r requirements.txt
pip install -e .
```

### 1. Run Combinatorial Pathfinding & Literal Quote Extraction (Tasks 6.1–6.4)

Run on the built-in Orwellian sample corpus or point `--corpus` to any full text file (e.g., `1984.txt`):

```bash
python scripts/run_pathfinding.py --sa-iter 3000
```

### 2. Run Length-Normalised $A^*$ Sentence Generation (Tasks 6.5A & 6.5B)

```bash
python scripts/generate_sentences.py \
    --completion-prompt "please believe my eyes <CONTENT>." \
    --start-prompt "two <CONTENT> can ask for a solution." \
    --heuristic-weight 0.3 \
    --max-n 3
```

### 3. Python API Example

```python
from graph_search_nlp import (
    process_text_network,
    print_long_path,
    print_long_quote,
    print_expensive_path,
    print_expensive_quote,
    complete_sentence,
    start_sentence,
)

# 1. Build transition graph and inverted co-occurrence distance matrix
net = process_text_network("path/to/corpus.txt", rare_threshold=0.01, distance_mode="inverted")

# 2. Extract extremal simple paths and literal quotes (Tasks 6.1 - 6.4)
longest_path = print_long_path(net, start_word="kind", end_word="ministry")
longest_quote = print_long_quote(net, start_word="perhaps", end_word="it")
expensive_path, max_path_cost = print_expensive_path(net, start_word="wanted", end_word="same")
expensive_quote, max_quote_cost = print_expensive_quote(net, start_word="perhaps", end_word="it")

# 3. Generate constrained completions via Length-Normalised A* (Tasks 6.5A & 6.5B)
completed = complete_sentence(net, prompt="please believe my eyes <CONTENT>.")
started = start_sentence(net, prompt="two <CONTENT> can ask for a solution.")
print("Completed:", " ".join(completed))
print("Started  :", " ".join(started))
```

### 4. Run Test Suite

```bash
pytest tests/ -v
```

---

## License

Released under the [MIT License](LICENSE). Authored by **Oscar Moreno**.
