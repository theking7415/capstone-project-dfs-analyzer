"""
Erdős-Rényi random graph G(n,p) implementation.

Generates random graphs where each edge appears independently with probability p.
Includes connectivity checking to ensure graph is a single connected component.
"""

import numpy as np
from typing import TypeVar, Generic, Set, Tuple
from collections import defaultdict, deque

from dfs_analyzer.core.graphs import Graph


class ErdosRenyiGraph(Graph[int]):
    """
    Erdős-Rényi random graph G(n,p).

    Each possible edge between vertices appears independently with probability p.
    Vertices are labeled 0, 1, 2, ..., n-1.

    Note: This class generates the graph structure but does not guarantee connectivity.
    Use is_connected() to check connectivity after generation.

    Attributes:
        n: Number of vertices.
        p: Edge probability (0 < p < 1).
        edges: Set of edges as (u, v) tuples where u < v.
        adj_list: Adjacency list representation.
    """

    def __init__(self, n: int, p: float, rng_seed: int = None):
        """
        Initializes and generates G(n,p) random graph.

        Args:
            n: Number of vertices (must be positive).
            p: Edge probability (must be in (0, 1)).
            rng_seed: Random seed for reproducibility.

        Raises:
            ValueError: If n < 1 or p not in (0, 1).
        """
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        if not (0 < p < 1):
            raise ValueError(f"p must be in (0, 1), got {p}")

        self.n = n
        self.p = p
        self.rng = np.random.default_rng(rng_seed)

        # Generates edges
        self.edges: Set[Tuple[int, int]] = set()
        self._generate_edges()

        # Builds adjacency list
        self.adj_list_data = self._build_adjacency_list()

    def _generate_edges(self):
        """Generates edges according to G(n,p) model."""
        # Iterates over all possible edges
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Adds edge with probability p
                if self.rng.random() < self.p:
                    self.edges.add((i, j))

    def _build_adjacency_list(self) -> dict:
        """Builds adjacency list from edge set."""
        adj = defaultdict(set)

        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)

        # Ensures all vertices present
        for i in range(self.n):
            if i not in adj:
                adj[i] = set()

        return adj

    def is_connected(self) -> bool:
        """
        Checks if graph is connected using BFS.

        Returns:
            True if graph is a single connected component, False otherwise.
        """
        if self.n == 0:
            return True
        if self.n == 1:
            return True

        # BFS from vertex 0
        visited = set()
        queue = deque([0])
        visited.add(0)

        while queue:
            u = queue.popleft()
            for v in self.adj_list_data[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # Connected if all vertices reached
        return len(visited) == self.n

    def get_start_vertex(self) -> int:
        """Returns starting vertex (always 0)."""
        return 0

    def get_adj_list(self, v: int) -> list[int]:
        """
        Returns list of neighbors for vertex v.

        Args:
            v: Vertex (0 to n-1).

        Returns:
            List of neighboring vertices.
        """
        return list(self.adj_list_data[v])

    def number_vertices(self) -> int:
        """Returns number of vertices."""
        return self.n

    def number_edges(self) -> int:
        """Returns number of edges."""
        return len(self.edges)

    def get_expected_edges(self) -> float:
        """
        Returns expected number of edges for G(n,p).

        Returns:
            Expected number of edges = p * C(n, 2).
        """
        return self.p * (self.n * (self.n - 1)) / 2

    def get_connectivity_threshold(self) -> float:
        """
        Returns probability threshold for high probability of connectivity.

        Uses the threshold p ≥ (ln n + c) / n where c ≈ 3 gives high probability.

        Returns:
            Recommended minimum p for connectivity.
        """
        if self.n <= 1:
            return 0.0
        return (np.log(self.n) + 3) / self.n

    def desc(self) -> str:
        """Returns short description for file naming."""
        return f"gnp_n{self.n}_p{self.p:.4f}"

    def plot_means_vars(self, summary_stats: dict, *, fname=None):
        """
        Plots mean and variance for each vertex (not implemented for G(n,p)).

        G(n,p) graphs are typically used in batch mode where per-vertex
        plotting is not meaningful due to non-regular structure.

        Args:
            summary_stats: Summary statistics dictionary.
            fname: Optional filename for saving plot.

        Note:
            This method is required by the Graph interface but not used
            in G(n,p) batch experiments.
        """
        print("Warning: plot_means_vars not implemented for G(n,p) graphs")
        print("  G(n,p) graphs are non-regular - use batch mode for aggregate statistics")

    def __str__(self) -> str:
        """Returns string representation."""
        expected = self.get_expected_edges()
        actual = self.number_edges()
        threshold = self.get_connectivity_threshold()
        connected = self.is_connected()

        return (
            f"G({self.n}, {self.p:.4f})\n"
            f"  Vertices: {self.n}\n"
            f"  Edges: {actual} (expected: {expected:.1f})\n"
            f"  Connected: {connected}\n"
            f"  Connectivity threshold: p ≥ {threshold:.4f}"
        )


def generate_connected_gnp(
    n: int,
    p: float,
    rng_seed: int = None,
    max_attempts: int = None
) -> ErdosRenyiGraph:
    """
    Generates connected G(n,p) graph using rejection sampling.

    Repeatedly generates G(n,p) graphs until a connected one is found.

    Args:
        n: Number of vertices.
        p: Edge probability.
        rng_seed: Random seed for reproducibility.
        max_attempts: Maximum number of attempts before giving up.
            If None, uses adaptive limit: 1000 for n<500, 100 for n>=500.

    Returns:
        Connected ErdosRenyiGraph instance.

    Raises:
        ValueError: If cannot generate connected graph within max_attempts.

    Example:
        >>> g = generate_connected_gnp(100, 0.05, rng_seed=42)
        >>> print(g.is_connected())
        True
    """
    # Determines max_attempts using adaptive strategy
    if max_attempts is None:
        if n < 500:
            max_attempts = 1000  # Fast generation for small/medium graphs
        else:
            max_attempts = 100   # Large graphs - encourage p >= threshold

    # Creates RNG for generating seeds
    master_rng = np.random.default_rng(rng_seed)

    # Warns if p is below connectivity threshold
    threshold = (np.log(n) + 3) / n if n > 1 else 0.0
    if n > 1 and p < threshold:
        print(f"Warning: p={p:.4f} is below connectivity threshold {threshold:.4f}")
        print(f"         Generation may require many attempts (max {max_attempts}).")

    # Attempts to generate connected graph
    for attempt in range(max_attempts):
        # Generates new graph with unique seed
        seed = master_rng.integers(0, 2**31)
        graph = ErdosRenyiGraph(n, p, rng_seed=seed)

        if graph.is_connected():
            if attempt > 0 and attempt % 10 == 0:
                print(f"  Generated connected graph after {attempt + 1} attempts")
            return graph

    # Failed to generate connected graph
    raise ValueError(
        f"Failed to generate connected G({n}, {p:.4f}) after {max_attempts} attempts.\n"
        f"Try increasing p above the connectivity threshold {threshold:.4f} or "
        f"increasing max_attempts."
    )
