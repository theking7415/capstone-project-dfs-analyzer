"""
Immediate neighbor analysis for DFS and random walks.

Focuses on discovery numbers and hitting times specifically for vertices
that are immediate neighbors of the starting vertex.
"""

import numpy as np
from collections import defaultdict
from typing import TypeVar, Dict, Any

from dfs_analyzer.core.graphs import Graph

Vertex = TypeVar("Vertex")


def collect_neighbor_statistics(
    graph: Graph[Vertex],
    num_samples: int,
    *,
    rng=None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Collects DFS statistics specifically for immediate neighbors.

    Runs RDFS multiple times and tracks discovery numbers only for vertices
    that are immediate neighbors of the starting vertex.

    Args:
        graph: Graph instance to analyze.
        num_samples: Number of RDFS runs to perform.
        rng: Random number generator (uses default if None).
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        Dictionary containing neighbor statistics and comparison data.
    """
    from dfs_analyzer.core.rdfs import rdfs

    if rng is None:
        rng = np.random.default_rng(1832479182)

    # Gets starting vertex and its immediate neighbors
    start_vertex = graph.get_start_vertex()
    immediate_neighbors = graph.get_adj_list(start_vertex)
    num_neighbors = len(immediate_neighbors)

    # Stores discovery numbers for each neighbor
    neighbor_dist_stats = defaultdict(list)

    # Runs RDFS multiple times
    for i in range(num_samples):
        # Runs single RDFS
        dist_stats = defaultdict(list)
        rdfs(graph, start_vertex, dist_stats=dist_stats, rng=rng)

        # Extracts discovery numbers for immediate neighbors only
        for neighbor in immediate_neighbors:
            if neighbor in dist_stats:
                neighbor_dist_stats[neighbor].append(dist_stats[neighbor][0])

        # Reports progress
        if progress_callback is not None:
            progress_callback(i + 1, num_samples)

    # Computes statistics for each neighbor
    neighbor_stats = {}
    for neighbor in immediate_neighbors:
        samples = neighbor_dist_stats[neighbor]
        neighbor_stats[neighbor] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "min": int(np.min(samples)),
            "max": int(np.max(samples)),
            "median": float(np.median(samples)),
            "samples": samples,
        }

    # Computes overall statistics for immediate neighbors
    all_neighbor_means = [stats["mean"] for stats in neighbor_stats.values()]
    overall_neighbor_mean = float(np.mean(all_neighbor_means))
    overall_neighbor_std = float(np.std(all_neighbor_means))

    # Gets graph properties
    num_vertices = graph.number_vertices()
    theoretical_all_vertices = (num_vertices - 1) / 2

    # Theoretical expectation for immediate neighbors
    # In symmetric graphs, immediate neighbors should be discovered early
    theoretical_neighbor = num_neighbors / np.pi  # Approximate for 2D grids
    # For hypercubes, use different formula based on degree
    if hasattr(graph, 'd'):  # Hypercube
        degree = graph.d
        theoretical_neighbor = (num_vertices - 1) / (2 * degree)

    return {
        "start_vertex": start_vertex,
        "immediate_neighbors": immediate_neighbors,
        "num_neighbors": num_neighbors,
        "num_samples": num_samples,
        "neighbor_stats": neighbor_stats,
        "overall_neighbor_mean": overall_neighbor_mean,
        "overall_neighbor_std": overall_neighbor_std,
        "theoretical_all_vertices": theoretical_all_vertices,
        "theoretical_neighbor": theoretical_neighbor,
        "num_vertices": num_vertices,
    }


def get_neighbor_summary(stats: Dict[str, Any]) -> str:
    """
    Creates human-readable summary of neighbor analysis.

    Args:
        stats: Dictionary returned by collect_neighbor_statistics().

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("IMMEDIATE NEIGHBOR ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Total vertices: {stats['num_vertices']}")
    lines.append(f"Starting vertex: {stats['start_vertex']}")
    lines.append(f"Number of immediate neighbors: {stats['num_neighbors']}")
    lines.append(f"Samples collected: {stats['num_samples']}")
    lines.append("")
    lines.append("--- Discovery Number Statistics ---")
    lines.append(f"Average discovery number (all vertices): {stats['theoretical_all_vertices']:.4f}")
    lines.append(f"Average discovery number (immediate neighbors): {stats['overall_neighbor_mean']:.4f}")
    lines.append(f"Standard deviation across neighbors: {stats['overall_neighbor_std']:.4f}")
    lines.append("")

    # Shows per-neighbor statistics
    lines.append("--- Per-Neighbor Statistics ---")
    lines.append(f"{'Neighbor':<20} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    lines.append("-" * 70)

    neighbor_stats = stats['neighbor_stats']
    for neighbor, nstats in sorted(neighbor_stats.items()):
        neighbor_str = str(neighbor) if not isinstance(neighbor, tuple) else "".join(map(str, neighbor))
        lines.append(
            f"{neighbor_str:<20} {nstats['mean']:<10.4f} {nstats['std']:<10.4f} "
            f"{nstats['min']:<8} {nstats['max']:<8}"
        )

    lines.append("")
    lines.append("--- Key Insights ---")

    # Comparison with overall average
    ratio = stats['overall_neighbor_mean'] / stats['theoretical_all_vertices']
    lines.append(f"Neighbor discovery ratio: {ratio:.4f}")
    lines.append(f"  (ratio < 1.0 means neighbors discovered earlier than average)")
    lines.append(f"  (ratio = 1.0 means neighbors typical)")
    lines.append(f"  (ratio > 1.0 means neighbors discovered later than average)")
    lines.append("")

    if ratio < 0.5:
        lines.append("[OK] Immediate neighbors discovered MUCH earlier than average")
    elif ratio < 0.9:
        lines.append("[OK] Immediate neighbors discovered earlier than average")
    elif ratio < 1.1:
        lines.append("≈ Immediate neighbors discovered at typical time")
    else:
        lines.append("[FAIL] Immediate neighbors discovered later than average (unusual)")

    lines.append("=" * 70)

    return "\n".join(lines)


def compute_neighbor_hitting_times_laplacian(
    graph: Graph[Vertex]
) -> Dict[str, Any]:
    """
    Computes hitting times for immediate neighbors using Laplacian.

    Args:
        graph: Graph instance to analyze.

    Returns:
        Dictionary with neighbor hitting time statistics.
    """
    from dfs_analyzer.core.random_walk import compute_laplacian_hitting_times

    # Gets starting vertex and its immediate neighbors
    start_vertex = graph.get_start_vertex()
    immediate_neighbors = graph.get_adj_list(start_vertex)

    # Computes hitting times using Laplacian
    all_hitting_times = compute_laplacian_hitting_times(graph, start_vertex)

    # Extracts hitting times for immediate neighbors
    neighbor_hitting_times = {}
    for neighbor in immediate_neighbors:
        neighbor_hitting_times[neighbor] = all_hitting_times[neighbor]

    # Computes statistics
    neighbor_values = list(neighbor_hitting_times.values())
    overall_neighbor_mean = float(np.mean(neighbor_values))
    overall_neighbor_std = float(np.std(neighbor_values))

    # Gets graph properties
    num_vertices = graph.number_vertices()
    all_values = list(all_hitting_times.values())
    overall_mean = float(np.mean(all_values))

    return {
        "start_vertex": start_vertex,
        "immediate_neighbors": immediate_neighbors,
        "num_neighbors": len(immediate_neighbors),
        "neighbor_hitting_times": neighbor_hitting_times,
        "overall_neighbor_mean": overall_neighbor_mean,
        "overall_neighbor_std": overall_neighbor_std,
        "overall_mean": overall_mean,
        "num_vertices": num_vertices,
    }


def get_neighbor_laplacian_summary(stats: Dict[str, Any]) -> str:
    """
    Creates human-readable summary of neighbor Laplacian analysis.

    Args:
        stats: Dictionary returned by compute_neighbor_hitting_times_laplacian().

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("IMMEDIATE NEIGHBOR ANALYSIS (LAPLACIAN)")
    lines.append("=" * 70)
    lines.append(f"Total vertices: {stats['num_vertices']}")
    lines.append(f"Starting vertex: {stats['start_vertex']}")
    lines.append(f"Number of immediate neighbors: {stats['num_neighbors']}")
    lines.append("")
    lines.append("--- Hitting Time Statistics ---")
    lines.append(f"Average hitting time (all vertices): {stats['overall_mean']:.4f}")
    lines.append(f"Average hitting time (immediate neighbors): {stats['overall_neighbor_mean']:.4f}")
    lines.append(f"Standard deviation across neighbors: {stats['overall_neighbor_std']:.4f}")
    lines.append("")

    # Shows per-neighbor statistics
    lines.append("--- Per-Neighbor Hitting Times ---")
    lines.append(f"{'Neighbor':<20} {'Hitting Time':<15}")
    lines.append("-" * 70)

    neighbor_times = stats['neighbor_hitting_times']
    for neighbor, time in sorted(neighbor_times.items()):
        neighbor_str = str(neighbor) if not isinstance(neighbor, tuple) else "".join(map(str, neighbor))
        lines.append(f"{neighbor_str:<20} {time:<15.4f}")

    lines.append("")
    lines.append("--- Key Insights ---")

    # Comparison with overall average
    ratio = stats['overall_neighbor_mean'] / stats['overall_mean']
    lines.append(f"Neighbor hitting time ratio: {ratio:.4f}")
    lines.append(f"  (ratio < 1.0 means neighbors reached faster than average)")
    lines.append(f"  (ratio = 1.0 means neighbors typical)")
    lines.append(f"  (ratio > 1.0 means neighbors take longer to reach)")
    lines.append("")

    if ratio < 0.5:
        lines.append("[OK] Immediate neighbors reached MUCH faster than average")
    elif ratio < 0.9:
        lines.append("[OK] Immediate neighbors reached faster than average")
    elif ratio < 1.1:
        lines.append("≈ Immediate neighbors reached at typical time")
    else:
        lines.append("[FAIL] Immediate neighbors take longer to reach (unusual)")

    lines.append("=" * 70)

    return "\n".join(lines)
