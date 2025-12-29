"""
Custom start-target vertex analysis.

Analyzes DFS discovery statistics for a specific target vertex
starting from a specific start vertex (user-specified pair).
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple, TypeVar

from dfs_analyzer.core.graphs import Graph, Hypercube

Vertex = TypeVar('Vertex')


def collect_custom_vertex_statistics(
    graph: Graph[Vertex],
    start_vertex: Vertex,
    target_vertex: Vertex,
    num_samples: int,
    *,
    rng=None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Collects DFS statistics for a specific start-target vertex pair.

    Runs RDFS multiple times from the specified start vertex and tracks
    discovery numbers specifically for the specified target vertex.

    Args:
        graph: Graph instance.
        start_vertex: Starting vertex for DFS.
        target_vertex: Target vertex to analyze.
        num_samples: Number of RDFS runs to perform.
        rng: Random number generator (uses default if None).
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        Dictionary containing target vertex statistics.

    Example:
        >>> cube = Hypercube(4)
        >>> stats = collect_custom_vertex_statistics(
        ...     cube, (0,0,0,0), (1,1,0,0), 1000
        ... )
        >>> print(stats['target_mean'])
    """
    from dfs_analyzer.core.rdfs import rdfs

    if rng is None:
        rng = np.random.default_rng(1832479182)

    # Stores discovery numbers for target vertex
    target_discoveries = []

    # Also collects all vertices for comparison
    all_dist_stats = defaultdict(list)

    # Runs RDFS multiple times
    for i in range(num_samples):
        # Runs single RDFS from start vertex
        dist_stats = defaultdict(list)
        rdfs(graph, start_vertex, dist_stats=dist_stats, rng=rng)

        # Extracts discovery number for target vertex
        if target_vertex in dist_stats:
            discovery_num = dist_stats[target_vertex][0]
            target_discoveries.append(discovery_num)

            # Also stores for all vertices
            for vertex, discoveries in dist_stats.items():
                all_dist_stats[vertex].append(discoveries[0])

        # Reports progress
        if progress_callback is not None:
            progress_callback(i + 1, num_samples)

    # Computes statistics for target vertex
    target_mean = float(np.mean(target_discoveries))
    target_std = float(np.std(target_discoveries))
    target_min = int(np.min(target_discoveries))
    target_max = int(np.max(target_discoveries))
    target_median = float(np.median(target_discoveries))

    # Computes overall statistics for comparison
    all_means = []
    for vertex_discoveries in all_dist_stats.values():
        all_means.append(np.mean(vertex_discoveries))
    overall_mean = float(np.mean(all_means))

    # Gets graph properties
    num_vertices = graph.number_vertices()
    theoretical_all_vertices = (num_vertices - 1) / 2

    # Computes distance if available (for hypercubes)
    distance = None
    if isinstance(graph, Hypercube):
        # Computes Hamming distance
        distance = sum(s != t for s, t in zip(start_vertex, target_vertex))

    return {
        "start_vertex": start_vertex,
        "target_vertex": target_vertex,
        "distance": distance,
        "num_samples": num_samples,
        "num_vertices": num_vertices,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_min": target_min,
        "target_max": target_max,
        "target_median": target_median,
        "target_discoveries": target_discoveries,
        "overall_mean": overall_mean,
        "theoretical_all_vertices": theoretical_all_vertices,
    }


def compute_custom_vertex_hitting_time_laplacian(
    graph: Graph[Vertex],
    start_vertex: Vertex,
    target_vertex: Vertex,
) -> Dict[str, Any]:
    """
    Computes hitting time for custom vertex pair using Laplacian.

    Args:
        graph: Graph instance.
        start_vertex: Starting vertex.
        target_vertex: Target vertex.

    Returns:
        Dictionary with target vertex hitting time statistics.
    """
    from dfs_analyzer.core.random_walk import compute_laplacian_hitting_times

    # Computes hitting times using Laplacian from start vertex
    all_hitting_times = compute_laplacian_hitting_times(graph, start_vertex)

    # Extracts hitting time for target vertex
    target_hitting_time = all_hitting_times[target_vertex]

    # Computes overall statistics for comparison
    all_values = list(all_hitting_times.values())
    overall_mean = float(np.mean(all_values))

    # Gets graph properties
    num_vertices = graph.number_vertices()

    # Computes distance if available (for hypercubes)
    distance = None
    if isinstance(graph, Hypercube):
        # Computes Hamming distance
        distance = sum(s != t for s, t in zip(start_vertex, target_vertex))

    return {
        "start_vertex": start_vertex,
        "target_vertex": target_vertex,
        "distance": distance,
        "num_vertices": num_vertices,
        "target_hitting_time": target_hitting_time,
        "overall_mean": overall_mean,
        "theoretical_average": (num_vertices - 1) / 2,
    }


def get_custom_vertex_summary(stats: Dict[str, Any]) -> str:
    """
    Creates human-readable summary of custom vertex analysis.

    Args:
        stats: Dictionary returned by collect_custom_vertex_statistics().

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CUSTOM VERTEX PAIR ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Total vertices: {stats['num_vertices']}")
    lines.append(f"Starting vertex: {stats['start_vertex']}")
    lines.append(f"Target vertex: {stats['target_vertex']}")

    if stats['distance'] is not None:
        lines.append(f"Hamming distance: {stats['distance']}")

    lines.append(f"Samples collected: {stats['num_samples']}")
    lines.append("")

    lines.append("--- Discovery Number Statistics ---")
    lines.append(f"Theoretical average (all vertices): {stats['theoretical_all_vertices']:.4f}")
    lines.append(f"Observed average (all vertices): {stats['overall_mean']:.4f}")
    lines.append("")
    lines.append(f"Target vertex mean: {stats['target_mean']:.4f}")
    lines.append(f"Target vertex std: {stats['target_std']:.4f}")
    lines.append(f"Target vertex min: {stats['target_min']}")
    lines.append(f"Target vertex max: {stats['target_max']}")
    lines.append(f"Target vertex median: {stats['target_median']:.4f}")
    lines.append("")

    # Comparison ratio
    ratio = stats['target_mean'] / stats['theoretical_all_vertices']
    lines.append("--- Key Insights ---")
    lines.append(f"Discovery ratio (target/average): {ratio:.4f}")
    lines.append(f"  (ratio < 1.0 means discovered earlier than average)")
    lines.append(f"  (ratio = 1.0 means discovered at typical time)")
    lines.append(f"  (ratio > 1.0 means discovered later than average)")
    lines.append("")

    if ratio < 0.9:
        lines.append("[OK] Target vertex discovered EARLIER than average")
    elif ratio < 1.1:
        lines.append("≈ Target vertex discovered at TYPICAL time")
    else:
        lines.append("[WARNING] Target vertex discovered LATER than average")

    lines.append("")

    # Additional interpretation based on distance
    if stats['distance'] is not None:
        lines.append("--- Interpretation ---")
        lines.append(f"The target vertex is at Hamming distance {stats['distance']} from start.")
        if stats['distance'] == 0:
            lines.append("This is the starting vertex itself (should be discovered first).")
        elif stats['distance'] == 1:
            lines.append("This is an immediate neighbor of the starting vertex.")
        elif stats['distance'] > stats['num_vertices'] // 2:
            lines.append("This is a relatively distant vertex.")
        else:
            lines.append("This is at moderate distance from the start.")

    lines.append("=" * 70)

    return "\n".join(lines)


def get_custom_vertex_laplacian_summary(stats: Dict[str, Any]) -> str:
    """
    Creates human-readable summary of custom vertex Laplacian analysis.

    Args:
        stats: Dictionary returned by compute_custom_vertex_hitting_time_laplacian().

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CUSTOM VERTEX PAIR ANALYSIS - LAPLACIAN")
    lines.append("=" * 70)
    lines.append(f"Total vertices: {stats['num_vertices']}")
    lines.append(f"Starting vertex: {stats['start_vertex']}")
    lines.append(f"Target vertex: {stats['target_vertex']}")

    if stats['distance'] is not None:
        lines.append(f"Hamming distance: {stats['distance']}")

    lines.append("")

    lines.append("--- Hitting Time Statistics ---")
    lines.append(f"Theoretical average (all vertices): {stats['theoretical_average']:.4f}")
    lines.append(f"Observed average (all vertices): {stats['overall_mean']:.4f}")
    lines.append("")
    lines.append(f"Target vertex hitting time: {stats['target_hitting_time']:.4f}")
    lines.append("")

    # Comparison ratio
    ratio = stats['target_hitting_time'] / stats['overall_mean']
    lines.append("--- Key Insights ---")
    lines.append(f"Hitting time ratio (target/average): {ratio:.4f}")
    lines.append(f"  (ratio < 1.0 means reached faster than average)")
    lines.append(f"  (ratio = 1.0 means reached at typical time)")
    lines.append(f"  (ratio > 1.0 means takes longer to reach)")
    lines.append("")

    if ratio < 0.9:
        lines.append("[OK] Target vertex reached FASTER than average")
    elif ratio < 1.1:
        lines.append("≈ Target vertex reached at TYPICAL time")
    else:
        lines.append("[WARNING] Target vertex takes LONGER to reach")

    lines.append("")

    # Additional interpretation based on distance
    if stats['distance'] is not None:
        lines.append("--- Theoretical Insight ---")
        lines.append(f"The Laplacian method provides exact expected hitting time")
        lines.append(f"for the target vertex at Hamming distance {stats['distance']}.")
        lines.append(f"This represents the theoretical value for random walks.")

    lines.append("=" * 70)

    return "\n".join(lines)
