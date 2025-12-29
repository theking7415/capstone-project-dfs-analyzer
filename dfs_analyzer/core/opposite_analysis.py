"""
Opposite vertex analysis for hypercubes.

Analyzes the discovery number and hitting time for the vertex that is
maximally distant (diagonally opposite) from the starting vertex.

For hypercubes: if start is (0,0,0,...), opposite is (1,1,1,...)
This is the vertex at maximum Hamming distance (all bits flipped).
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple

from dfs_analyzer.core.graphs import Hypercube


def get_opposite_vertex(hypercube: Hypercube) -> Tuple[int, ...]:
    """
    Gets the diagonally opposite vertex in a hypercube.

    For a hypercube starting at (0,0,0,...), the opposite vertex
    is (1,1,1,...) - all bits flipped.

    Args:
        hypercube: Hypercube graph instance.

    Returns:
        Tuple representing the opposite vertex.
    """
    start_vertex = hypercube.get_start_vertex()
    # Flips all bits (0 becomes 1, 1 becomes 0)
    opposite = tuple(1 - bit for bit in start_vertex)
    return opposite


def collect_opposite_statistics(
    hypercube: Hypercube,
    num_samples: int,
    *,
    rng=None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Collects DFS statistics for the opposite vertex.

    Runs RDFS multiple times and tracks discovery numbers specifically
    for the vertex that is diagonally opposite to the starting vertex.

    Args:
        hypercube: Hypercube graph instance.
        num_samples: Number of RDFS runs to perform.
        rng: Random number generator (uses default if None).
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        Dictionary containing opposite vertex statistics.
    """
    from dfs_analyzer.core.rdfs import rdfs

    if rng is None:
        rng = np.random.default_rng(1832479182)

    # Gets starting vertex and opposite vertex
    start_vertex = hypercube.get_start_vertex()
    opposite_vertex = get_opposite_vertex(hypercube)

    # Stores discovery numbers for opposite vertex
    opposite_discoveries = []

    # Also collects all vertices for comparison
    all_dist_stats = defaultdict(list)

    # Runs RDFS multiple times
    for i in range(num_samples):
        # Runs single RDFS
        dist_stats = defaultdict(list)
        rdfs(hypercube, start_vertex, dist_stats=dist_stats, rng=rng)

        # Extracts discovery number for opposite vertex
        if opposite_vertex in dist_stats:
            discovery_num = dist_stats[opposite_vertex][0]
            opposite_discoveries.append(discovery_num)

            # Also stores for all vertices
            for vertex, discoveries in dist_stats.items():
                all_dist_stats[vertex].append(discoveries[0])

        # Reports progress
        if progress_callback is not None:
            progress_callback(i + 1, num_samples)

    # Computes statistics for opposite vertex
    opposite_mean = float(np.mean(opposite_discoveries))
    opposite_std = float(np.std(opposite_discoveries))
    opposite_min = int(np.min(opposite_discoveries))
    opposite_max = int(np.max(opposite_discoveries))
    opposite_median = float(np.median(opposite_discoveries))

    # Computes overall statistics for comparison
    all_means = []
    for vertex_discoveries in all_dist_stats.values():
        all_means.append(np.mean(vertex_discoveries))
    overall_mean = float(np.mean(all_means))

    # Gets graph properties
    num_vertices = hypercube.number_vertices()
    theoretical_all_vertices = (num_vertices - 1) / 2
    hamming_distance = hypercube.d  # Maximum distance in hypercube

    return {
        "start_vertex": start_vertex,
        "opposite_vertex": opposite_vertex,
        "hamming_distance": hamming_distance,
        "num_samples": num_samples,
        "num_vertices": num_vertices,
        "opposite_mean": opposite_mean,
        "opposite_std": opposite_std,
        "opposite_min": opposite_min,
        "opposite_max": opposite_max,
        "opposite_median": opposite_median,
        "opposite_discoveries": opposite_discoveries,
        "overall_mean": overall_mean,
        "theoretical_all_vertices": theoretical_all_vertices,
    }


def get_opposite_summary(stats: Dict[str, Any]) -> str:
    """
    Creates human-readable summary of opposite vertex analysis.

    Args:
        stats: Dictionary returned by collect_opposite_statistics().

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("OPPOSITE VERTEX ANALYSIS (HYPERCUBE)")
    lines.append("=" * 70)
    lines.append(f"Total vertices: {stats['num_vertices']}")
    lines.append(f"Hypercube dimension: {stats['hamming_distance']}")
    lines.append(f"Starting vertex: {stats['start_vertex']}")

    # Formats opposite vertex nicely
    opp_str = "".join(map(str, stats['opposite_vertex']))
    lines.append(f"Opposite vertex: ({', '.join(map(str, stats['opposite_vertex']))})")
    lines.append(f"Hamming distance: {stats['hamming_distance']} (maximum)")
    lines.append(f"Samples collected: {stats['num_samples']}")
    lines.append("")

    lines.append("--- Discovery Number Statistics ---")
    lines.append(f"Theoretical average (all vertices): {stats['theoretical_all_vertices']:.4f}")
    lines.append(f"Observed average (all vertices): {stats['overall_mean']:.4f}")
    lines.append("")
    lines.append(f"Opposite vertex mean: {stats['opposite_mean']:.4f}")
    lines.append(f"Opposite vertex std: {stats['opposite_std']:.4f}")
    lines.append(f"Opposite vertex min: {stats['opposite_min']}")
    lines.append(f"Opposite vertex max: {stats['opposite_max']}")
    lines.append(f"Opposite vertex median: {stats['opposite_median']:.4f}")
    lines.append("")

    # Comparison ratio
    ratio = stats['opposite_mean'] / stats['theoretical_all_vertices']
    lines.append("--- Key Insights ---")
    lines.append(f"Discovery ratio (opposite/average): {ratio:.4f}")
    lines.append(f"  (ratio < 1.0 means discovered earlier than average)")
    lines.append(f"  (ratio = 1.0 means discovered at typical time)")
    lines.append(f"  (ratio > 1.0 means discovered later than average)")
    lines.append("")

    if ratio < 0.9:
        lines.append("[OK] Opposite vertex discovered EARLIER than average")
        lines.append("  → DFS reaches maximally distant vertex quickly")
    elif ratio < 1.1:
        lines.append("≈ Opposite vertex discovered at TYPICAL time")
        lines.append("  → Consistent with expected (n-1)/2 behavior")
    else:
        lines.append("[WARNING] Opposite vertex discovered LATER than average")
        lines.append("  → DFS takes longer to reach maximally distant vertex")

    lines.append("")

    # Additional interpretation
    lines.append("--- Interpretation ---")
    lines.append(f"The opposite vertex is at Hamming distance {stats['hamming_distance']},")
    lines.append(f"meaning ALL {stats['hamming_distance']} bits must be flipped to reach it.")
    lines.append(f"This represents the farthest possible vertex from the start.")

    if ratio > 1.0:
        lines.append("")
        lines.append("Expected behavior: DFS explores local neighborhoods first,")
        lines.append("so distant vertices tend to be discovered later.")

    lines.append("=" * 70)

    return "\n".join(lines)


def compute_opposite_hitting_time_laplacian(
    hypercube: Hypercube
) -> Dict[str, Any]:
    """
    Computes hitting time for opposite vertex using Laplacian.

    Args:
        hypercube: Hypercube graph instance.

    Returns:
        Dictionary with opposite vertex hitting time statistics.
    """
    from dfs_analyzer.core.random_walk import compute_laplacian_hitting_times

    # Gets starting vertex and opposite vertex
    start_vertex = hypercube.get_start_vertex()
    opposite_vertex = get_opposite_vertex(hypercube)

    # Computes hitting times using Laplacian
    all_hitting_times = compute_laplacian_hitting_times(hypercube, start_vertex)

    # Extracts hitting time for opposite vertex
    opposite_hitting_time = all_hitting_times[opposite_vertex]

    # Computes overall statistics for comparison
    all_values = list(all_hitting_times.values())
    overall_mean = float(np.mean(all_values))

    # Gets graph properties
    num_vertices = hypercube.number_vertices()
    hamming_distance = hypercube.d

    return {
        "start_vertex": start_vertex,
        "opposite_vertex": opposite_vertex,
        "hamming_distance": hamming_distance,
        "num_vertices": num_vertices,
        "opposite_hitting_time": opposite_hitting_time,
        "overall_mean": overall_mean,
        "theoretical_average": (num_vertices - 1) / 2,
    }


def get_opposite_laplacian_summary(stats: Dict[str, Any]) -> str:
    """
    Creates human-readable summary of opposite vertex Laplacian analysis.

    Args:
        stats: Dictionary returned by compute_opposite_hitting_time_laplacian().

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("OPPOSITE VERTEX ANALYSIS - LAPLACIAN (HYPERCUBE)")
    lines.append("=" * 70)
    lines.append(f"Total vertices: {stats['num_vertices']}")
    lines.append(f"Hypercube dimension: {stats['hamming_distance']}")
    lines.append(f"Starting vertex: {stats['start_vertex']}")
    lines.append(f"Opposite vertex: ({', '.join(map(str, stats['opposite_vertex']))})")
    lines.append(f"Hamming distance: {stats['hamming_distance']} (maximum)")
    lines.append("")

    lines.append("--- Hitting Time Statistics ---")
    lines.append(f"Theoretical average (all vertices): {stats['theoretical_average']:.4f}")
    lines.append(f"Observed average (all vertices): {stats['overall_mean']:.4f}")
    lines.append("")
    lines.append(f"Opposite vertex hitting time: {stats['opposite_hitting_time']:.4f}")
    lines.append("")

    # Comparison ratio
    ratio = stats['opposite_hitting_time'] / stats['overall_mean']
    lines.append("--- Key Insights ---")
    lines.append(f"Hitting time ratio (opposite/average): {ratio:.4f}")
    lines.append(f"  (ratio < 1.0 means reached faster than average)")
    lines.append(f"  (ratio = 1.0 means reached at typical time)")
    lines.append(f"  (ratio > 1.0 means takes longer to reach)")
    lines.append("")

    if ratio < 0.9:
        lines.append("[OK] Opposite vertex reached FASTER than average")
    elif ratio < 1.1:
        lines.append("≈ Opposite vertex reached at TYPICAL time")
    else:
        lines.append("[WARNING] Opposite vertex takes LONGER to reach")

    lines.append("")
    lines.append("--- Theoretical Insight ---")
    lines.append(f"The Laplacian method provides exact expected hitting time")
    lines.append(f"for the vertex at maximum Hamming distance {stats['hamming_distance']}.")
    lines.append(f"This represents the theoretical value for random walks.")

    lines.append("=" * 70)

    return "\n".join(lines)
