"""
Batch experiment runner for G(n,p) random graphs.

Runs RDFS experiments on multiple randomly generated G(n,p) graphs
and aggregates statistics across all graphs.
"""

import os
import numpy as np
from datetime import datetime
from typing import Optional, Callable
from collections import defaultdict

from dfs_analyzer.core.gnp_graph import generate_connected_gnp
from dfs_analyzer.core.rdfs import rdfs, get_summary_stats
from dfs_analyzer.core.statistics import validate_result
from dfs_analyzer.experiments.config import ExperimentConfig


class GNPBatchResults:
    """
    Container for G(n,p) batch experiment results.

    Stores aggregate statistics across multiple random graphs.

    Attributes:
        n: Number of vertices per graph.
        p: Edge probability.
        num_graphs: Number of graphs generated.
        num_samples_per_graph: RDFS samples per graph.
        graph_stats: Statistics for each individual graph.
        aggregate_stats: Overall statistics across all graphs.
        timestamp: When experiment was run.
        output_path: Directory where results are saved.
    """

    def __init__(
        self,
        n: int,
        p: float,
        num_graphs: int,
        num_samples_per_graph: int,
        graph_stats: list[dict],
        timestamp: Optional[str] = None,
        output_dir: str = "data_output",
    ):
        """
        Initializes batch results.

        Args:
            n: Number of vertices.
            p: Edge probability.
            num_graphs: Number of graphs analyzed.
            num_samples_per_graph: RDFS samples per graph.
            graph_stats: List of statistics dictionaries for each graph.
            timestamp: Optional timestamp string.
            output_dir: Output directory.
        """
        self.n = n
        self.p = p
        self.num_graphs = num_graphs
        self.num_samples_per_graph = num_samples_per_graph
        self.graph_stats = graph_stats
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Computes aggregate statistics
        self.aggregate_stats = self._compute_aggregate_stats()

        # Sets output path
        experiment_name = f"gnp_n{n}_p{p:.4f}_graphs{num_graphs}"
        self.output_path = os.path.join(output_dir, f"{experiment_name}_{self.timestamp}")

    def _compute_aggregate_stats(self) -> dict:
        """
        Computes aggregate statistics across all graphs.

        Returns:
            Dictionary with overall mean, std, and other statistics.
        """
        # Extracts per-graph overall means
        overall_means = [stats["overall_mean"] for stats in self.graph_stats]
        overall_stds = [stats["overall_std"] for stats in self.graph_stats]

        # Computes statistics of overall means
        mean_of_means = float(np.mean(overall_means))
        std_of_means = float(np.std(overall_means))
        min_mean = float(np.min(overall_means))
        max_mean = float(np.max(overall_means))

        # Computes statistics of overall stds
        mean_of_stds = float(np.mean(overall_stds))
        std_of_stds = float(np.std(overall_stds))

        # Validates result on aggregate
        theoretical_value = (self.n - 1) / 2
        validation = validate_result(
            num_vertices=self.n,
            observed_mean=mean_of_means,
            tolerance=0.01,
        )

        return {
            "mean_of_means": mean_of_means,
            "std_of_means": std_of_means,
            "min_mean": min_mean,
            "max_mean": max_mean,
            "mean_of_stds": mean_of_stds,
            "std_of_stds": std_of_stds,
            "theoretical_value": theoretical_value,
            "validation": validation,
        }

    def get_summary(self) -> str:
        """
        Gets human-readable summary of batch results.

        Returns:
            Formatted summary string.
        """
        agg = self.aggregate_stats
        val = agg["validation"]

        lines = []
        lines.append("=" * 70)
        lines.append("G(n,p) BATCH EXPERIMENT RESULTS")
        lines.append("=" * 70)
        lines.append(f"Graph parameters:")
        lines.append(f"  n (vertices): {self.n}")
        lines.append(f"  p (edge probability): {self.p:.4f}")
        lines.append(f"  Expected edges per graph: {self.p * self.n * (self.n - 1) / 2:.1f}")
        lines.append(f"")
        lines.append(f"Experiment parameters:")
        lines.append(f"  Number of graphs generated: {self.num_graphs}")
        lines.append(f"  RDFS samples per graph: {self.num_samples_per_graph}")
        lines.append(f"  Total RDFS runs: {self.num_graphs * self.num_samples_per_graph}")
        lines.append("")

        lines.append("--- Aggregate Statistics (across all graphs) ---")
        lines.append(f"Mean of overall means: {agg['mean_of_means']:.4f}")
        lines.append(f"Std of overall means: {agg['std_of_means']:.4f}")
        lines.append(f"Min mean (best graph): {agg['min_mean']:.4f}")
        lines.append(f"Max mean (worst graph): {agg['max_mean']:.4f}")
        lines.append("")
        lines.append(f"Mean of overall stds: {agg['mean_of_stds']:.4f}")
        lines.append(f"Std of overall stds: {agg['std_of_stds']:.4f}")
        lines.append("")

        lines.append("--- Result Validation ---")
        lines.append(f"Theoretical (n-1)/2: {agg['theoretical_value']:.4f}")
        lines.append(f"Observed (aggregate): {agg['mean_of_means']:.4f}")
        lines.append(f"Absolute error: {val['absolute_error']:.6f}")
        lines.append(f"Relative error: {val['relative_error'] * 100:.4f}%")
        lines.append("")

        if val["is_valid"]:
            lines.append(f"[OK] RESULT VALID (within {val['tolerance'] * 100:.2f}% tolerance)")
        else:
            lines.append(f"[FAIL] RESULT INVALID (exceeds {val['tolerance'] * 100:.2f}% tolerance)")

        lines.append("")
        lines.append("--- Interpretation ---")
        lines.append(f"Note: G(n,p) graphs are NOT regular or symmetric.")
        lines.append(f"The expected (n-1)/2 behavior applies to symmetric regular graphs.")
        lines.append(f"This experiment explores whether the expected behavior holds")
        lines.append(f"approximately for random graphs with sufficient connectivity.")
        lines.append("")

        if val["is_valid"]:
            lines.append(f"Result: Despite being non-regular, G({self.n}, {self.p:.4f}) graphs")
            lines.append(f"        show average discovery numbers close to (n-1)/2.")
        else:
            lines.append(f"Result: G({self.n}, {self.p:.4f}) graphs deviate from (n-1)/2,")
            lines.append(f"        as expected for non-regular graphs.")

        lines.append("=" * 70)

        return "\n".join(lines)

    def save(self):
        """
        Saves results to output directory.

        Creates summary.txt with aggregate statistics.
        Optionally saves per-graph statistics to CSV.
        """
        os.makedirs(self.output_path, exist_ok=True)

        # Saves summary
        summary_path = os.path.join(self.output_path, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.get_summary())

        # Saves per-graph statistics to CSV
        csv_path = os.path.join(self.output_path, "per_graph_stats.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            # Header
            f.write("graph_id,num_edges,overall_mean,overall_std,min_discovery,max_discovery\n")

            # Data rows
            for i, stats in enumerate(self.graph_stats):
                f.write(
                    f"{i+1},"
                    f"{stats['num_edges']},"
                    f"{stats['overall_mean']:.6f},"
                    f"{stats['overall_std']:.6f},"
                    f"{stats['min_discovery']},"
                    f"{stats['max_discovery']}\n"
                )


class GNPBatchRunner:
    """
    Orchestrates batch experiments on G(n,p) random graphs.

    Generates multiple connected G(n,p) graphs, runs RDFS on each,
    and aggregates statistics across all graphs.
    """

    def __init__(self):
        """Initializes batch runner."""
        pass

    def run(
        self,
        n: int,
        p: float,
        num_graphs: int,
        num_samples_per_graph: int,
        rng_seed: int = None,
        output_dir: str = "data_output",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> GNPBatchResults:
        """
        Runs batch experiment on multiple G(n,p) graphs.

        Args:
            n: Number of vertices per graph.
            p: Edge probability.
            num_graphs: Number of graphs to generate and analyze.
            num_samples_per_graph: Number of RDFS samples per graph.
            rng_seed: Random seed for reproducibility.
            output_dir: Directory for output files.
            progress_callback: Optional callback(current, total, message).

        Returns:
            GNPBatchResults with aggregate statistics.

        Example:
            >>> runner = GNPBatchRunner()
            >>> results = runner.run(n=100, p=0.05, num_graphs=50, num_samples_per_graph=1000)
            >>> print(results.get_summary())
        """
        # Creates master RNG
        master_rng = np.random.default_rng(rng_seed)

        # Stores statistics for each graph
        graph_stats_list = []

        # Processes each graph
        for graph_idx in range(num_graphs):
            if progress_callback:
                progress_callback(
                    graph_idx + 1,
                    num_graphs,
                    f"Processing graph {graph_idx + 1}/{num_graphs}"
                )

            # Step 1: Generates connected graph
            graph_seed = master_rng.integers(0, 2**31)
            try:
                graph = generate_connected_gnp(n, p, rng_seed=graph_seed, max_attempts=1000)
            except ValueError as e:
                print(f"  Warning: Skipping graph {graph_idx + 1} - {e}")
                continue

            # Step 2: Runs RDFS multiple times
            rdfs_seed = master_rng.integers(0, 2**31)
            rdfs_rng = np.random.default_rng(rdfs_seed)

            all_dist_stats = defaultdict(list)
            for sample_idx in range(num_samples_per_graph):
                dist_stats = defaultdict(list)
                rdfs(graph, graph.get_start_vertex(), dist_stats=dist_stats, rng=rdfs_rng)

                # Collects discovery numbers
                for vertex, discoveries in dist_stats.items():
                    all_dist_stats[vertex].append(discoveries[0])

            # Step 3: Computes summary statistics for this graph
            summary_stats = get_summary_stats(all_dist_stats)

            # Computes overall statistics across all vertices
            all_vertex_means = []
            min_discovery = float('inf')
            max_discovery = float('-inf')

            for vertex, vertex_stats in summary_stats.items():
                vertex_mean = vertex_stats.mean
                all_vertex_means.append(vertex_mean)
                min_discovery = min(min_discovery, vertex_stats.minmax[0])
                max_discovery = max(max_discovery, vertex_stats.minmax[1])

            overall_mean = float(np.mean(all_vertex_means))
            overall_std = float(np.std(all_vertex_means))

            # Stores graph statistics
            graph_stats = {
                "graph_id": graph_idx + 1,
                "num_edges": graph.number_edges(),
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "min_discovery": min_discovery,
                "max_discovery": max_discovery,
            }
            graph_stats_list.append(graph_stats)

        # Creates results object
        results = GNPBatchResults(
            n=n,
            p=p,
            num_graphs=len(graph_stats_list),
            num_samples_per_graph=num_samples_per_graph,
            graph_stats=graph_stats_list,
            output_dir=output_dir,
        )

        # Saves results
        results.save()

        return results
