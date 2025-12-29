"""
Random walk experiment runner.

Runs Laplacian-based random walk analysis on graphs.
"""

import os
from datetime import datetime
from typing import Optional

from dfs_analyzer.core.graphs import Graph, Hypercube, GeneralizedPetersen
from dfs_analyzer.core.random_walk import (
    compute_laplacian_hitting_times,
    get_laplacian_summary_stats,
)
from dfs_analyzer.core.statistics import validate_result
from dfs_analyzer.experiments.config import ExperimentConfig


class RandomWalkResults:
    """
    Container for random walk experiment results.

    Attributes:
        graph: The graph that was analyzed.
        config: The experiment configuration used.
        hitting_times: Dictionary of hitting times for each vertex.
        summary_stats: Summary statistics.
        validation: Result validation information.
        overall_mean: Mean hitting time across all vertices.
        overall_std: Standard deviation across all vertices.
    """

    def __init__(
        self,
        graph: Graph,
        config: ExperimentConfig,
        hitting_times: dict,
        summary_stats: dict,
        timestamp: Optional[str] = None,
    ):
        """
        Initializes random walk results.

        Args:
            graph: The graph that was analyzed.
            config: ExperimentConfig instance.
            hitting_times: Hitting times from compute_laplacian_hitting_times().
            summary_stats: Summary statistics from get_laplacian_summary_stats().
            timestamp: Optional timestamp string (auto-generated if None).
        """
        self.graph = graph
        self.config = config
        self.hitting_times = hitting_times
        self.summary_stats = summary_stats
        self.overall_mean = summary_stats["mean"]
        self.overall_std = summary_stats["std"]
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Validates result
        self.validation = validate_result(
            graph.number_vertices(), self.overall_mean
        )

        # Sets output path
        experiment_name = (
            config.experiment_name or f"{config.get_auto_experiment_name()}-laplacian"
        )
        self.output_path = os.path.join(
            config.output_dir, f"{experiment_name}_{self.timestamp}"
        )

    def get_summary(self) -> str:
        """
        Gets human-readable summary of results.

        Returns:
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("RANDOM WALK (LAPLACIAN) RESULTS")
        lines.append("=" * 70)
        lines.append(f"Graph: {self.config.get_graph_description()}")
        lines.append(f"Method: Laplacian matrix (theoretical)")
        lines.append("")
        lines.append("--- Statistical Summary ---")
        lines.append(f"Theoretical (n-1)/2: {self.validation['theoretical_value']:.4f}")
        lines.append(f"Observed mean: {self.overall_mean:.4f}")
        lines.append(f"Absolute error: {self.validation['absolute_error']:.6f}")
        lines.append(f"Relative error: {self.validation['relative_error'] * 100:.4f}%")
        lines.append(f"Status: {'[OK] VALID' if self.validation['is_valid'] else '[FAIL] INVALID'} (tolerance: {self.validation['tolerance'] * 100:.2f}%)")
        lines.append(f"Standard deviation: {self.overall_std:.4f}")
        lines.append(f"Min hitting time: {self.summary_stats['min']:.4f}")
        lines.append(f"Max hitting time: {self.summary_stats['max']:.4f}")
        lines.append(f"Median hitting time: {self.summary_stats['median']:.4f}")
        lines.append("")

        if self.validation["is_valid"]:
            lines.append("[OK] RESULT VALID (Laplacian method)")
        else:
            lines.append("[FAIL] RESULT INVALID")

        lines.append("=" * 70)

        return "\n".join(lines)

    def save(self):
        """
        Saves results to summary.txt.

        Creates the output directory and saves only summary.txt by default
        (matching the new storage-efficient approach).
        """
        os.makedirs(self.output_path, exist_ok=True)

        # Always save summary text
        filepath = os.path.join(self.output_path, "summary.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_summary())


class RandomWalkRunner:
    """
    Orchestrates random walk analysis experiments.

    Uses the Laplacian matrix approach to compute expected hitting times,
    providing theoretical results for comparison with empirical RDFS.
    """

    def __init__(self):
        """Initializes the random walk runner."""
        pass

    def run(self, config: ExperimentConfig) -> RandomWalkResults:
        """
        Runs a random walk analysis based on configuration.

        Args:
            config: ExperimentConfig specifying the experiment parameters.

        Returns:
            RandomWalkResults object containing all results.

        Example:
            >>> config = ExperimentConfig(dimension=5, graph_type="hypercube")
            >>> runner = RandomWalkRunner()
            >>> results = runner.run(config)
            >>> print(results.get_summary())
        """
        # Step 1: Creates the graph
        graph = self._create_graph(config)

        # Step 2: Computes hitting times using Laplacian
        hitting_times = compute_laplacian_hitting_times(graph)

        # Step 3: Computes summary statistics
        summary_stats = get_laplacian_summary_stats(hitting_times)

        # Step 4: Creates results object
        results = RandomWalkResults(
            graph=graph,
            config=config,
            hitting_times=hitting_times,
            summary_stats=summary_stats,
        )

        # Step 5: Saves results
        results.save()

        return results

    def _create_graph(self, config: ExperimentConfig):
        """
        Creates a graph based on configuration.

        Args:
            config: ExperimentConfig instance.

        Returns:
            A Graph instance.

        Raises:
            ValueError: If graph type is not supported.
        """
        if config.graph_type == "hypercube":
            return Hypercube(config.dimension)
        elif config.graph_type == "petersen":
            return GeneralizedPetersen(config.dimension, config.petersen_k)
        else:
            raise ValueError(
                f"Unsupported graph type: {config.graph_type}. "
                f"Currently supported: hypercube, petersen"
            )
