"""
Neighbor analysis experiment runner.

Runs analysis focused on immediate neighbors of the starting vertex.
"""

import os
from datetime import datetime
from typing import Optional, Callable

from dfs_analyzer.core.graphs import Hypercube, GeneralizedPetersen
from dfs_analyzer.core.neighbor_analysis import (
    collect_neighbor_statistics,
    get_neighbor_summary,
)
from dfs_analyzer.experiments.config import ExperimentConfig
import numpy as np


class NeighborAnalysisResults:
    """
    Container for neighbor analysis results.

    Attributes:
        graph: The graph that was analyzed.
        config: The experiment configuration used.
        stats: Neighbor statistics dictionary.
        method: Analysis method (always "rdfs").
        timestamp: When the experiment was run.
        output_path: Directory where results are saved.
    """

    def __init__(
        self,
        graph,
        config: ExperimentConfig,
        stats: dict,
        method: str,
        timestamp: Optional[str] = None,
    ):
        """
        Initializes neighbor analysis results.

        Args:
            graph: The graph that was analyzed.
            config: ExperimentConfig instance.
            stats: Statistics from neighbor analysis.
            method: "rdfs" (only supported method).
            timestamp: Optional timestamp string.
        """
        self.graph = graph
        self.config = config
        self.stats = stats
        self.method = method
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Sets output path
        experiment_name = (
            config.experiment_name or f"{config.get_auto_experiment_name()}-neighbors-{method}"
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
        return get_neighbor_summary(self.stats)

    def save(self):
        """
        Saves results to summary.txt.

        Creates the output directory and saves only summary.txt by default.
        """
        os.makedirs(self.output_path, exist_ok=True)

        # Always saves summary text
        filepath = os.path.join(self.output_path, "summary.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_summary())


class NeighborAnalysisRunner:
    """
    Orchestrates neighbor analysis experiments.

    Runs focused analysis on immediate neighbors of the starting vertex
    using RDFS (Randomized DFS) method.
    """

    def __init__(self):
        """Initializes the neighbor analysis runner."""
        pass

    def run(
        self,
        config: ExperimentConfig,
        method: str = "rdfs",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> NeighborAnalysisResults:
        """
        Runs neighbor analysis based on configuration.

        Args:
            config: ExperimentConfig specifying the experiment parameters.
            method: "rdfs" (only supported method).
            progress_callback: Optional callback for progress updates.

        Returns:
            NeighborAnalysisResults object containing all results.

        Example:
            >>> config = ExperimentConfig(dimension=5, num_samples=1000)
            >>> runner = NeighborAnalysisRunner()
            >>> results = runner.run(config, method="rdfs")
            >>> print(results.get_summary())
        """
        # Step 1: Creates the graph
        graph = self._create_graph(config)

        # Step 2: Runs RDFS analysis
        # Sets up RNG
        rng = np.random.default_rng(config.rng_seed)

        # Collects neighbor statistics
        stats = collect_neighbor_statistics(
            graph,
            config.num_samples,
            rng=rng,
            progress_callback=progress_callback
        )

        # Step 3: Creates results object
        results = NeighborAnalysisResults(
            graph=graph,
            config=config,
            stats=stats,
            method=method,
        )

        # Step 4: Saves results
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
