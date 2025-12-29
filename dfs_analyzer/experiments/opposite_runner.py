"""
Opposite vertex analysis experiment runner.

Runs analysis focused on the diagonally opposite vertex in hypercubes.
"""

import os
from datetime import datetime
from typing import Optional, Callable

from dfs_analyzer.core.graphs import Hypercube
from dfs_analyzer.core.opposite_analysis import (
    collect_opposite_statistics,
    get_opposite_summary,
)
from dfs_analyzer.experiments.config import ExperimentConfig
import numpy as np


class OppositeAnalysisResults:
    """
    Container for opposite vertex analysis results.

    Attributes:
        hypercube: The hypercube that was analyzed.
        config: The experiment configuration used.
        stats: Opposite vertex statistics dictionary.
        method: Analysis method (always "rdfs").
        timestamp: When the experiment was run.
        output_path: Directory where results are saved.
    """

    def __init__(
        self,
        hypercube: Hypercube,
        config: ExperimentConfig,
        stats: dict,
        method: str,
        timestamp: Optional[str] = None,
    ):
        """
        Initializes opposite analysis results.

        Args:
            hypercube: The hypercube that was analyzed.
            config: ExperimentConfig instance.
            stats: Statistics from opposite analysis.
            method: "rdfs" (only supported method).
            timestamp: Optional timestamp string.
        """
        self.hypercube = hypercube
        self.config = config
        self.stats = stats
        self.method = method
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Sets output path
        experiment_name = (
            config.experiment_name or f"{config.get_auto_experiment_name()}-opposite-{method}"
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
        return get_opposite_summary(self.stats)

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


class OppositeAnalysisRunner:
    """
    Orchestrates opposite vertex analysis experiments.

    Runs focused analysis on the diagonally opposite vertex in hypercubes
    using RDFS (Randomized DFS) method.

    Note: Only works with hypercube graphs (not Petersen graphs).
    """

    def __init__(self):
        """Initializes the opposite analysis runner."""
        pass

    def run(
        self,
        config: ExperimentConfig,
        method: str = "rdfs",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> OppositeAnalysisResults:
        """
        Runs opposite vertex analysis based on configuration.

        Args:
            config: ExperimentConfig specifying the experiment parameters.
            method: "rdfs" (only supported method).
            progress_callback: Optional callback for progress updates.

        Returns:
            OppositeAnalysisResults object containing all results.

        Raises:
            ValueError: If graph type is not hypercube.

        Example:
            >>> config = ExperimentConfig(dimension=5, num_samples=1000)
            >>> runner = OppositeAnalysisRunner()
            >>> results = runner.run(config, method="rdfs")
            >>> print(results.get_summary())
        """
        # Validates graph type
        if config.graph_type != "hypercube":
            raise ValueError(
                "Opposite vertex analysis only supports hypercube graphs. "
                f"Current graph type: {config.graph_type}"
            )

        # Step 1: Creates the hypercube
        hypercube = Hypercube(config.dimension)

        # Step 2: Runs RDFS analysis
        # Sets up RNG
        rng = np.random.default_rng(config.rng_seed)

        # Collects opposite vertex statistics
        stats = collect_opposite_statistics(
            hypercube,
            config.num_samples,
            rng=rng,
            progress_callback=progress_callback
        )

        # Step 3: Creates results object
        results = OppositeAnalysisResults(
            hypercube=hypercube,
            config=config,
            stats=stats,
            method=method,
        )

        # Step 4: Saves results
        results.save()

        return results
