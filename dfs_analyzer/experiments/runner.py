"""
Experiment runner for orchestrating DFS analysis.
"""

from typing import Callable, Optional

import numpy as np

from dfs_analyzer.core.graphs import Hypercube
from dfs_analyzer.core.rdfs import collect_statistics, get_summary_stats
from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.results import ExperimentResults


class ExperimentRunner:
    """
    Orchestrates the execution of DFS analysis experiments.

    This class handles the full pipeline:
    1. Create the graph based on configuration
    2. Run RDFS with progress tracking
    3. Compute statistics
    4. Create results object
    5. Save results
    """

    def __init__(self):
        """Initialize the experiment runner."""
        pass

    def run(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExperimentResults:
        """
        Run a complete experiment based on configuration.

        Args:
            config: ExperimentConfig specifying the experiment parameters.
            progress_callback: Optional callback for progress updates.
                Called with (current_sample, total_samples).

        Returns:
            ExperimentResults object containing all results.

        Example:
            >>> config = ExperimentConfig(dimension=5, num_samples=1000)
            >>> runner = ExperimentRunner()
            >>> results = runner.run(config)
            >>> print(results.get_summary())
        """
        # Step 1: Create the graph
        graph = self._create_graph(config)

        # Step 2: Set up RNG
        rng = np.random.default_rng(config.rng_seed)

        # Step 3: Collect statistics with progress tracking
        dist_stats = collect_statistics(
            graph, config.num_samples, rng=rng, progress_callback=progress_callback
        )

        # Step 4: Compute summary statistics
        summary_stats = get_summary_stats(dist_stats)

        # Step 5: Create results object
        results = ExperimentResults(
            graph=graph,
            config=config,
            dist_stats=dist_stats,
            summary_stats=summary_stats,
        )

        # Step 6: Save results
        results.save()

        return results

    def _create_graph(self, config: ExperimentConfig):
        """
        Create a graph based on configuration.

        Args:
            config: ExperimentConfig instance.

        Returns:
            A Graph instance.

        Raises:
            ValueError: If graph type is not supported.
        """
        if config.graph_type == "hypercube":
            return Hypercube(config.dimension)
        else:
            raise ValueError(
                f"Unsupported graph type: {config.graph_type}. "
                f"Currently supported: hypercube"
            )
