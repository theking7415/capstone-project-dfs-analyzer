"""
Custom vertex pair analysis experiment runner.

Runs analysis for user-specified start and target vertex pairs.
"""

import os
from datetime import datetime
from typing import Optional, Callable, TypeVar

from dfs_analyzer.core.graphs import Graph
from dfs_analyzer.core.custom_vertex_analysis import (
    collect_custom_vertex_statistics,
    get_custom_vertex_summary,
)
import numpy as np

Vertex = TypeVar('Vertex')


class CustomVertexResults:
    """
    Container for custom vertex pair analysis results.

    Attributes:
        graph: The graph that was analyzed.
        start_vertex: Starting vertex.
        target_vertex: Target vertex.
        stats: Custom vertex statistics dictionary.
        method: Analysis method (always "rdfs").
        timestamp: When the experiment was run.
        output_path: Directory where results are saved.
    """

    def __init__(
        self,
        graph: Graph[Vertex],
        start_vertex: Vertex,
        target_vertex: Vertex,
        stats: dict,
        method: str,
        graph_desc: str,
        timestamp: Optional[str] = None,
        output_dir: str = "data_output",
    ):
        """
        Initializes custom vertex results.

        Args:
            graph: The graph that was analyzed.
            start_vertex: Starting vertex.
            target_vertex: Target vertex.
            stats: Statistics from custom vertex analysis.
            method: "rdfs" (only supported method).
            graph_desc: Graph description for naming.
            timestamp: Optional timestamp string.
            output_dir: Output directory.
        """
        self.graph = graph
        self.start_vertex = start_vertex
        self.target_vertex = target_vertex
        self.stats = stats
        self.method = method
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Sets output path
        experiment_name = f"{graph_desc}-custom-{method}"
        self.output_path = os.path.join(
            output_dir, f"{experiment_name}_{self.timestamp}"
        )

    def get_summary(self) -> str:
        """
        Gets human-readable summary of results.

        Returns:
            Formatted summary string.
        """
        return get_custom_vertex_summary(self.stats)

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


class CustomVertexRunner:
    """
    Orchestrates custom vertex pair analysis experiments.

    Runs focused analysis on a specific start-target vertex pair
    using RDFS (Randomized DFS) method.
    """

    def __init__(self):
        """Initializes the custom vertex runner."""
        pass

    def run(
        self,
        graph: Graph[Vertex],
        start_vertex: Vertex,
        target_vertex: Vertex,
        num_samples: int = 1000,
        method: str = "rdfs",
        rng_seed: int = None,
        output_dir: str = "data_output",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> CustomVertexResults:
        """
        Runs custom vertex pair analysis.

        Args:
            graph: Graph instance to analyze.
            start_vertex: Starting vertex for DFS.
            target_vertex: Target vertex to analyze.
            num_samples: Number of RDFS samples (only for RDFS method).
            method: "rdfs" (only supported method).
            rng_seed: Random seed for reproducibility.
            output_dir: Directory for output files.
            progress_callback: Optional callback for progress updates.

        Returns:
            CustomVertexResults object containing all results.

        Example:
            >>> from dfs_analyzer.core.graphs import Hypercube
            >>> cube = Hypercube(5)
            >>> runner = CustomVertexRunner()
            >>> results = runner.run(
            ...     cube,
            ...     start_vertex=(0,0,0,0,0),
            ...     target_vertex=(1,0,1,0,1),
            ...     num_samples=1000,
            ...     method="rdfs"
            ... )
            >>> print(results.get_summary())
        """
        # Validates that vertices exist in graph
        # (Note: For complete validation, would need to check graph.get_adj_list
        # or similar, but we'll rely on RDFS to catch invalid vertices)

        # Runs RDFS analysis
        # Sets up RNG
        rng = np.random.default_rng(rng_seed)

        # Collects custom vertex statistics
        stats = collect_custom_vertex_statistics(
            graph,
            start_vertex,
            target_vertex,
            num_samples,
            rng=rng,
            progress_callback=progress_callback
        )

        # Creates results object
        results = CustomVertexResults(
            graph=graph,
            start_vertex=start_vertex,
            target_vertex=target_vertex,
            stats=stats,
            method=method,
            graph_desc=graph.desc(),
            output_dir=output_dir,
        )

        # Saves results
        results.save()

        return results
