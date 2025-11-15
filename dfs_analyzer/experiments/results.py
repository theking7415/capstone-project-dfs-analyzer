"""
Experiment results storage and export.
"""

import csv
import json
import os
import pickle
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from dfs_analyzer.core.graphs import Graph
from dfs_analyzer.core.statistics import (
    compute_overall_average,
    compute_overall_std,
    validate_conjecture,
    format_validation_result,
)


class ExperimentResults:
    """
    Container for experiment results with export capabilities.

    Attributes:
        graph: The graph that was analyzed.
        config: The experiment configuration used.
        dist_stats: Raw discovery number statistics.
        summary_stats: Summary statistics for each vertex.
        timestamp: When the experiment was run.
        output_path: Directory where results are saved.
    """

    def __init__(
        self,
        graph: Graph,
        config: Any,
        dist_stats: dict,
        summary_stats: dict,
        timestamp: Optional[str] = None,
    ):
        """
        Initialize experiment results.

        Args:
            graph: The graph that was analyzed.
            config: ExperimentConfig instance.
            dist_stats: Raw discovery statistics from collect_statistics().
            summary_stats: Summary statistics from get_summary_stats().
            timestamp: Optional timestamp string (auto-generated if None).
        """
        self.graph = graph
        self.config = config
        self.dist_stats = dist_stats
        self.summary_stats = summary_stats
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Compute overall statistics
        self.overall_mean = compute_overall_average(summary_stats)
        self.overall_std = compute_overall_std(summary_stats)

        # Validate conjecture
        self.validation = validate_conjecture(
            graph.number_vertices(), self.overall_mean
        )

        # Set output path
        experiment_name = (
            config.experiment_name or config.get_auto_experiment_name()
        )
        self.output_path = os.path.join(
            config.output_dir, f"{experiment_name}_{self.timestamp}"
        )

        # Visualization figure (created on demand)
        self._figure: Optional[plt.Figure] = None

    def get_summary(self) -> str:
        """
        Get a human-readable summary of results.

        Returns:
            Formatted string with experiment summary.
        """
        lines = [
            "=" * 70,
            "EXPERIMENT RESULTS",
            "=" * 70,
            f"Graph: {self.config.get_graph_description()}",
            f"Samples: {self.config.num_samples}",
            f"RNG Seed: {self.config.rng_seed}",
            f"Timestamp: {self.timestamp}",
            "",
            "--- Statistical Summary ---",
            format_validation_result(self.validation),
            f"Standard deviation across vertices: {self.overall_std:.4f}",
            "",
        ]

        if self.validation["is_valid"]:
            lines.append("✓ CONJECTURE VALIDATED")
        else:
            lines.append("✗ CONJECTURE NOT VALIDATED")

        lines.append("=" * 70)

        return "\n".join(lines)

    def get_summary_dict(self) -> dict[str, Any]:
        """
        Get summary as a dictionary.

        Returns:
            Dictionary with key results.
        """
        return {
            "graph_type": self.config.graph_type,
            "dimension": self.config.dimension,
            "num_vertices": self.graph.number_vertices(),
            "num_samples": self.config.num_samples,
            "theoretical": self.validation["theoretical_value"],
            "observed": self.validation["observed_value"],
            "absolute_error": self.validation["absolute_error"],
            "relative_error": self.validation["relative_error"],
            "is_valid": self.validation["is_valid"],
            "std_dev": self.overall_std,
            "timestamp": self.timestamp,
        }

    def save(self):
        """
        Save results in all configured export formats.

        Creates the output directory and saves files in each requested format.
        """
        os.makedirs(self.output_path, exist_ok=True)

        # Always save summary text
        self._save_summary_txt()

        # Save in requested formats
        for fmt in self.config.export_formats:
            if fmt == "csv":
                self._save_csv()
            elif fmt == "json":
                self._save_json()
            elif fmt == "txt":
                self._save_detailed_txt()
            elif fmt == "pickle":
                self._save_pickle()

        # Save plot if requested
        if self.config.save_plots:
            self._save_plot()

    def _save_summary_txt(self):
        """Save brief summary to summary.txt."""
        filepath = os.path.join(self.output_path, "summary.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_summary())

    def _save_csv(self):
        """Save per-vertex statistics to CSV."""
        filepath = os.path.join(self.output_path, "data.csv")

        # Sort vertices for consistent ordering
        sorted_vertices = sorted(self.summary_stats.keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "vertex",
                "mean",
                "variance",
                "std_dev",
                "min",
                "max",
                "skewness",
                "kurtosis",
            ])

            # Write data rows
            for vertex in sorted_vertices:
                stats = self.summary_stats[vertex]
                vertex_str = "".join(map(str, vertex)) if isinstance(vertex, tuple) else str(vertex)

                writer.writerow([
                    vertex_str,
                    stats.mean,
                    stats.variance,
                    np.sqrt(stats.variance),
                    stats.minmax[0],
                    stats.minmax[1],
                    stats.skewness,
                    stats.kurtosis,
                ])

    def _save_json(self):
        """Save results to JSON."""
        filepath = os.path.join(self.output_path, "data.json")

        # Convert results to JSON-serializable format
        sorted_vertices = sorted(self.summary_stats.keys())

        # Helper to convert numpy types to Python types
        def convert_to_python(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        data = {
            "metadata": {
                "graph_type": self.config.graph_type,
                "dimension": self.config.dimension,
                "num_vertices": self.graph.number_vertices(),
                "num_samples": self.config.num_samples,
                "rng_seed": self.config.rng_seed,
                "timestamp": self.timestamp,
            },
            "validation": {k: convert_to_python(v) for k, v in self.validation.items()},
            "overall_statistics": {
                "mean": float(self.overall_mean),
                "std_dev": float(self.overall_std),
            },
            "per_vertex_statistics": {},
        }

        for vertex in sorted_vertices:
            stats = self.summary_stats[vertex]
            vertex_key = "".join(map(str, vertex)) if isinstance(vertex, tuple) else str(vertex)

            data["per_vertex_statistics"][vertex_key] = {
                "mean": float(stats.mean),
                "variance": float(stats.variance),
                "std_dev": float(np.sqrt(stats.variance)),
                "min": int(stats.minmax[0]),
                "max": int(stats.minmax[1]),
                "skewness": float(stats.skewness),
                "kurtosis": float(stats.kurtosis),
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_detailed_txt(self):
        """Save detailed statistics to text file."""
        filepath = os.path.join(self.output_path, "detailed_stats.txt")

        sorted_vertices = sorted(self.summary_stats.keys())

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_summary())
            f.write("\n\n")
            f.write("=" * 70 + "\n")
            f.write("PER-VERTEX STATISTICS\n")
            f.write("=" * 70 + "\n\n")

            for vertex in sorted_vertices:
                stats = self.summary_stats[vertex]
                vertex_str = "".join(map(str, vertex)) if isinstance(vertex, tuple) else str(vertex)

                f.write(f"Vertex: {vertex_str}\n")
                f.write(f"  Mean:     {stats.mean:.4f}\n")
                f.write(f"  Variance: {stats.variance:.4f}\n")
                f.write(f"  Std Dev:  {np.sqrt(stats.variance):.4f}\n")
                f.write(f"  Min:      {stats.minmax[0]}\n")
                f.write(f"  Max:      {stats.minmax[1]}\n")
                f.write(f"  Skewness: {stats.skewness:.4f}\n")
                f.write(f"  Kurtosis: {stats.kurtosis:.4f}\n")
                f.write("\n")

    def _save_pickle(self):
        """Save raw data to pickle file."""
        filepath = os.path.join(self.output_path, "data.pickle")
        with open(filepath, "wb") as f:
            pickle.dump((self.graph, self.dist_stats), f)

    def _save_plot(self):
        """Save visualization plot."""
        if self._figure is None:
            self._figure = self.graph.plot_means_vars(self.summary_stats)

        if self._figure is not None:
            filepath = os.path.join(self.output_path, "visualization.png")
            self._figure.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(self._figure)

    @property
    def plot_figure(self) -> Optional[plt.Figure]:
        """
        Get the matplotlib figure for the visualization.

        Returns:
            The figure object, creating it if necessary.
        """
        if self._figure is None:
            self._figure = self.graph.plot_means_vars(self.summary_stats)
        return self._figure
