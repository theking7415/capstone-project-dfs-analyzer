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
    validate_result,
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

        # Computes overall statistics
        self.overall_mean = compute_overall_average(summary_stats)
        self.overall_std = compute_overall_std(summary_stats)

        # Validates result
        self.validation = validate_result(
            graph.number_vertices(), self.overall_mean
        )

        # Sets output path
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
            lines.append("[OK] RESULT VALID")
        else:
            lines.append("[FAIL] RESULT INVALID")

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
        Save results based on configuration flags.

        Creates the output directory and saves files according to user preferences.
        summary.txt is always saved; other outputs are optional.
        """
        os.makedirs(self.output_path, exist_ok=True)

        # Always save summary text
        self._save_summary_txt()

        # Saves CSV if requested
        if self.config.save_csv:
            self._save_csv()

        # Saves detailed stats if requested
        if self.config.save_detailed_stats:
            self._save_detailed_txt()

        # Saves additional formats if requested
        for fmt in self.config.export_formats:
            if fmt == "json":
                self._save_json()
            elif fmt == "pickle":
                self._save_pickle()

        # Saves plot if requested
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

        # Sorts vertices for consistent ordering
        sorted_vertices = sorted(self.summary_stats.keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Writes header
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

            # Writes data rows
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

        # Converts results to JSON-serializable format
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
        """Save visualization plots."""
        # Skip per-vertex bar chart entirely (not useful for analysis)
        # Only generate histogram and layer analysis plots
        num_vertices = self.graph.number_vertices()
        print(f"  Skipping visualization.png (per-vertex bar chart disabled)")

        # Generates and save histogram
        histogram_fig = self._create_histogram_plot()
        if histogram_fig is not None:
            filepath = os.path.join(self.output_path, "histogram.png")
            histogram_fig.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(histogram_fig)

        # Generates and save layer analysis (graph distance)
        layer_fig = self._create_layer_analysis_plot()
        if layer_fig is not None:
            filepath = os.path.join(self.output_path, "layer_analysis.png")
            layer_fig.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(layer_fig)
            # Generates corresponding statistics table
            self._create_layer_statistics_table()

        # Skip L1/L2 analysis for graphs where coordinate distances don't reflect structure
        from dfs_analyzer.core.graphs import GeneralizedPetersen, TriangularLattice, TorusGrid, HexagonalLattice, CompleteGraph, NDGrid
        from dfs_analyzer.core.gnp_graph import ErdosRenyiGraph
        skip_coordinate_analysis = isinstance(self.graph, (GeneralizedPetersen, TriangularLattice, TorusGrid, HexagonalLattice, CompleteGraph, NDGrid, ErdosRenyiGraph))

        if not skip_coordinate_analysis:
            # Generates and save L1 layer analysis
            l1_fig = self._create_l1_layer_analysis_plot()
            if l1_fig is not None:
                filepath = os.path.join(self.output_path, "layer_analysis_l1.png")
                l1_fig.savefig(filepath, dpi=150, bbox_inches="tight")
                plt.close(l1_fig)
                # Generates corresponding statistics table
                self._create_l1_layer_statistics_table()

            # Generates and save L2 layer analysis
            l2_fig = self._create_l2_layer_analysis_plot()
            if l2_fig is not None:
                filepath = os.path.join(self.output_path, "layer_analysis_l2.png")
                l2_fig.savefig(filepath, dpi=150, bbox_inches="tight")
                plt.close(l2_fig)
                # Generates corresponding statistics table
                self._create_l2_layer_statistics_table()

    def _create_histogram_plot(self):
        """
        Creates histogram showing distribution of vertices by average discovery number.

        Groups vertices into buckets based on their mean discovery number and
        shows how many vertices fall into each bucket.
        Excludes the origin vertex which always has discovery number 0.

        Returns:
            Matplotlib figure object or None.
        """
        import matplotlib.pyplot as plt

        # Gets origin vertex
        origin = self.graph.get_start_vertex()

        # Extracts mean discovery numbers for all vertices EXCEPT origin
        means = [stats.mean for v, stats in self.summary_stats.items() if v != origin]

        if not means:
            return None

        # Calculates expected value
        # Mathematics: Non-origin vertices occupy positions 1, 2, ..., n-1
        # Expected position = (1 + 2 + ... + (n-1)) / (n-1) = n/2
        # Differs from (n-1)/2 which represents the average across ALL vertices including origin
        n = self.graph.number_vertices()
        expected = n / 2

        # Creates figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Creates histogram with automatic binning
        # Uses integer bins centered on whole numbers
        min_mean = int(np.floor(min(means)))
        max_mean = int(np.ceil(max(means)))
        bins = np.arange(min_mean - 0.5, max_mean + 1.5, 1.0)

        counts, edges, patches = ax.hist(
            means,
            bins=bins,
            color='steelblue',
            edgecolor='black',
            alpha=0.7,
            label='Vertex count'
        )

        # Adds vertical line for expected value (n/2 since origin excluded)
        ax.axvline(
            expected,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Expected n/2 = {expected:.2f}'
        )

        # Adds vertical line for actual mean
        actual_mean = np.mean(means)
        ax.axvline(
            actual_mean,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Actual mean = {actual_mean:.2f}'
        )

        # Labels and title
        ax.set_xlabel('Average Discovery Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Vertices', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Distribution of Vertex Discovery Numbers (Origin Excluded)\n{self.graph.desc()}, {self.config.num_samples} samples',
            fontsize=14,
            fontweight='bold'
        )

        # Grid for easier reading
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Legend
        ax.legend(fontsize=10)

        # Adds text box with statistics
        stats_text = f'Vertices plotted: {len(means)} (of {n})\n'
        stats_text += f'Min: {min(means):.2f}\n'
        stats_text += f'Max: {max(means):.2f}\n'
        stats_text += f'Std Dev: {np.std(means):.2f}\n'
        stats_text += f'Range: {max(means) - min(means):.2f}'

        ax.text(
            0.98, 0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9,
            family='monospace'
        )

        plt.tight_layout()
        return fig

    def _create_layer_analysis_plot(self):
        """
        Creates line graph showing average discovery number by distance layer.

        Uses BFS to determine distance from starting vertex, groups vertices
        by distance (layer), and plots average discovery number for each layer.

        Returns:
            Matplotlib figure object or None.
        """
        import matplotlib.pyplot as plt

        # Computes distance layers using BFS
        start_vertex = self.graph.get_start_vertex()
        distances = self._compute_bfs_distances(start_vertex)

        if not distances:
            return None

        # Groups vertices by layer and compute average discovery number per layer
        # Exclude origin (distance 0) for clearer visualization
        layers = {}
        for vertex, dist in distances.items():
            if dist > 0 and vertex in self.summary_stats:  # Exclude origin
                if dist not in layers:
                    layers[dist] = []
                layers[dist].append(self.summary_stats[vertex].mean)

        if not layers:
            return None

        # Computes statistics for each layer
        layer_numbers = sorted(layers.keys())
        layer_means = [np.mean(layers[layer]) for layer in layer_numbers]
        layer_stds = [np.std(layers[layer]) for layer in layer_numbers]
        layer_mins = [np.min(layers[layer]) for layer in layer_numbers]
        layer_maxs = [np.max(layers[layer]) for layer in layer_numbers]
        layer_sizes = [len(layers[layer]) for layer in layer_numbers]

        # Calculates expected value (excluding origin at position 0)
        # Expected position for remaining n-1 vertices: n/2
        n = self.graph.number_vertices()
        expected = n / 2

        # Creates figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Top plot: Average discovery number by layer
        ax1.plot(layer_numbers, layer_means, marker='o', linewidth=2,
                markersize=8, color='steelblue', label='Layer average', zorder=3)

        # Adds shaded range (min to max)
        ax1.fill_between(layer_numbers, layer_mins, layer_maxs,
                        alpha=0.2, color='steelblue', label='Min-Max range')

        # Adds error bars (standard deviation)
        ax1.errorbar(layer_numbers, layer_means, yerr=layer_stds,
                    fmt='none', ecolor='gray', alpha=0.5, capsize=5, zorder=2)

        # Adds expected line
        ax1.axhline(expected, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (n-1)/2 = {expected:.2f}')

        # Labels and title
        ax1.set_xlabel('Distance from Origin (Layer)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Discovery Number', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'Discovery Number by Distance Layer\n{self.graph.desc()}, {self.config.num_samples} samples',
            fontsize=14,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)

        # Adds text annotations for each layer
        for layer, mean, size in zip(layer_numbers, layer_means, layer_sizes):
            ax1.annotate(f'{mean:.1f}',
                        xy=(layer, mean),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color='darkblue')

        # Bottom plot: Number of vertices in each layer
        ax2.bar(layer_numbers, layer_sizes, color='lightcoral',
               edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Distance from Origin (Layer)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vertex Count', fontsize=12, fontweight='bold')
        ax2.set_title('Layer Sizes', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Adds count labels on bars
        for layer, size in zip(layer_numbers, layer_sizes):
            ax2.text(layer, size, str(size),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Adds statistics box
        stats_text = f'Total layers: {len(layer_numbers)}\n'
        stats_text += f'Max distance: {max(layer_numbers)}\n'
        stats_text += f'Graph diameter: {max(layer_numbers)}\n'

        # Calculates slope (change per layer)
        if len(layer_means) > 1:
            slopes = [layer_means[i+1] - layer_means[i]
                     for i in range(len(layer_means)-1)]
            avg_slope = np.mean(slopes)
            stats_text += f'Avg slope: {avg_slope:.2f}/layer'

        ax1.text(
            0.98, 0.02,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9,
            family='monospace'
        )

        plt.tight_layout()
        return fig

    def _compute_bfs_distances(self, start_vertex):
        """
        Computes BFS distances from start vertex to all other vertices.

        Args:
            start_vertex: The starting vertex for BFS.

        Returns:
            Dictionary mapping vertices to their BFS distance from start vertex.
        """
        from collections import deque

        distances = {}
        queue = deque([(start_vertex, 0)])
        visited = {start_vertex}

        while queue:
            vertex, dist = queue.popleft()
            distances[vertex] = dist

            for neighbor in self.graph.get_adj_list(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return distances

    def _compute_l1_distance(self, vertex1, vertex2):
        """
        Computes L1 (Manhattan) distance between two vertices.

        For tuples (e.g., hypercube): sum of absolute coordinate differences.
        For graphs with get_coordinates(): uses 2D embedding.
        For other types: returns None (not applicable).

        Args:
            vertex1: First vertex
            vertex2: Second vertex

        Returns:
            L1 distance as float, or None if not applicable.
        """
        # Tries using get_coordinates() if available (e.g., Petersen graphs)
        if hasattr(self.graph, 'get_coordinates'):
            try:
                x1, y1 = self.graph.get_coordinates(vertex1)
                x2, y2 = self.graph.get_coordinates(vertex2)
                return abs(x1 - x2) + abs(y1 - y2)
            except (TypeError, ValueError, AttributeError):
                pass

        # Fall back to direct coordinate computation (e.g., hypercubes)
        if isinstance(vertex1, tuple) and isinstance(vertex2, tuple):
            if len(vertex1) != len(vertex2):
                return None
            # Checks if all elements are numeric
            try:
                return sum(abs(v1 - v2) for v1, v2 in zip(vertex1, vertex2))
            except (TypeError, ValueError):
                # Non-numeric elements
                return None
        return None

    def _compute_l2_distance(self, vertex1, vertex2):
        """
        Computes L2 (Euclidean) distance between two vertices.

        For tuples (e.g., hypercube): sqrt of sum of squared coordinate differences.
        For graphs with get_coordinates(): uses 2D embedding.
        For other types: returns None (not applicable).

        Args:
            vertex1: First vertex
            vertex2: Second vertex

        Returns:
            L2 distance as float, or None if not applicable.
        """
        # Tries using get_coordinates() if available (e.g., Petersen graphs)
        if hasattr(self.graph, 'get_coordinates'):
            try:
                x1, y1 = self.graph.get_coordinates(vertex1)
                x2, y2 = self.graph.get_coordinates(vertex2)
                return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            except (TypeError, ValueError, AttributeError):
                pass

        # Fall back to direct coordinate computation (e.g., hypercubes)
        if isinstance(vertex1, tuple) and isinstance(vertex2, tuple):
            if len(vertex1) != len(vertex2):
                return None
            # Checks if all elements are numeric
            try:
                return np.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vertex1, vertex2)))
            except (TypeError, ValueError):
                # Non-numeric elements
                return None
        return None

    def _create_l1_layer_analysis_plot(self):
        """
        Creates line graph showing average discovery number by L1 (Manhattan) distance.

        Groups vertices by L1 distance from origin and plots average discovery
        number for each distance value.

        Returns:
            Matplotlib figure object or None if L1 distance not applicable.
        """
        import matplotlib.pyplot as plt

        start_vertex = self.graph.get_start_vertex()

        # Computes L1 distances for all vertices
        l1_distances = {}
        for vertex in self.summary_stats.keys():
            l1_dist = self._compute_l1_distance(start_vertex, vertex)
            if l1_dist is None:
                return None  # L1 distance not applicable for this graph type
            l1_distances[vertex] = l1_dist

        # Groups vertices by L1 distance, excluding origin (distance 0)
        layers = {}
        for vertex, dist in l1_distances.items():
            if dist > 0 and vertex in self.summary_stats:  # Exclude origin
                if dist not in layers:
                    layers[dist] = []
                layers[dist].append(self.summary_stats[vertex].mean)

        if not layers:
            return None

        # Computes statistics for each layer
        layer_numbers = sorted(layers.keys())
        layer_means = [np.mean(layers[layer]) for layer in layer_numbers]
        layer_stds = [np.std(layers[layer]) for layer in layer_numbers]
        layer_mins = [np.min(layers[layer]) for layer in layer_numbers]
        layer_maxs = [np.max(layers[layer]) for layer in layer_numbers]
        layer_sizes = [len(layers[layer]) for layer in layer_numbers]

        # Calculates expected value (excluding origin at position 0)
        # Expected position for remaining n-1 vertices: n/2
        n = self.graph.number_vertices()
        expected = n / 2

        # Creates figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Top plot: Average discovery number by L1 distance
        ax1.plot(layer_numbers, layer_means, marker='s', linewidth=2,
                markersize=8, color='forestgreen', label='L1 layer average', zorder=3)

        # Adds shaded range (min to max)
        ax1.fill_between(layer_numbers, layer_mins, layer_maxs,
                        alpha=0.2, color='forestgreen', label='Min-Max range')

        # Adds error bars
        ax1.errorbar(layer_numbers, layer_means, yerr=layer_stds,
                    fmt='none', ecolor='gray', alpha=0.5, capsize=5, zorder=2)

        # Adds expected line
        ax1.axhline(expected, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (n-1)/2 = {expected:.2f}')

        # Labels and title
        ax1.set_xlabel('L1 Distance (Manhattan) from Origin', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Discovery Number', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'Discovery Number by L1 Distance\n{self.graph.desc()}, {self.config.num_samples} samples',
            fontsize=14,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)

        # Adds text annotations
        for layer, mean in zip(layer_numbers, layer_means):
            ax1.annotate(f'{mean:.1f}',
                        xy=(layer, mean),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color='darkgreen')

        # Bottom plot: Number of vertices at each L1 distance
        ax2.bar(layer_numbers, layer_sizes, color='lightgreen',
               edgecolor='black', alpha=0.7)
        ax2.set_xlabel('L1 Distance from Origin', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vertex Count', fontsize=12, fontweight='bold')
        ax2.set_title('L1 Distance Distribution', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Adds count labels on bars
        for layer, size in zip(layer_numbers, layer_sizes):
            ax2.text(layer, size, str(size),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Adds statistics box
        stats_text = f'Max L1 distance: {max(layer_numbers)}\n'
        stats_text += f'Distance metric: Manhattan\n'

        if len(layer_means) > 1:
            slopes = [layer_means[i+1] - layer_means[i]
                     for i in range(len(layer_means)-1)]
            avg_slope = np.mean(slopes)
            stats_text += f'Avg slope: {avg_slope:.2f}/unit'

        ax1.text(
            0.98, 0.02,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontsize=9,
            family='monospace'
        )

        plt.tight_layout()
        return fig

    def _create_l2_layer_analysis_plot(self):
        """
        Creates line graph showing average discovery number by L2 (Euclidean) distance.

        Groups vertices by L2 distance from origin and plots average discovery
        number for each distance value.

        Returns:
            Matplotlib figure object or None if L2 distance not applicable.
        """
        import matplotlib.pyplot as plt

        start_vertex = self.graph.get_start_vertex()

        # Computes L2 distances for all vertices
        l2_distances = {}
        for vertex in self.summary_stats.keys():
            l2_dist = self._compute_l2_distance(start_vertex, vertex)
            if l2_dist is None:
                return None  # L2 distance not applicable for this graph type
            l2_distances[vertex] = l2_dist

        # Groups vertices by L2 distance (rounded to 2 decimals to handle float precision)
        # Exclude origin (distance 0)
        layers = {}
        for vertex, dist in l2_distances.items():
            if vertex in self.summary_stats:
                dist_rounded = round(dist, 2)
                if dist_rounded > 0 and dist_rounded not in layers:  # Exclude origin
                    layers[dist_rounded] = []
                if dist_rounded > 0:  # Exclude origin
                    layers[dist_rounded].append(self.summary_stats[vertex].mean)

        if not layers:
            return None

        # Computes statistics for each layer
        layer_numbers = sorted(layers.keys())
        layer_means = [np.mean(layers[layer]) for layer in layer_numbers]
        layer_stds = [np.std(layers[layer]) for layer in layer_numbers]
        layer_mins = [np.min(layers[layer]) for layer in layer_numbers]
        layer_maxs = [np.max(layers[layer]) for layer in layer_numbers]
        layer_sizes = [len(layers[layer]) for layer in layer_numbers]

        # Calculates expected value (excluding origin at position 0)
        # Expected position for remaining n-1 vertices: n/2
        n = self.graph.number_vertices()
        expected = n / 2

        # Creates figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Top plot: Average discovery number by L2 distance
        ax1.plot(layer_numbers, layer_means, marker='D', linewidth=2,
                markersize=8, color='darkorange', label='L2 layer average', zorder=3)

        # Adds shaded range (min to max)
        ax1.fill_between(layer_numbers, layer_mins, layer_maxs,
                        alpha=0.2, color='darkorange', label='Min-Max range')

        # Adds error bars (standard deviation)
        ax1.errorbar(layer_numbers, layer_means, yerr=layer_stds,
                    fmt='none', ecolor='gray', alpha=0.5, capsize=5, zorder=2)

        # Adds expected line
        ax1.axhline(expected, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (n-1)/2 = {expected:.2f}')

        # Labels and title
        ax1.set_xlabel('L2 Distance (Euclidean) from Origin', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Discovery Number', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'Discovery Number by L2 Distance\n{self.graph.desc()}, {self.config.num_samples} samples',
            fontsize=14,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)

        # Adds text annotations (only for some points to avoid clutter)
        step = max(1, len(layer_numbers) // 10)  # Shows ~10 labels
        for i, (layer, mean) in enumerate(zip(layer_numbers, layer_means)):
            if i % step == 0 or i == len(layer_numbers) - 1:
                ax1.annotate(f'{mean:.1f}',
                            xy=(layer, mean),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            fontsize=8,
                            color='darkorange')

        # Bottom plot: Number of vertices at each L2 distance
        ax2.bar(layer_numbers, layer_sizes, color='peachpuff',
               edgecolor='black', alpha=0.7, width=0.05)
        ax2.set_xlabel('L2 Distance from Origin', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vertex Count', fontsize=12, fontweight='bold')
        ax2.set_title('L2 Distance Distribution', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Adds count labels on bars (only for larger bars to avoid clutter)
        for layer, size in zip(layer_numbers, layer_sizes):
            if size > max(layer_sizes) * 0.1:  # Only label if > 10% of max
                ax2.text(layer, size, str(size),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Adds statistics box
        stats_text = f'Max L2 distance: {max(layer_numbers):.2f}\n'
        stats_text += f'Distance metric: Euclidean\n'
        stats_text += f'Unique distances: {len(layer_numbers)}\n'

        if len(layer_means) > 1:
            slopes = [layer_means[i+1] - layer_means[i]
                     for i in range(len(layer_means)-1)]
            avg_slope = np.mean(slopes)
            stats_text += f'Avg slope: {avg_slope:.2f}/unit'

        ax1.text(
            0.98, 0.02,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='peachpuff', alpha=0.5),
            fontsize=9,
            family='monospace'
        )

        plt.tight_layout()
        return fig

    def _create_layer_statistics_table(self):
        """
        Creates CSV table with detailed statistics for each BFS layer.

        Returns:
            None (saves CSV file to output directory)
        """
        start_vertex = self.graph.get_start_vertex()

        # Computes BFS distances for all vertices
        distances = self._compute_bfs_distances(start_vertex)
        if not distances:
            return

        # Groups vertices by distance (layer)
        layers = {}
        for vertex, dist in distances.items():
            if vertex in self.summary_stats:
                if dist not in layers:
                    layers[dist] = []
                layers[dist].append(self.summary_stats[vertex].mean)

        if not layers:
            return

        # Computes statistics for each layer
        layer_numbers = sorted(layers.keys())

        # Prepare table data
        table_data = []
        prev_mean = None

        for layer in layer_numbers:
            values = layers[layer]
            mean = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            std_dev = np.std(values)
            range_val = max_val - min_val
            count = len(values)

            # Calculates delta from previous layer
            delta = mean - prev_mean if prev_mean is not None else 0.0
            prev_mean = mean

            table_data.append({
                'Layer': layer,
                'Mean': f'{mean:.4f}',
                'Min': f'{min_val:.4f}',
                'Max': f'{max_val:.4f}',
                'Range': f'{range_val:.4f}',
                'StdDev': f'{std_dev:.4f}',
                'Count': count,
                'Delta_Mean': f'{delta:.4f}'
            })

        # Saves to CSV
        filepath = os.path.join(self.output_path, "layer_statistics_bfs.csv")
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['Layer', 'Mean', 'Min', 'Max', 'Range', 'StdDev', 'Count', 'Delta_Mean']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_data)

    def _create_l1_layer_statistics_table(self):
        """
        Creates CSV table with detailed statistics for each L1 (Manhattan) distance layer.

        Returns:
            None (saves CSV file to output directory)
        """
        start_vertex = self.graph.get_start_vertex()

        # Computes L1 distances for all vertices
        l1_distances = {}
        for vertex in self.summary_stats.keys():
            l1_dist = self._compute_l1_distance(start_vertex, vertex)
            if l1_dist is None:
                return  # L1 distance not applicable for this graph type
            l1_distances[vertex] = l1_dist

        # Groups vertices by L1 distance, excluding origin (distance 0)
        layers = {}
        for vertex, dist in l1_distances.items():
            if dist > 0 and vertex in self.summary_stats:  # Exclude origin
                if dist not in layers:
                    layers[dist] = []
                layers[dist].append(self.summary_stats[vertex].mean)

        if not layers:
            return

        # Computes statistics for each layer
        layer_numbers = sorted(layers.keys())

        # Prepare table data
        table_data = []
        prev_mean = None

        for layer in layer_numbers:
            values = layers[layer]
            mean = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            std_dev = np.std(values)
            range_val = max_val - min_val
            count = len(values)

            # Calculates delta from previous layer
            delta = mean - prev_mean if prev_mean is not None else 0.0
            prev_mean = mean

            table_data.append({
                'Layer': layer,
                'Mean': f'{mean:.4f}',
                'Min': f'{min_val:.4f}',
                'Max': f'{max_val:.4f}',
                'Range': f'{range_val:.4f}',
                'StdDev': f'{std_dev:.4f}',
                'Count': count,
                'Delta_Mean': f'{delta:.4f}'
            })

        # Saves to CSV
        filepath = os.path.join(self.output_path, "layer_statistics_l1.csv")
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['Layer', 'Mean', 'Min', 'Max', 'Range', 'StdDev', 'Count', 'Delta_Mean']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_data)

    def _create_l2_layer_statistics_table(self):
        """
        Creates CSV table with detailed statistics for each L2 (Euclidean) distance layer.

        Returns:
            None (saves CSV file to output directory)
        """
        start_vertex = self.graph.get_start_vertex()

        # Computes L2 distances for all vertices
        l2_distances = {}
        for vertex in self.summary_stats.keys():
            l2_dist = self._compute_l2_distance(start_vertex, vertex)
            if l2_dist is None:
                return  # L2 distance not applicable for this graph type
            l2_distances[vertex] = l2_dist

        # Groups vertices by L2 distance (rounded to 2 decimals)
        layers = {}
        for vertex, dist in l2_distances.items():
            if vertex in self.summary_stats:
                dist_rounded = round(dist, 2)
                if dist_rounded not in layers:
                    layers[dist_rounded] = []
                layers[dist_rounded].append(self.summary_stats[vertex].mean)

        if not layers:
            return

        # Computes statistics for each layer
        layer_numbers = sorted(layers.keys())

        # Prepare table data
        table_data = []
        prev_mean = None

        for layer in layer_numbers:
            values = layers[layer]
            mean = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            std_dev = np.std(values)
            range_val = max_val - min_val
            count = len(values)

            # Calculates delta from previous layer
            delta = mean - prev_mean if prev_mean is not None else 0.0
            prev_mean = mean

            table_data.append({
                'Layer': f'{layer:.2f}',  # Formats with 2 decimals for L2 distances
                'Mean': f'{mean:.4f}',
                'Min': f'{min_val:.4f}',
                'Max': f'{max_val:.4f}',
                'Range': f'{range_val:.4f}',
                'StdDev': f'{std_dev:.4f}',
                'Count': count,
                'Delta_Mean': f'{delta:.4f}'
            })

        # Saves to CSV
        filepath = os.path.join(self.output_path, "layer_statistics_l2.csv")
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['Layer', 'Mean', 'Min', 'Max', 'Range', 'StdDev', 'Count', 'Delta_Mean']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_data)

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
