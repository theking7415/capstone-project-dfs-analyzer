"""
Test the theory: In a random walk, immediate neighbors of the starting node
have on average an n/π DFS discovery number, where n is the number of nodes.

This module implements a 2D grid graph and analyzes RDFS discovery numbers
for immediate neighbors of the starting vertex.
"""

import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from scipy import stats
import csv

from mygraphs import Graph
from myrdfs import collect_statistics, get_summary_stats, RNG


GridVertexType = tuple[int, int]


class Grid2D(Graph[GridVertexType]):
    """A 2D grid graph with width x height vertices."""

    def __init__(self, width: int, height: int):
        """Creates a 2D grid of dimensions width x height.

        Vertices are represented as (x, y) tuples where:
        - 0 <= x < width
        - 0 <= y < height
        """
        if width < 1 or height < 1:
            raise ValueError("Width and height must be positive integers.")
        self.width = width
        self.height = height

    def get_start_vertex(self) -> GridVertexType:
        """Start at the center of the grid for symmetry."""
        return (self.width // 2, self.height // 2)

    def get_adj_list(self, v: GridVertexType) -> list[GridVertexType]:
        """Returns neighbors in 4-connectivity (up, down, left, right)."""
        x, y = v
        neighbors = []

        # Check all four directions
        directions = [
            (x - 1, y),  # left
            (x + 1, y),  # right
            (x, y - 1),  # down
            (x, y + 1),  # up
        ]

        for nx, ny in directions:
            # Check if neighbor is within grid bounds
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))

        return neighbors

    def desc(self) -> str:
        return f"grid2d-{self.width}x{self.height}"

    def number_vertices(self) -> int:
        return self.width * self.height

    def get_immediate_neighbors(self) -> list[GridVertexType]:
        """Returns the immediate neighbors of the starting vertex."""
        start = self.get_start_vertex()
        return self.get_adj_list(start)

    def plot_means_vars(self, summary_stats: dict[GridVertexType, Any], *, fname=None):
        """Visualize mean DFS numbers as a heatmap on the grid."""
        # Create a 2D array to hold the mean values
        heatmap = np.zeros((self.height, self.width))

        for (x, y), stat in summary_stats.items():
            heatmap[y, x] = stat.mean

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap, cmap='viridis', origin='lower')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean DFS Discovery Number')

        # Mark the starting vertex
        start_x, start_y = self.get_start_vertex()
        ax.plot(start_x, start_y, 'r*', markersize=20, label='Start Vertex')

        # Mark immediate neighbors
        for nx, ny in self.get_immediate_neighbors():
            ax.plot(nx, ny, 'ro', markersize=12, label='Immediate Neighbor' if nx == self.get_immediate_neighbors()[0][0] else '')

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Mean DFS Discovery Numbers - {self.width}x{self.height} Grid')
        ax.legend()

        if fname:
            plt.savefig(fname)
        plt.tight_layout()
        plt.show()


def analyze_immediate_neighbors(grid: Grid2D, num_samples: int, *, rng=RNG):
    """
    Analyze the discovery numbers of immediate neighbors vs the n/π theory.

    Args:
        grid: The Grid2D graph to analyze
        num_samples: Number of RDFS samples to run
        rng: Random number generator

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Testing n/π theory on {grid.width}x{grid.height} grid")
    print(f"{'='*60}")

    n = grid.number_vertices()
    theoretical_value = n / np.pi

    print(f"Number of vertices (n): {n}")
    print(f"Theoretical value (n/π): {theoretical_value:.4f}")
    print(f"Running {num_samples} RDFS samples...")

    # Collect statistics
    dist_stats = collect_statistics(grid, num_samples, rng=rng)
    summary_stats = get_summary_stats(dist_stats)

    # Get immediate neighbors
    immediate_neighbors = grid.get_immediate_neighbors()
    start_vertex = grid.get_start_vertex()

    print(f"\nStarting vertex: {start_vertex}")
    print(f"Immediate neighbors: {immediate_neighbors}")
    print(f"Number of immediate neighbors: {len(immediate_neighbors)}")

    # Analyze discovery numbers for immediate neighbors
    neighbor_means = []
    neighbor_vars = []

    print(f"\n{'Neighbor':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8}")
    print("-" * 60)

    for neighbor in immediate_neighbors:
        stat = summary_stats[neighbor]
        neighbor_means.append(stat.mean)
        neighbor_vars.append(stat.variance)

        print(f"{str(neighbor):<15} {stat.mean:<12.4f} {np.sqrt(stat.variance):<12.4f} "
              f"{stat.minmax[0]:<8} {stat.minmax[1]:<8}")

    # Calculate average discovery number across all immediate neighbors
    avg_neighbor_discovery = np.mean(neighbor_means)
    std_neighbor_discovery = np.std(neighbor_means)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Average discovery number of immediate neighbors: {avg_neighbor_discovery:.4f}")
    print(f"Standard deviation across neighbors: {std_neighbor_discovery:.4f}")
    print(f"Theoretical prediction (n/π): {theoretical_value:.4f}")
    print(f"Difference: {abs(avg_neighbor_discovery - theoretical_value):.4f}")
    print(f"Relative error: {abs(avg_neighbor_discovery - theoretical_value) / theoretical_value * 100:.2f}%")
    print(f"{'='*60}\n")

    # Store results
    results = {
        'grid': grid,
        'n': n,
        'theoretical_n_over_pi': theoretical_value,
        'num_samples': num_samples,
        'immediate_neighbors': immediate_neighbors,
        'neighbor_means': neighbor_means,
        'neighbor_vars': neighbor_vars,
        'avg_neighbor_discovery': avg_neighbor_discovery,
        'std_neighbor_discovery': std_neighbor_discovery,
        'relative_error': abs(avg_neighbor_discovery - theoretical_value) / theoretical_value,
        'summary_stats': summary_stats
    }

    return results


def plot_neighbor_analysis(results: dict):
    """Create visualization of immediate neighbor discovery numbers."""
    grid = results['grid']
    immediate_neighbors = results['immediate_neighbors']
    neighbor_means = results['neighbor_means']
    neighbor_vars = results['neighbor_vars']
    theoretical_value = results['theoretical_n_over_pi']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Bar chart of neighbor discovery numbers
    neighbor_labels = [str(n) for n in immediate_neighbors]
    x_pos = np.arange(len(immediate_neighbors))

    ax1.bar(x_pos, neighbor_means, yerr=np.sqrt(neighbor_vars),
            capsize=5, color='skyblue', ecolor='gray', alpha=0.7)
    ax1.axhline(y=theoretical_value, color='r', linestyle='--',
                linewidth=2, label=f'Theoretical n/π = {theoretical_value:.2f}')
    ax1.axhline(y=results['avg_neighbor_discovery'], color='g', linestyle='-',
                linewidth=2, label=f'Observed avg = {results["avg_neighbor_discovery"]:.2f}')

    ax1.set_xlabel('Immediate Neighbor')
    ax1.set_ylabel('Mean Discovery Number')
    ax1.set_title(f'Discovery Numbers of Immediate Neighbors\n{grid.width}x{grid.height} Grid')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(neighbor_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heatmap visualization
    heatmap = np.full((grid.height, grid.width), np.nan)
    summary_stats = results['summary_stats']

    for (x, y), stat in summary_stats.items():
        heatmap[y, x] = stat.mean

    im = ax2.imshow(heatmap, cmap='viridis', origin='lower')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Mean Discovery Number')

    # Mark special vertices
    start_x, start_y = grid.get_start_vertex()
    ax2.plot(start_x, start_y, 'r*', markersize=20, label='Start', markeredgecolor='white', markeredgewidth=2)

    for i, (nx, ny) in enumerate(immediate_neighbors):
        ax2.plot(nx, ny, 'ro', markersize=15, markeredgecolor='white',
                markeredgewidth=2, label='Immediate Neighbor' if i == 0 else '')

    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_title(f'DFS Discovery Number Heatmap\n{grid.width}x{grid.height} Grid')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_multiple_grid_experiments(grid_sizes: list[int], num_samples: int):
    """
    Run experiments on multiple grid sizes with constant sample count.

    Args:
        grid_sizes: List of grid dimensions (for square grids)
        num_samples: Number of RDFS samples to run for each grid

    Returns:
        List of result dictionaries
    """
    all_results = []

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENTS ON MULTIPLE GRID SIZES")
    print(f"Samples per grid: {num_samples}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"{'='*80}\n")

    for size in grid_sizes:
        grid = Grid2D(size, size)
        results = analyze_immediate_neighbors(grid, num_samples, rng=RNG)
        all_results.append(results)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE - CONSTANT SAMPLES ({num_samples} samples per grid)")
    print(f"{'='*80}")
    print(f"{'Grid':<10} {'Vertices':<10} {'n/π':<12} {'Observed':<12} {'Diff':<12} {'Error %':<10}")
    print(f"{'-'*80}")

    for res in all_results:
        grid_desc = f"{res['grid'].width}x{res['grid'].height}"
        n = res['n']
        theoretical = res['theoretical_n_over_pi']
        observed = res['avg_neighbor_discovery']
        diff = observed - theoretical
        error_pct = res['relative_error'] * 100

        print(f"{grid_desc:<10} {n:<10} {theoretical:<12.4f} {observed:<12.4f} "
              f"{diff:<12.4f} {error_pct:<10.2f}")

    print(f"{'='*80}\n")

    # Save results to CSV file (Excel compatible)
    csv_filename = f"results/immediate_neighbors_grid_experiment_{num_samples}_samples.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Grid Size', 'Vertices (n)', 'Theoretical (n/π)',
                           'Observed Mean', 'Difference', 'Relative Error (%)',
                           'Std Dev Across Neighbors'])

            # Write data rows
            for res in all_results:
                grid_desc = f"{res['grid'].width}x{res['grid'].height}"
                n = res['n']
                theoretical = res['theoretical_n_over_pi']
                observed = res['avg_neighbor_discovery']
                diff = observed - theoretical
                error_pct = res['relative_error'] * 100
                std_dev = res['std_neighbor_discovery']

                writer.writerow([grid_desc, n, theoretical, observed, diff, error_pct, std_dev])

        print(f"Results saved to: {csv_filename}\n")
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}\n")

    return all_results


def analyze_hypercube_immediate_neighbors(dimensions: list[int], num_samples: int):
    """
    Test the n/π theory on hypercube immediate neighbors.

    Args:
        dimensions: List of hypercube dimensions to test
        num_samples: Number of RDFS samples per hypercube

    Returns:
        List of result dictionaries
    """
    from mygraphs import Hypercube

    all_results = []

    print(f"\n{'='*80}")
    print(f"TESTING n/π THEORY ON HYPERCUBE IMMEDIATE NEIGHBORS")
    print(f"Samples per hypercube: {num_samples}")
    print(f"Dimensions to test: {dimensions}")
    print(f"{'='*80}\n")

    for d in dimensions:
        hypercube = Hypercube(d)
        n = hypercube.number_vertices()
        theoretical_value = n / np.pi

        print(f"\n{'='*60}")
        print(f"Testing {d}D Hypercube")
        print(f"{'='*60}")
        print(f"Number of vertices (n): {n}")
        print(f"Theoretical value (n/π): {theoretical_value:.4f}")
        print(f"Running {num_samples} RDFS samples...")

        # Collect statistics
        dist_stats = collect_statistics(hypercube, num_samples, rng=RNG)
        summary_stats = get_summary_stats(dist_stats)

        # Get immediate neighbors of starting vertex
        start_vertex = hypercube.get_start_vertex()
        immediate_neighbors = hypercube.get_adj_list(start_vertex)

        print(f"\nStarting vertex: {start_vertex}")
        print(f"Immediate neighbors: {immediate_neighbors}")
        print(f"Number of immediate neighbors: {len(immediate_neighbors)}")

        # Analyze discovery numbers for immediate neighbors
        neighbor_means = []
        neighbor_vars = []

        print(f"\n{'Neighbor':<20} {'Mean':<12} {'Std Dev':<12}")
        print("-" * 50)

        for neighbor in immediate_neighbors:
            stat = summary_stats[neighbor]
            neighbor_means.append(stat.mean)
            neighbor_vars.append(stat.variance)
            print(f"{str(neighbor):<20} {stat.mean:<12.4f} {np.sqrt(stat.variance):<12.4f}")

        # Calculate average discovery number across all immediate neighbors
        avg_neighbor_discovery = np.mean(neighbor_means)
        std_neighbor_discovery = np.std(neighbor_means)

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"Average discovery number of immediate neighbors: {avg_neighbor_discovery:.4f}")
        print(f"Standard deviation across neighbors: {std_neighbor_discovery:.4f}")
        print(f"Theoretical prediction (n/π): {theoretical_value:.4f}")
        print(f"Difference: {abs(avg_neighbor_discovery - theoretical_value):.4f}")
        print(f"Relative error: {abs(avg_neighbor_discovery - theoretical_value) / theoretical_value * 100:.2f}%")
        print(f"{'='*60}\n")

        # Store results
        results = {
            'dimension': d,
            'graph_desc': f"{d}D Hypercube",
            'n': n,
            'theoretical_n_over_pi': theoretical_value,
            'num_samples': num_samples,
            'immediate_neighbors': immediate_neighbors,
            'neighbor_means': neighbor_means,
            'neighbor_vars': neighbor_vars,
            'avg_neighbor_discovery': avg_neighbor_discovery,
            'std_neighbor_discovery': std_neighbor_discovery,
            'relative_error': abs(avg_neighbor_discovery - theoretical_value) / theoretical_value,
            'summary_stats': summary_stats
        }

        all_results.append(results)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE - n/π THEORY ON HYPERCUBE IMMEDIATE NEIGHBORS")
    print(f"{'='*90}")
    print(f"{'Dimension':<12} {'Vertices':<10} {'n/π':<12} {'Observed':<12} {'Diff':<12} {'Error %':<10}")
    print(f"{'-'*90}")

    for res in all_results:
        dim = res['dimension']
        n = res['n']
        theoretical = res['theoretical_n_over_pi']
        observed = res['avg_neighbor_discovery']
        diff = observed - theoretical
        error_pct = res['relative_error'] * 100

        print(f"{dim}D{' ':<10} {n:<10} {theoretical:<12.4f} {observed:<12.4f} "
              f"{diff:<12.4f} {error_pct:<10.2f}")

    print(f"{'='*90}\n")

    # Save results to CSV
    csv_filename = f"results/immediate_neighbors_hypercube_experiment_{num_samples}_samples.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Dimension', 'Vertices (n)', 'Theoretical (n/π)',
                           'Observed Mean', 'Difference', 'Relative Error (%)',
                           'Std Dev Across Neighbors', 'Num Neighbors', 'Samples'])

            for res in all_results:
                dim = res['dimension']
                n = res['n']
                theoretical = res['theoretical_n_over_pi']
                observed = res['avg_neighbor_discovery']
                diff = observed - theoretical
                error_pct = res['relative_error'] * 100
                std_dev = res['std_neighbor_discovery']
                num_neighbors = len(res['immediate_neighbors'])
                samples = res['num_samples']

                writer.writerow([dim, n, theoretical, observed, diff, error_pct,
                               std_dev, num_neighbors, samples])

        print(f"Results saved to: {csv_filename}\n")
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}\n")

    return all_results


if __name__ == "__main__":
    # Test n/π theory on hypercube immediate neighbors instead of grids
    hypercube_dimensions = [3, 4, 5, 6, 7, 8]
    num_samples = 10000

    all_results = analyze_hypercube_immediate_neighbors(hypercube_dimensions, num_samples)
