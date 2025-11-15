"""
Test the (n-1)/2 conjecture on Generalized Petersen Graphs.

A Generalized Petersen Graph GP(n, k) has:
- 2n vertices total
- Outer ring: n vertices connected sequentially
- Inner ring: n vertices with each v_i connected to v_{(i+k) mod n}
- Spokes: Each outer vertex u_i connected to inner vertex v_i

This module tests whether the average discovery number in RDFS tends to (n-1)/2.
"""

import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from scipy import stats
import csv
import networkx as nx

from mygraphs import Graph
from myrdfs import collect_statistics, get_summary_stats, RNG


PetersenVertexType = tuple[str, int]  # ('outer', i) or ('inner', i)


class GeneralizedPetersen(Graph[PetersenVertexType]):
    """A Generalized Petersen Graph GP(n, k)."""

    def __init__(self, n: int, k: int):
        """
        Creates a Generalized Petersen Graph GP(n, k).

        Args:
            n: Number of vertices in each ring (total vertices = 2n)
            k: Skip parameter for inner ring connections

        Vertices are represented as tuples:
        - ('outer', i) for outer ring vertices, i in [0, n-1]
        - ('inner', i) for inner ring vertices, i in [0, n-1]
        """
        if n < 3:
            raise ValueError("n must be at least 3")
        if k < 1 or k >= n:
            raise ValueError("k must be in range [1, n-1]")
        if k >= n / 2:
            print(f"Warning: k={k} >= n/2={n/2} may create multi-edges")

        self.n = n
        self.k = k

    def get_start_vertex(self) -> PetersenVertexType:
        """Start at the first outer vertex for consistency."""
        return ('outer', 0)

    def get_adj_list(self, v: PetersenVertexType) -> list[PetersenVertexType]:
        """Returns neighbors of vertex v."""
        ring, i = v
        neighbors = []

        if ring == 'outer':
            # Outer ring: connected to previous and next in ring, plus spoke to inner
            neighbors.append(('outer', (i - 1) % self.n))  # Previous in outer ring
            neighbors.append(('outer', (i + 1) % self.n))  # Next in outer ring
            neighbors.append(('inner', i))                  # Spoke to inner ring
        else:  # ring == 'inner'
            # Inner ring: connected to spoke and two inner ring neighbors
            neighbors.append(('outer', i))                      # Spoke to outer ring
            neighbors.append(('inner', (i - self.k) % self.n))  # Previous in inner ring
            neighbors.append(('inner', (i + self.k) % self.n))  # Next in inner ring

        return neighbors

    def desc(self) -> str:
        return f"gp-{self.n}-{self.k}"

    def number_vertices(self) -> int:
        return 2 * self.n

    def plot_means_vars(self, summary_stats: dict[PetersenVertexType, Any], *, fname=None):
        """Visualize mean DFS numbers for Petersen graph vertices."""
        # Separate outer and inner ring statistics
        outer_vertices = sorted([(ring, i) for ring, i in summary_stats.keys() if ring == 'outer'],
                               key=lambda x: x[1])
        inner_vertices = sorted([(ring, i) for ring, i in summary_stats.keys() if ring == 'inner'],
                               key=lambda x: x[1])

        outer_means = [summary_stats[v].mean for v in outer_vertices]
        outer_sds = [np.sqrt(summary_stats[v].variance) for v in outer_vertices]
        inner_means = [summary_stats[v].mean for v in inner_vertices]
        inner_sds = [np.sqrt(summary_stats[v].variance) for v in inner_vertices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot outer ring
        x_outer = list(range(self.n))
        ax1.bar(x_outer, outer_means, yerr=outer_sds, capsize=5, color='skyblue', ecolor='gray')
        ax1.set_ylabel('Mean DFS Number')
        ax1.set_xlabel('Vertex Index (Outer Ring)')
        ax1.set_title(f'Outer Ring - GP({self.n}, {self.k})')
        ax1.grid(True, alpha=0.3)

        # Plot inner ring
        x_inner = list(range(self.n))
        ax2.bar(x_inner, inner_means, yerr=inner_sds, capsize=5, color='lightcoral', ecolor='gray')
        ax2.set_ylabel('Mean DFS Number')
        ax2.set_xlabel('Vertex Index (Inner Ring)')
        ax2.set_title(f'Inner Ring - GP({self.n}, {self.k})')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Mean DFS Numbers - Generalized Petersen Graph GP({self.n}, {self.k})', fontsize=14)
        plt.tight_layout()

        if fname:
            plt.savefig(fname)
        plt.show()


def analyze_petersen_graph(n: int, k: int, num_samples: int, *, rng=RNG):
    """
    Analyze the (n-1)/2 conjecture on a Generalized Petersen Graph.

    Args:
        n: Number of vertices in each ring
        k: Skip parameter for inner ring
        num_samples: Number of RDFS samples to run
        rng: Random number generator

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*70}")
    print(f"Testing (n-1)/2 conjecture on Generalized Petersen Graph GP({n}, {k})")
    print(f"{'='*70}")

    graph = GeneralizedPetersen(n, k)
    total_vertices = graph.number_vertices()
    theoretical_value = (total_vertices - 1) / 2

    print(f"Graph: GP({n}, {k})")
    print(f"Number of vertices (n): {total_vertices}")
    print(f"Degree: 3 (cubic graph)")
    print(f"Theoretical value (n-1)/2: {theoretical_value:.4f}")
    print(f"Running {num_samples} RDFS samples...")

    # Collect statistics
    dist_stats = collect_statistics(graph, num_samples, rng=rng)
    summary_stats = get_summary_stats(dist_stats)

    # Calculate overall average discovery number
    all_means = [stat.mean for stat in summary_stats.values()]
    avg_discovery = np.mean(all_means)
    std_discovery = np.std(all_means)

    # Separate outer and inner ring statistics
    outer_means = [summary_stats[('outer', i)].mean for i in range(n)]
    inner_means = [summary_stats[('inner', i)].mean for i in range(n)]

    avg_outer = np.mean(outer_means)
    avg_inner = np.mean(inner_means)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Average discovery number (all vertices): {avg_discovery:.4f}")
    print(f"  - Outer ring average: {avg_outer:.4f}")
    print(f"  - Inner ring average: {avg_inner:.4f}")
    print(f"Standard deviation across all vertices: {std_discovery:.4f}")
    print(f"Theoretical prediction (n-1)/2: {theoretical_value:.4f}")
    print(f"Difference: {abs(avg_discovery - theoretical_value):.4f}")
    print(f"Relative error: {abs(avg_discovery - theoretical_value) / theoretical_value * 100:.2f}%")
    print(f"{'='*70}\n")

    # Store results
    results = {
        'graph_name': f"GP({n}, {k})",
        'n_param': n,
        'k_param': k,
        'total_vertices': total_vertices,
        'theoretical_n_minus_1_over_2': theoretical_value,
        'num_samples': num_samples,
        'avg_discovery': avg_discovery,
        'avg_outer': avg_outer,
        'avg_inner': avg_inner,
        'std_discovery': std_discovery,
        'relative_error': abs(avg_discovery - theoretical_value) / theoretical_value,
        'summary_stats': summary_stats,
        'graph': graph
    }

    return results


def run_petersen_experiments(petersen_configs: list[tuple[int, int]], num_samples: int):
    """
    Run experiments on multiple Generalized Petersen graphs.

    Args:
        petersen_configs: List of (n, k) tuples defining GP(n, k) graphs
        num_samples: Number of RDFS samples per graph

    Returns:
        List of result dictionaries
    """
    all_results = []

    print(f"\n{'='*80}")
    print(f"RUNNING (n-1)/2 CONJECTURE TESTS ON GENERALIZED PETERSEN GRAPHS")
    print(f"Samples per graph: {num_samples}")
    print(f"Graphs to test: {[f'GP({n}, {k})' for n, k in petersen_configs]}")
    print(f"{'='*80}\n")

    for n, k in petersen_configs:
        results = analyze_petersen_graph(n, k, num_samples, rng=RNG)
        all_results.append(results)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE - (n-1)/2 CONJECTURE ON GENERALIZED PETERSEN GRAPHS")
    print(f"{'='*90}")
    print(f"{'Graph':<12} {'Vertices':<10} {'(n-1)/2':<12} {'Observed':<12} {'Diff':<12} {'Error %':<10}")
    print(f"{'-'*90}")

    for res in all_results:
        graph_name = res['graph_name']
        vertices = res['total_vertices']
        theoretical = res['theoretical_n_minus_1_over_2']
        observed = res['avg_discovery']
        diff = observed - theoretical
        error_pct = res['relative_error'] * 100

        print(f"{graph_name:<12} {vertices:<10} {theoretical:<12.4f} {observed:<12.4f} "
              f"{diff:<12.4f} {error_pct:<10.2f}")

    print(f"{'='*90}\n")

    # Save results to CSV file
    csv_filename = f"results/generalised_petersen_experiment_{num_samples}_samples.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Graph', 'n (param)', 'k (param)', 'Total Vertices',
                           'Theoretical (n-1)/2', 'Observed Mean', 'Outer Ring Mean',
                           'Inner Ring Mean', 'Difference', 'Relative Error (%)',
                           'Std Dev Across Vertices', 'Samples'])

            # Write data rows
            for res in all_results:
                graph_name = res['graph_name']
                n_param = res['n_param']
                k_param = res['k_param']
                vertices = res['total_vertices']
                theoretical = res['theoretical_n_minus_1_over_2']
                observed = res['avg_discovery']
                avg_outer = res['avg_outer']
                avg_inner = res['avg_inner']
                diff = observed - theoretical
                error_pct = res['relative_error'] * 100
                std_dev = res['std_discovery']
                samples = res['num_samples']

                writer.writerow([graph_name, n_param, k_param, vertices, theoretical,
                               observed, avg_outer, avg_inner, diff, error_pct,
                               std_dev, samples])

        print(f"Results saved to: {csv_filename}\n")
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}\n")

    return all_results


def visualize_all_results(all_results):
    """Create comprehensive visualizations for all Petersen graph results."""
    n_graphs = len(all_results)

    # Create a figure with subplots for each graph
    fig = plt.figure(figsize=(18, 12))

    for idx, res in enumerate(all_results):
        graph = res['graph']
        summary_stats = res['summary_stats']
        n = graph.n
        k = graph.k

        # Separate outer and inner ring statistics
        outer_vertices = [('outer', i) for i in range(n)]
        inner_vertices = [('inner', i) for i in range(n)]

        outer_means = [summary_stats[v].mean for v in outer_vertices]
        inner_means = [summary_stats[v].mean for v in inner_vertices]
        outer_sds = [np.sqrt(summary_stats[v].variance) for v in outer_vertices]
        inner_sds = [np.sqrt(summary_stats[v].variance) for v in inner_vertices]

        theoretical = res['theoretical_n_minus_1_over_2']

        # Create two subplots per graph (outer and inner ring)
        ax1 = plt.subplot(n_graphs, 2, idx*2 + 1)
        ax2 = plt.subplot(n_graphs, 2, idx*2 + 2)

        # Plot outer ring
        x_outer = list(range(n))
        bars1 = ax1.bar(x_outer, outer_means, yerr=outer_sds, capsize=5,
                        color='skyblue', ecolor='gray', alpha=0.7)
        ax1.axhline(y=theoretical, color='red', linestyle='--',
                   linewidth=2, label=f'(n-1)/2 = {theoretical:.2f}')
        ax1.axhline(y=res['avg_outer'], color='blue', linestyle='-',
                   linewidth=2, alpha=0.7, label=f'Outer avg = {res["avg_outer"]:.2f}')

        # Mark the starting vertex (vertex 0 in outer ring)
        bars1[0].set_color('orange')
        bars1[0].set_alpha(1.0)

        ax1.set_ylabel('Mean Discovery Number', fontsize=10)
        ax1.set_xlabel('Vertex Index', fontsize=10)
        ax1.set_title(f'GP({n}, {k}) - Outer Ring\n(Orange = Start Vertex)', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x_outer)

        # Plot inner ring
        x_inner = list(range(n))
        ax2.bar(x_inner, inner_means, yerr=inner_sds, capsize=5,
               color='lightcoral', ecolor='gray', alpha=0.7)
        ax2.axhline(y=theoretical, color='red', linestyle='--',
                   linewidth=2, label=f'(n-1)/2 = {theoretical:.2f}')
        ax2.axhline(y=res['avg_inner'], color='darkred', linestyle='-',
                   linewidth=2, alpha=0.7, label=f'Inner avg = {res["avg_inner"]:.2f}')

        ax2.set_ylabel('Mean Discovery Number', fontsize=10)
        ax2.set_xlabel('Vertex Index', fontsize=10)
        ax2.set_title(f'GP({n}, {k}) - Inner Ring', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x_inner)

    fig.suptitle('Discovery Numbers in Generalized Petersen Graphs\nOuter vs Inner Ring Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save the figure
    plt.savefig('results/petersen_graphs_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: results/petersen_graphs_visualization.png")
    plt.close(fig)  # Close instead of show

    # Create a second figure showing the comparison summary
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, res in enumerate(all_results):
        ax = axes[idx]
        graph = res['graph']
        n = graph.n
        k = graph.k

        # Create comparison bars
        categories = ['Outer Ring\nAverage', 'Inner Ring\nAverage', 'Overall\nAverage', 'Theory\n(n-1)/2']
        values = [res['avg_outer'], res['avg_inner'], res['avg_discovery'], res['theoretical_n_minus_1_over_2']]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'red']

        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Mean Discovery Number', fontsize=11)
        ax.set_title(f'GP({n}, {k}) - {res["graph_name"]}\n'
                    f'Vertices: {res["total_vertices"]}, Samples: {res["num_samples"]:,}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.2)

    fig2.suptitle('Summary: Outer vs Inner Ring Discovery Numbers',
                  fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the summary figure
    plt.savefig('results/petersen_graphs_summary.png', dpi=150, bbox_inches='tight')
    print("Summary visualization saved to: results/petersen_graphs_summary.png")
    plt.close(fig2)  # Close instead of show


if __name__ == "__main__":
    # Test the four example Generalized Petersen graphs
    petersen_configs = [
        (5, 2),   # Classic Petersen graph (10 vertices)
        (4, 1),   # Cube graph (8 vertices)
        (5, 1),   # Prism graph (10 vertices)
        (8, 3),   # Möbius-Kantor graph (16 vertices)
    ]

    num_samples = 20000

    all_results = run_petersen_experiments(petersen_configs, num_samples)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS...")
    print("="*80)

    # Create visualizations
    visualize_all_results(all_results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("The (n-1)/2 conjecture was tested on 4 different Generalized Petersen graphs.")
    print("Results show how well the conjecture holds for these symmetric 3-regular graphs.")
    print("Check the generated PNG files for visualizations of the outer/inner ring pattern.")
    print("="*80 + "\n")
