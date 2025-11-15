"""
Test the (n-1)/2 conjecture on Erdős-Rényi G(n,p) random graphs.

G(n,p) graphs are random graphs where each pair of vertices is connected
with independent probability p.

We test at the connectivity threshold: p = log(n)/n
Disconnected graphs are discarded, and we analyze only connected instances.
"""

import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from scipy import stats
import csv
import networkx as nx
import math

from mygraphs import Graph
from myrdfs import collect_statistics, get_summary_stats, RNG


GNPVertexType = int


class GNPGraph(Graph[GNPVertexType]):
    """An Erdős-Rényi G(n,p) random graph."""

    def __init__(self, n: int, p: float, seed: int = None):
        """
        Creates a G(n,p) random graph.

        Args:
            n: Number of vertices
            p: Probability of edge between any two vertices
            seed: Random seed for reproducibility
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")

        self.n = n
        self.p = p
        self.seed = seed

        # Generate the random graph using NetworkX
        self.G = nx.erdos_renyi_graph(n, p, seed=seed)

        # Check if connected
        self.is_connected = nx.is_connected(self.G)

    def get_start_vertex(self) -> GNPVertexType:
        """Start at vertex 0."""
        return 0

    def get_adj_list(self, v: GNPVertexType) -> list[GNPVertexType]:
        """Returns neighbors of vertex v."""
        return list(self.G.neighbors(v))

    def desc(self) -> str:
        return f"gnp-n{self.n}-p{self.p:.4f}"

    def number_vertices(self) -> int:
        return self.n

    def plot_means_vars(self, summary_stats: dict[GNPVertexType, Any], *, fname=None):
        """Visualize mean DFS numbers for G(n,p) graph vertices."""
        vertices = sorted(summary_stats.keys())
        means = [summary_stats[v].mean for v in vertices]
        sds = [np.sqrt(summary_stats[v].variance) for v in vertices]

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(vertices, means, yerr=sds, capsize=3, color='skyblue', ecolor='gray', alpha=0.7)

        # Highlight starting vertex
        ax.bar([0], [means[0]], color='orange', alpha=1.0)

        theoretical = (self.n - 1) / 2
        ax.axhline(y=theoretical, color='red', linestyle='--',
                   linewidth=2, label=f'(n-1)/2 = {theoretical:.2f}')

        ax.set_ylabel('Mean Discovery Number', fontsize=11)
        ax.set_xlabel('Vertex ID', fontsize=11)
        ax.set_title(f'G(n={self.n}, p={self.p:.4f}) - Discovery Numbers\n'
                    f'Orange = Start Vertex',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if fname:
            plt.savefig(fname)
        plt.tight_layout()
        plt.close()


def analyze_gnp_graph(n: int, p: float, num_samples: int, max_attempts: int = 100, *, rng=RNG):
    """
    Analyze the (n-1)/2 conjecture on a G(n,p) random graph.

    Generates random G(n,p) graphs until a connected one is found.

    Args:
        n: Number of vertices
        p: Edge probability
        num_samples: Number of RDFS samples to run
        max_attempts: Maximum attempts to generate a connected graph
        rng: Random number generator

    Returns:
        Dictionary with analysis results, or None if no connected graph found
    """
    print(f"\n{'='*70}")
    print(f"Testing (n-1)/2 conjecture on G(n={n}, p={p:.4f})")
    print(f"{'='*70}")

    # Try to generate a connected graph
    connected_graph = None
    for attempt in range(max_attempts):
        # Use different seeds for each attempt (NetworkX requires int type, not numpy int)
        graph = GNPGraph(n, p, seed=int(RNG.integers(0, 1000000)))
        if graph.is_connected:
            connected_graph = graph
            print(f"Connected graph found on attempt {attempt + 1}")
            break

    if connected_graph is None:
        print(f"WARNING: Could not find connected graph after {max_attempts} attempts")
        print(f"Skipping this configuration.")
        return None

    graph = connected_graph
    theoretical_value = (n - 1) / 2

    # Get some graph statistics
    degrees = [graph.G.degree(v) for v in graph.G.nodes()]
    avg_degree = np.mean(degrees)
    min_degree = np.min(degrees)
    max_degree = np.max(degrees)

    print(f"Graph: G(n={n}, p={p:.4f})")
    print(f"Number of vertices: {n}")
    print(f"Number of edges: {graph.G.number_of_edges()}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Degree range: [{min_degree}, {max_degree}]")
    print(f"Theoretical value (n-1)/2: {theoretical_value:.4f}")
    print(f"Running {num_samples} RDFS samples...")

    # Collect statistics
    dist_stats = collect_statistics(graph, num_samples, rng=rng)
    summary_stats = get_summary_stats(dist_stats)

    # Calculate overall average discovery number
    all_means = [stat.mean for stat in summary_stats.values()]
    avg_discovery = np.mean(all_means)
    std_discovery = np.std(all_means)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Average discovery number: {avg_discovery:.4f}")
    print(f"Standard deviation across vertices: {std_discovery:.4f}")
    print(f"Theoretical prediction (n-1)/2: {theoretical_value:.4f}")
    print(f"Difference: {abs(avg_discovery - theoretical_value):.4f}")
    print(f"Relative error: {abs(avg_discovery - theoretical_value) / theoretical_value * 100:.2f}%")
    print(f"{'='*70}\n")

    # Store results
    results = {
        'n': n,
        'p': p,
        'p_formula': f'log(n)/n',
        'num_edges': graph.G.number_of_edges(),
        'avg_degree': avg_degree,
        'min_degree': min_degree,
        'max_degree': max_degree,
        'theoretical_n_minus_1_over_2': theoretical_value,
        'num_samples': num_samples,
        'avg_discovery': avg_discovery,
        'std_discovery': std_discovery,
        'relative_error': abs(avg_discovery - theoretical_value) / theoretical_value,
        'summary_stats': summary_stats,
        'graph': graph
    }

    return results


def run_gnp_experiments(n_values: list[int], num_samples: int):
    """
    Run experiments on multiple G(n,p) graphs at connectivity threshold.

    Args:
        n_values: List of n values to test
        num_samples: Number of RDFS samples per graph

    Returns:
        List of result dictionaries
    """
    all_results = []

    print(f"\n{'='*80}")
    print(f"RUNNING (n-1)/2 CONJECTURE TESTS ON G(n,p) RANDOM GRAPHS")
    print(f"Testing at connectivity threshold: p = log(n)/n")
    print(f"Samples per graph: {num_samples}")
    print(f"n values to test: {n_values}")
    print(f"{'='*80}\n")

    for n in n_values:
        # Connectivity threshold: p = log(n)/n
        p = math.log(n) / n
        results = analyze_gnp_graph(n, p, num_samples, rng=RNG)

        if results is not None:
            all_results.append(results)

    if not all_results:
        print("WARNING: No connected graphs were found. Exiting.")
        return []

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE - (n-1)/2 CONJECTURE ON G(n,p) RANDOM GRAPHS")
    print(f"{'='*90}")
    print(f"{'n':<8} {'p':<12} {'Edges':<8} {'Avg Deg':<10} {'(n-1)/2':<12} {'Observed':<12} {'Diff':<12} {'Error %':<10}")
    print(f"{'-'*90}")

    for res in all_results:
        n = res['n']
        p = res['p']
        edges = res['num_edges']
        avg_deg = res['avg_degree']
        theoretical = res['theoretical_n_minus_1_over_2']
        observed = res['avg_discovery']
        diff = observed - theoretical
        error_pct = res['relative_error'] * 100

        print(f"{n:<8} {p:<12.4f} {edges:<8} {avg_deg:<10.2f} {theoretical:<12.4f} {observed:<12.4f} "
              f"{diff:<12.4f} {error_pct:<10.2f}")

    print(f"{'='*90}\n")

    # Save results to CSV file
    csv_filename = f"results/gnp_graphs_experiment_{num_samples}_samples.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['n', 'p', 'p_formula', 'Edges', 'Avg Degree', 'Min Degree',
                           'Max Degree', 'Theoretical (n-1)/2', 'Observed Mean',
                           'Difference', 'Relative Error (%)', 'Std Dev Across Vertices',
                           'Samples'])

            # Write data rows
            for res in all_results:
                n = res['n']
                p = res['p']
                p_formula = res['p_formula']
                edges = res['num_edges']
                avg_deg = res['avg_degree']
                min_deg = res['min_degree']
                max_deg = res['max_degree']
                theoretical = res['theoretical_n_minus_1_over_2']
                observed = res['avg_discovery']
                diff = observed - theoretical
                error_pct = res['relative_error'] * 100
                std_dev = res['std_discovery']
                samples = res['num_samples']

                writer.writerow([n, p, p_formula, edges, avg_deg, min_deg, max_deg,
                               theoretical, observed, diff, error_pct, std_dev, samples])

        print(f"Results saved to: {csv_filename}\n")
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}\n")

    return all_results


def visualize_gnp_results(all_results):
    """Create visualizations for G(n,p) results."""
    if not all_results:
        print("No results to visualize.")
        return

    n_graphs = len(all_results)
    fig, axes = plt.subplots(2, (n_graphs + 1) // 2, figsize=(16, 10))
    if n_graphs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, res in enumerate(all_results):
        ax = axes[idx]
        n = res['n']
        p = res['p']
        theoretical = res['theoretical_n_minus_1_over_2']
        observed = res['avg_discovery']

        # Create comparison bars
        categories = ['Theoretical\n(n-1)/2', 'Observed\nMean']
        values = [theoretical, observed]
        colors = ['red', 'skyblue']

        bars = ax.bar(categories, values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        error_pct = res['relative_error'] * 100
        ax.set_ylabel('Mean Discovery Number', fontsize=11)
        ax.set_title(f'G(n={n}, p={p:.4f})\n'
                    f'Edges: {res["num_edges"]}, Avg Deg: {res["avg_degree"]:.1f}\n'
                    f'Error: {error_pct:.2f}%',
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.2)

    # Hide unused subplots
    for idx in range(n_graphs, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('(n-1)/2 Conjecture on G(n,p) Random Graphs\n' +
                 'p = log(n)/n (Connectivity Threshold)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig('results/gnp_graphs_summary.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: results/gnp_graphs_summary.png")
    plt.close()


if __name__ == "__main__":
    # Test G(n,p) graphs at connectivity threshold
    # Using fewer graphs but more samples per graph
    n_values = [20, 50, 100, 200, 500]
    num_samples = 10000  # More samples per graph

    all_results = run_gnp_experiments(n_values, num_samples)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS...")
    print("="*80)

    # Create visualizations
    visualize_gnp_results(all_results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("The (n-1)/2 conjecture was tested on G(n,p) random graphs.")
    print("These are non-regular, asymmetric random graphs at the connectivity threshold.")
    print("Results show how well the conjecture holds for random graph structures.")
    print("="*80 + "\n")
