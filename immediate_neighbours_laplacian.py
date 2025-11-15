"""
Test the n/π theory using graph Laplacian (random walk) approach.

This module uses the Laplacian matrix and effective resistance to compute
the expected hitting times for immediate neighbors in 2D grid graphs,
similar to the approach used in random_walk_hypercube.py.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import csv
import time


def create_2d_grid_graph(width: int, height: int):
    """
    Create a 2D grid graph using NetworkX.

    Args:
        width: Number of columns
        height: Number of rows

    Returns:
        NetworkX grid graph
    """
    G = nx.grid_2d_graph(width, height)
    return G


def get_immediate_neighbors(G, start_node):
    """
    Get the immediate neighbors of the starting node.

    Args:
        G: NetworkX graph
        start_node: Starting vertex

    Returns:
        List of immediate neighbor nodes
    """
    return list(G.neighbors(start_node))


def compute_hitting_times_laplacian(G, start_node):
    """
    Compute expected hitting times from all nodes to the start_node
    using the Laplacian matrix approach.

    This implements the "single-factor" method from random_walk_hypercube.py

    Args:
        G: NetworkX graph
        start_node: Target node (sink)

    Returns:
        Dictionary mapping each node to its expected hitting time to start_node
    """
    # Get sorted list of all nodes
    nodelist = sorted(list(G.nodes()))
    N = len(nodelist)

    # Get the graph Laplacian matrix
    L = nx.laplacian_matrix(G, nodelist=nodelist).astype(np.float64)

    # Find the index of the target (start) node
    target_idx = nodelist.index(start_node)

    # Create the minor of the Laplacian by removing the row and column
    # corresponding to the target node
    minor_nodes_indices = [i for i in range(N) if i != target_idx]
    L_minor = L[minor_nodes_indices, :][:, minor_nodes_indices]

    # Factorize L_minor using sparse LU decomposition
    lu_factor = spla.splu(L_minor.tocsc())

    # Compute effective resistances
    effective_resistances = np.zeros(N - 1)
    for i in range(N - 1):
        # Create the standard basis vector e_x
        e_x = np.zeros(N - 1)
        e_x[i] = 1.0

        # Solve the system using the pre-computed LU factorization
        y = lu_factor.solve(e_x)

        # The effective resistance is the diagonal entry of the inverse
        effective_resistances[i] = y[i]

    # Assemble the "current" vector b
    b = 1.0 / effective_resistances

    # Solve the final linear system L_minor * phi = b to get the costs
    phi_costs = lu_factor.solve(b)

    # The full cost array includes the cost from the target to itself (which is 0)
    all_costs = np.zeros(N)
    all_costs[minor_nodes_indices] = phi_costs

    # Create a dictionary mapping nodes to their hitting times
    hitting_times = {}
    for i, node in enumerate(nodelist):
        hitting_times[node] = all_costs[i]

    return hitting_times


def analyze_grid_laplacian(width: int, height: int):
    """
    Analyze immediate neighbor hitting times for a 2D grid using Laplacian method.

    Args:
        width: Grid width
        height: Grid height

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {width}x{height} grid using Laplacian method")
    print(f"{'='*60}")

    # Create the grid graph
    G = create_2d_grid_graph(width, height)
    n = G.number_of_nodes()

    # Starting node is at the center
    start_node = (width // 2, height // 2)

    # Get immediate neighbors
    immediate_neighbors = get_immediate_neighbors(G, start_node)

    print(f"Number of vertices (n): {n}")
    print(f"Starting vertex: {start_node}")
    print(f"Immediate neighbors: {immediate_neighbors}")
    print(f"Number of immediate neighbors: {len(immediate_neighbors)}")

    # Compute hitting times using Laplacian
    start_time = time.time()
    hitting_times = compute_hitting_times_laplacian(G, start_node)
    computation_time = time.time() - start_time

    print(f"Computation time: {computation_time:.2f} seconds")

    # Get hitting times for immediate neighbors
    neighbor_hitting_times = [hitting_times[neighbor] for neighbor in immediate_neighbors]

    # Compute statistics
    avg_hitting_time = np.mean(neighbor_hitting_times)
    std_hitting_time = np.std(neighbor_hitting_times)
    theoretical_value = n / np.pi

    print(f"\n{'Neighbor':<15} {'Hitting Time':<15}")
    print("-" * 35)
    for neighbor, ht in zip(immediate_neighbors, neighbor_hitting_times):
        print(f"{str(neighbor):<15} {ht:<15.4f}")

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Average hitting time of immediate neighbors: {avg_hitting_time:.4f}")
    print(f"Standard deviation across neighbors: {std_hitting_time:.4f}")
    print(f"Theoretical prediction (n/π): {theoretical_value:.4f}")
    print(f"Difference: {abs(avg_hitting_time - theoretical_value):.4f}")
    print(f"Relative error: {abs(avg_hitting_time - theoretical_value) / theoretical_value * 100:.2f}%")
    print(f"{'='*60}\n")

    # Store results
    results = {
        'grid_size': f"{width}x{height}",
        'width': width,
        'height': height,
        'n': n,
        'theoretical_n_over_pi': theoretical_value,
        'start_node': start_node,
        'immediate_neighbors': immediate_neighbors,
        'neighbor_hitting_times': neighbor_hitting_times,
        'avg_hitting_time': avg_hitting_time,
        'std_hitting_time': std_hitting_time,
        'relative_error': abs(avg_hitting_time - theoretical_value) / theoretical_value,
        'computation_time': computation_time
    }

    return results


def run_multiple_grid_experiments_laplacian(grid_sizes: list[int]):
    """
    Run Laplacian-based experiments on multiple grid sizes.

    Args:
        grid_sizes: List of grid dimensions (for square grids)

    Returns:
        List of result dictionaries
    """
    all_results = []

    print(f"\n{'='*80}")
    print(f"RUNNING LAPLACIAN-BASED EXPERIMENTS ON MULTIPLE GRID SIZES")
    print(f"Grid sizes: {grid_sizes}")
    print(f"{'='*80}\n")

    for size in grid_sizes:
        results = analyze_grid_laplacian(size, size)
        all_results.append(results)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE - LAPLACIAN METHOD")
    print(f"{'='*80}")
    print(f"{'Grid':<10} {'Vertices':<10} {'n/π':<12} {'Observed':<12} {'Diff':<12} {'Error %':<10} {'Time (s)':<10}")
    print(f"{'-'*90}")

    for res in all_results:
        grid_desc = res['grid_size']
        n = res['n']
        theoretical = res['theoretical_n_over_pi']
        observed = res['avg_hitting_time']
        diff = observed - theoretical
        error_pct = res['relative_error'] * 100
        comp_time = res['computation_time']

        print(f"{grid_desc:<10} {n:<10} {theoretical:<12.4f} {observed:<12.4f} "
              f"{diff:<12.4f} {error_pct:<10.2f} {comp_time:<10.2f}")

    print(f"{'='*90}\n")

    # Save results to CSV file
    csv_filename = "results/immediate_neighbors_laplacian_experiment.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Grid Size', 'Vertices (n)', 'Theoretical (n/π)',
                           'Observed Mean', 'Difference', 'Relative Error (%)',
                           'Std Dev Across Neighbors', 'Computation Time (s)'])

            # Write data rows
            for res in all_results:
                grid_desc = res['grid_size']
                n = res['n']
                theoretical = res['theoretical_n_over_pi']
                observed = res['avg_hitting_time']
                diff = observed - theoretical
                error_pct = res['relative_error'] * 100
                std_dev = res['std_hitting_time']
                comp_time = res['computation_time']

                writer.writerow([grid_desc, n, theoretical, observed, diff,
                               error_pct, std_dev, comp_time])

        print(f"Results saved to: {csv_filename}\n")
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}\n")

    return all_results


if __name__ == "__main__":
    # Run experiments on multiple grid sizes using Laplacian method
    grid_sizes = [5, 10, 20, 30, 40, 50, 100]

    all_results = run_multiple_grid_experiments_laplacian(grid_sizes)

    print("\n" + "="*80)
    print("COMPARISON: RDFS vs LAPLACIAN METHOD")
    print("="*80)
    print("Note: Load both CSV files to compare the empirical RDFS approach")
    print("      with the theoretical Laplacian (random walk) approach.")
    print("="*80 + "\n")
