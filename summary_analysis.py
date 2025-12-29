"""
Generate a summary analysis of all RDFS results
"""

import pickle
import os
from pathlib import Path
from myrdfs import get_summary_stats
import numpy as np

def read_pickle(fname):
    """Load graph and statistics from pickle file"""
    with open(fname, "rb") as f:
        return pickle.load(f)

def main():
    """Generate summary analysis"""
    
    data_dir = "data"
    pickle_files = sorted(Path(data_dir).glob("rdfs-hypercube-*-samples.pickle"))
    
    # Group by dimension
    results_by_dim = {}
    for pickle_file in pickle_files:
        # Extract dimension from filename (e.g., "rdfs-hypercube-3d-...")
        parts = pickle_file.name.split('-')
        dim = parts[2]  # e.g., "3d"
        
        if dim not in results_by_dim:
            results_by_dim[dim] = []
        results_by_dim[dim].append(pickle_file)
    
    print("=" * 100)
    print("COMPREHENSIVE RDFS ANALYSIS SUMMARY")
    print("Testing Conjecture: Mean Discovery Number → (n-1)/2")
    print("=" * 100)
    
    for dim in sorted(results_by_dim.keys(), key=lambda x: int(x[:-1])):
        print(f"\n{dim.upper()} HYPERCUBE")
        print("-" * 100)
        
        # Get the most recent/best data for this dimension
        pickle_file = results_by_dim[dim][-1]  # Last file (newest)
        
        try:
            graph, dist_stats = read_pickle(str(pickle_file))
            summary_stats = get_summary_stats(dist_stats)
            
            n = graph.number_vertices()
            conjecture_value = (n - 1) / 2
            
            # Get all mean discovery numbers
            means = [summary_stats[v].mean for v in sorted(summary_stats.keys())]
            overall_mean = np.mean(means)
            overall_std = np.std(means)
            
            num_samples = len(list(dist_stats.values())[0])
            
            print(f"  Graph: {graph.desc()}")
            print(f"  Vertices (n): {n}")
            print(f"  Samples per vertex: {num_samples:,}")
            print(f"  File: {pickle_file.name}")
            print()
            print(f"  Conjecture (n-1)/2: {conjecture_value:.2f}")
            print(f"  Observed mean: {overall_mean:.4f}")
            print(f"  Std deviation: {overall_std:.4f}")
            print(f"  Error: {abs(overall_mean - conjecture_value):.4f}")
            print(f"  % of conjecture: {(overall_mean / conjecture_value * 100):.2f}%")
            
            # Show vertex-by-vertex for small graphs
            if n <= 16:
                print(f"\n  Per-vertex breakdown:")
                for vertex in sorted(summary_stats.keys()):
                    mean_disc = summary_stats[vertex].mean
                    print(f"    {str(vertex):20} → {mean_disc:7.4f}")
        
        except Exception as e:
            print(f"  Error processing {pickle_file.name}: {e}")
    
    print("\n" + "=" * 100)
    print("CONJECTURE VALIDATION")
    print("=" * 100)
    print("\nIf the conjecture is correct, the 'Observed mean' should approach '(n-1)/2'")
    print("As dimensions increase, the convergence to the theoretical value should improve.")
    print("=" * 100)

if __name__ == "__main__":
    main()
