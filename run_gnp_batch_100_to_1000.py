"""
Batch runner for G(n,p) random graph experiments.
Fixed expected degree = 10
n = 10,000 to 100,000 (increments of 10,000)
1000 samples per experiment
"""
from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner
import time
import numpy as np

def main():
    # Target expected degree
    target_degree = 20

    # n values: 100 to 1000
    n_values = [100 * i for i in range(1, 11)]  # 100, 200, ..., 1000

    # Calculates p for each n to maintain E[degree] = target_degree
    # E[degree] = (n-1) * p, so p = target_degree / (n-1)
    experiments = []
    for n in n_values:
        p = target_degree / (n - 1)
        expected_edges = n * target_degree / 2
        threshold = (np.log(n) + 3) / n

        experiments.append({
            "n": n,
            "p": p,
            "expected_edges": int(expected_edges),
            "threshold": threshold
        })

    print("="*70)
    print("G(n,p) RANDOM GRAPH BATCH EXPERIMENTS")
    print("="*70)
    print(f"Fixed expected degree: {target_degree}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Samples per experiment: 1000")
    print()
    print("Experiment configurations:")
    print(f"{'n':<10} {'p':<12} {'E[edges]':<12} {'Threshold':<12}")
    print("-"*70)
    for exp in experiments:
        print(f"{exp['n']:<10} {exp['p']:<12.6f} {exp['expected_edges']:<12} {exp['threshold']:<12.6f}")
    print()

    runner = ExperimentRunner()
    results_list = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        n = exp["n"]
        p = exp["p"]
        expected_edges = exp["expected_edges"]

        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(experiments)}: G({n}, {p:.6f})")
        print(f"{'='*70}")
        print(f"Expected degree: {target_degree}")
        print(f"Expected edges: {expected_edges:,}")
        print(f"Connectivity threshold: {exp['threshold']:.6f}")
        print(f"p/threshold ratio: {p/exp['threshold']:.2f}x")
        print()

        config = ExperimentConfig(
            graph_type='gnp',
            dimension=n,
            gnp_p=p,
            num_samples=1000,
            rng_seed=1832479182,
            save_csv=True,
            save_plots=True,
            save_detailed_stats=True
        )

        start_time = time.time()

        try:
            results = runner.run(config)
            elapsed = time.time() - start_time

            # Gets actual edge count from graph
            actual_edges = results.graph.number_edges()

            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            print(f"  Actual edges: {actual_edges:,} (expected: {expected_edges:,})")
            print(f"  Mean discovery: {results.summary_stats[0].mean:.4f}")
            print(f"  Expected: {(n - 1) / 2:.4f}")
            print(f"  Results: {results.output_path}")

            results_list.append({
                "n": n,
                "p": p,
                "expected_edges": expected_edges,
                "actual_edges": actual_edges,
                "elapsed": elapsed,
                "path": results.output_path
            })

        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    total_elapsed = time.time() - total_start

    print("\n" + "="*70)
    print("BATCH COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print()
    print("Results summary:")
    print(f"{'n':<10} {'p':<12} {'Edges':<12} {'Time (s)':<12} {'Output Path'}")
    print("-"*70)

    for r in results_list:
        print(f"{r['n']:<10} {r['p']:<12.6f} {r['actual_edges']:<12,} {r['elapsed']:<12.1f} {r['path']}")

    print("="*70)

if __name__ == '__main__':
    main()
