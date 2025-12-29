"""
Batch runner for Triangular Lattice experiments.
Sizes: 10×10, 20×20, 30×30, ..., 100×100
1000 samples per experiment
"""
from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner
import time

def main():
    # Grid sizes: 10×10 to 100×100
    sizes = [10 * i for i in range(1, 11)]  # 10, 20, 30, ..., 100

    experiments = []
    for size in sizes:
        vertices = size * size
        experiments.append({
            "rows": size,
            "cols": size,
            "vertices": vertices
        })

    print("="*70)
    print("TRIANGULAR LATTICE BATCH EXPERIMENTS")
    print("="*70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Samples per experiment: 1000")
    print(f"Degree: 6 (constant for all triangular lattices)")
    print()
    print("Experiment configurations:")
    print(f"{'Size':<15} {'Vertices':<12}")
    print("-"*70)
    for exp in experiments:
        print(f"{exp['rows']}×{exp['cols']:<12} {exp['vertices']:<12}")
    print()

    runner = ExperimentRunner()
    results_list = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        rows = exp["rows"]
        cols = exp["cols"]
        vertices = exp["vertices"]

        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(experiments)}: Triangular Lattice {rows}×{cols}")
        print(f"{'='*70}")
        print(f"Vertices: {vertices:,}")
        print(f"Degree: 6")
        print(f"Samples: 1000")
        print()

        config = ExperimentConfig(
            graph_type='triangular',
            lattice_rows=rows,
            lattice_cols=cols,
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

            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            print(f"  Mean discovery: {results.summary_stats[results.graph.get_start_vertex()].mean:.4f}")
            print(f"  Expected: {(vertices - 1) / 2:.4f}")
            print(f"  Results: {results.output_path}")

            results_list.append({
                "rows": rows,
                "cols": cols,
                "vertices": vertices,
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
    print(f"{'Size':<15} {'Vertices':<12} {'Time (s)':<12} {'Output Path'}")
    print("-"*70)

    for r in results_list:
        print(f"{r['rows']}×{r['cols']:<12} {r['vertices']:<12} {r['elapsed']:<12.1f} {r['path']}")

    print("="*70)

if __name__ == '__main__':
    main()
