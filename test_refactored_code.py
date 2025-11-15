"""
Test script to verify the refactored code produces identical results.
"""

import sys
sys.path.insert(0, '/mnt/c/Users/mahes/Desktop/Ashoka/Capstone project/My data')

from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner

def test_basic_experiment():
    """Test a basic experiment with a small hypercube."""
    print("Testing refactored code with Hypercube 3D, 1000 samples...")
    print("=" * 70)

    # Create configuration
    config = ExperimentConfig(
        graph_type="hypercube",
        dimension=3,
        num_samples=1000,
        rng_seed=1832479182,
        output_dir="data_output",
        save_plots=True,
        export_formats=["csv", "json", "txt", "pickle"]
    )

    print(f"Configuration:")
    print(f"  Graph: {config.get_graph_description()}")
    print(f"  Samples: {config.num_samples}")
    print(f"  RNG Seed: {config.rng_seed}")
    print()

    # Run experiment
    runner = ExperimentRunner()

    def progress_callback(current, total):
        if current % 100 == 0 or current == total:
            print(f"  Progress: {current}/{total} ({100*current/total:.1f}%)")

    print("Running experiment...")
    results = runner.run(config, progress_callback=progress_callback)

    print()
    print(results.get_summary())
    print()
    print(f"Results saved to: {results.output_path}")
    print()

    # Verify the conjecture
    if results.validation["is_valid"]:
        print("✓ SUCCESS: Refactored code produces valid results!")
        print(f"  Theoretical: {results.validation['theoretical_value']:.4f}")
        print(f"  Observed: {results.validation['observed_value']:.4f}")
        print(f"  Error: {results.validation['relative_error']*100:.4f}%")
        return True
    else:
        print("✗ FAILURE: Results do not match conjecture!")
        return False

if __name__ == "__main__":
    success = test_basic_experiment()
    sys.exit(0 if success else 1)
