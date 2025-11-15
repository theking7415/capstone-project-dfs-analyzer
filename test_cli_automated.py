#!/usr/bin/env python3
"""
Automated test for CLI - simulates user input.
"""

import sys
sys.path.insert(0, '/mnt/c/Users/mahes/Desktop/Ashoka/Capstone project/My data')

from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner

def test_cli_backend():
    """Test the CLI backend with a small experiment."""
    print("Testing CLI backend with automated experiment...")
    print("=" * 70)

    # Simulate user choosing: Hypercube 3D, 500 samples
    config = ExperimentConfig(
        graph_type="hypercube",
        dimension=3,
        num_samples=500,
        rng_seed=1832479182,
        output_dir="data_output",
        save_plots=True,
        export_formats=["csv", "txt", "json"]
    )

    print("\nConfiguration:")
    print(f"  {config.get_graph_description()}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Output: {config.output_dir}")
    print()

    # Progress bar
    def progress(current, total):
        if current % 50 == 0 or current == total:
            percent = 100 * current / total
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\rProgress: [{bar}] {int(percent)}% ({current}/{total})", end="", flush=True)

    # Run experiment
    runner = ExperimentRunner()
    results = runner.run(config, progress_callback=progress)

    print("\n")
    print(results.get_summary())
    print(f"\n✓ CLI backend test completed successfully!")
    print(f"Results saved to: {results.output_path}")

    return results.validation["is_valid"]

if __name__ == "__main__":
    success = test_cli_backend()
    sys.exit(0 if success else 1)
