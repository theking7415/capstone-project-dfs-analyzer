#!/usr/bin/env python3
"""
Interactive CLI for DFS Graph Analyzer.

Provides a user-friendly menu-driven interface for running experiments.
"""

import sys
from typing import Optional

from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 70)
    print("DFS GRAPH ANALYZER".center(70))
    print("=" * 70)
    print("Validating the (n-1)/2 conjecture for symmetric regular graphs")
    print("=" * 70 + "\n")


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 70)
    print("MAIN MENU")
    print("=" * 70)
    print("1. Run new experiment")
    print("2. Help & Documentation")
    print("3. About")
    print("4. Exit")
    print("=" * 70)


def get_user_choice(valid_choices: list[int]) -> int:
    """
    Get a valid menu choice from the user.

    Args:
        valid_choices: List of valid integer choices.

    Returns:
        The user's choice.
    """
    while True:
        try:
            choice = input("\nEnter your choice: ").strip()
            choice_int = int(choice)
            if choice_int in valid_choices:
                return choice_int
            else:
                print(f"Invalid choice. Please enter one of: {valid_choices}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


def get_integer_input(
    prompt: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    default: Optional[int] = None,
) -> int:
    """
    Get an integer input from the user with validation.

    Args:
        prompt: The prompt to display.
        min_val: Minimum valid value.
        max_val: Maximum valid value.
        default: Default value if user presses Enter.

    Returns:
        The validated integer.
    """
    while True:
        try:
            if default is not None:
                full_prompt = f"{prompt} (default: {default}): "
            else:
                full_prompt = prompt

            user_input = input(full_prompt).strip()

            if user_input == "" and default is not None:
                return default

            value = int(user_input)

            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Ask a yes/no question.

    Args:
        prompt: The question to ask.
        default: Default answer if user presses Enter.

    Returns:
        True for yes, False for no.
    """
    default_str = "Y/n" if default else "y/N"
    while True:
        try:
            response = input(f"{prompt} [{default_str}]: ").strip().lower()

            if response == "":
                return default

            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


def select_export_formats() -> list[str]:
    """
    Allow user to select export formats.

    Returns:
        List of selected format strings.
    """
    print("\nSelect export formats (comma-separated):")
    print("  csv  - Comma-separated values")
    print("  json - JSON format")
    print("  txt  - Detailed text report")
    print("  pickle - Python pickle (for reanalysis)")

    while True:
        try:
            response = input("Formats (default: csv,txt): ").strip()

            if response == "":
                return ["csv", "txt"]

            formats = [f.strip().lower() for f in response.split(",")]

            valid_formats = ["csv", "json", "txt", "pickle"]
            invalid = [f for f in formats if f not in valid_formats]

            if invalid:
                print(f"Invalid formats: {invalid}")
                print(f"Valid formats: {valid_formats}")
                continue

            return formats

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)


def get_recommended_samples(num_vertices: int) -> int:
    """
    Get recommended sample size based on graph size.

    Args:
        num_vertices: Number of vertices in the graph.

    Returns:
        Recommended sample count.
    """
    if num_vertices <= 16:
        return 50000
    elif num_vertices <= 64:
        return 25000
    elif num_vertices <= 256:
        return 10000
    elif num_vertices <= 1024:
        return 5000
    else:
        return 1000


def run_new_experiment():
    """Run a new experiment with user input."""
    print("\n" + "=" * 70)
    print("NEW EXPERIMENT")
    print("=" * 70)

    # Step 1: Graph type (only Hypercube for now)
    print("\nGraph type: Hypercube")
    print("(More graph types coming in future versions)")

    # Step 2: Get dimension
    print("\n--- Graph Parameters ---")
    dimension = get_integer_input(
        "Enter hypercube dimension (3-10 recommended): ", min_val=2, max_val=15
    )

    num_vertices = 2**dimension
    print(f"\nHypercube {dimension}D has {num_vertices} vertices")

    # Step 3: Get sample size
    print("\n--- Sampling Configuration ---")
    recommended = get_recommended_samples(num_vertices)
    print(f"Recommended samples for {num_vertices} vertices:")
    print(f"  Quick test:    {max(1000, recommended // 10)}")
    print(f"  Standard:      {recommended}")
    print(f"  High accuracy: {recommended * 2}")

    num_samples = get_integer_input(
        f"\nEnter number of samples (recommended: {recommended}): ", min_val=100
    )

    # Step 4: Advanced options
    print("\n--- Advanced Options ---")
    configure_advanced = ask_yes_no(
        "Configure advanced options?", default=False
    )

    if configure_advanced:
        rng_seed = get_integer_input(
            "RNG seed for reproducibility", default=1832479182
        )
        save_plots = ask_yes_no("Save visualizations?", default=True)
        export_formats = select_export_formats()
        output_dir = input("Output directory (default: data_output): ").strip()
        if not output_dir:
            output_dir = "data_output"
    else:
        rng_seed = 1832479182
        save_plots = True
        export_formats = ["csv", "txt"]
        output_dir = "data_output"

    # Step 5: Confirm and run
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Graph:          Hypercube {dimension}D ({num_vertices} vertices)")
    print(f"Samples:        {num_samples}")
    print(f"RNG Seed:       {rng_seed}")
    print(f"Save plots:     {'Yes' if save_plots else 'No'}")
    print(f"Export formats: {', '.join(export_formats)}")
    print(f"Output dir:     {output_dir}")
    print("=" * 70)

    if not ask_yes_no("\nProceed with experiment?", default=True):
        print("Experiment cancelled.")
        return

    # Step 6: Create config and run
    config = ExperimentConfig(
        graph_type="hypercube",
        dimension=dimension,
        num_samples=num_samples,
        rng_seed=rng_seed,
        output_dir=output_dir,
        save_plots=save_plots,
        export_formats=export_formats,
    )

    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT")
    print("=" * 70)
    print("This may take a moment...\n")

    # Progress callback
    last_percent = -1

    def progress_callback(current, total):
        nonlocal last_percent
        percent = int(100 * current / total)
        if percent != last_percent or current == total:
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\rProgress: [{bar}] {percent}% ({current}/{total})", end="", flush=True)
            last_percent = percent

    runner = ExperimentRunner()
    try:
        results = runner.run(config, progress_callback=progress_callback)
        print("\n")  # New line after progress bar
    except Exception as e:
        print(f"\n\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 7: Display results
    print("\n" + results.get_summary())
    print(f"\nResults saved to: {results.output_path}")

    # Step 8: Post-experiment options
    print("\n" + "=" * 70)
    print("What would you like to do next?")
    print("=" * 70)
    print("1. Run another experiment")
    print("2. View detailed statistics")
    print("3. Return to main menu")
    print("=" * 70)

    choice = get_user_choice([1, 2, 3])

    if choice == 1:
        run_new_experiment()
    elif choice == 2:
        print(f"\nDetailed statistics saved to:")
        print(f"  {results.output_path}/detailed_stats.txt")
        print(f"  {results.output_path}/data.csv")
        input("\nPress Enter to continue...")
    # choice == 3 returns to main menu


def show_help():
    """Display help and documentation."""
    print("\n" + "=" * 70)
    print("HELP & DOCUMENTATION")
    print("=" * 70)
    print("""
This tool validates the (n-1)/2 conjecture for symmetric regular graphs.

THE CONJECTURE:
  For large symmetric regular graphs, the average discovery number of a
  node in randomized DFS tends to (n-1)/2, where n is the number of nodes.

GRAPH TYPES:
  - Hypercube: A d-dimensional hypercube has 2^d vertices. Each vertex
    is represented as a binary tuple (e.g., (0,1,0) for 3D). Two vertices
    are adjacent if they differ in exactly one bit.

SAMPLE SIZE:
  - More samples = more accurate results but longer runtime
  - Recommendations scale with graph size
  - For publication-quality results, use 10,000+ samples

OUTPUT FILES:
  - summary.txt: Human-readable summary
  - data.csv: Per-vertex statistics (open in Excel)
  - data.json: Machine-readable format
  - visualization.png: Bar chart of discovery numbers
  - data.pickle: Raw data for further analysis

REPRODUCIBILITY:
  - The RNG seed ensures identical results across runs
  - Default seed: 1832479182
  - Change seed for different random samples

For more information, visit:
  https://github.com/yourusername/dfs-graph-analyzer
    """)
    print("=" * 70)
    input("\nPress Enter to continue...")


def show_about():
    """Display about information."""
    print("\n" + "=" * 70)
    print("ABOUT DFS GRAPH ANALYZER")
    print("=" * 70)
    print("""
DFS Graph Analyzer v0.1.0

A tool for empirically validating the (n-1)/2 conjecture on symmetric
regular graphs using randomized depth-first search.

Research conducted at Ashoka University as part of a capstone project
investigating graph traversal properties.

CITATION:
  If you use this tool in your research, please cite:
  [Citation details to be added]

LICENSE:
  MIT License (see LICENSE file for details)

AUTHOR:
  Venkat Mahesh Mandava
  Ashoka University

SOURCE CODE:
  https://github.com/yourusername/dfs-graph-analyzer

CONTRIBUTORS:
  - Venkat Mahesh Mandava (original research and implementation)
  - [Add contributors here]
    """)
    print("=" * 70)
    input("\nPress Enter to continue...")


def main():
    """Main entry point for the CLI."""
    print_banner()

    while True:
        print_menu()
        choice = get_user_choice([1, 2, 3, 4])

        if choice == 1:
            run_new_experiment()
        elif choice == 2:
            show_help()
        elif choice == 3:
            show_about()
        elif choice == 4:
            print("\nThank you for using DFS Graph Analyzer!")
            print("Goodbye!\n")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
