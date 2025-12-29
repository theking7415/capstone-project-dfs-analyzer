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
    print("Validating the expected behavior for symmetric regular graphs")
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


def get_float_input(
    prompt: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """
    Get a float input from the user with validation.

    Args:
        prompt: The prompt to display.
        min_val: Minimum valid value.
        max_val: Maximum valid value.
        default: Default value if user presses Enter.

    Returns:
        The validated float.
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

            value = float(user_input)

            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a valid number.")
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


def select_output_options() -> dict:
    """
    Allow user to select which outputs to generate.

    Returns:
        Dictionary with output option flags.
    """
    print("\nOutput Options:")
    print("  summary.txt is always generated")
    print("\nOptional outputs:")

    try:
        save_csv = ask_yes_no("  Save data.csv (per-vertex statistics)?", default=False)
        save_detailed = ask_yes_no("  Save detailed_stats.txt (full report)?", default=False)
        save_plots = ask_yes_no("  Save visualization.png (graph plot)?", default=False)

        # Ask about additional formats
        additional_formats = []
        if ask_yes_no("\n  Save additional formats (json/pickle)?", default=False):
            print("\n  Select additional formats (comma-separated):")
            print("    json   - JSON format (human-readable)")
            print("    pickle - Python pickle (for reanalysis)")

            response = input("  Formats (press Enter to skip): ").strip()
            if response:
                formats = [f.strip().lower() for f in response.split(",")]
                valid_formats = ["json", "pickle"]
                additional_formats = [f for f in formats if f in valid_formats]

        return {
            "save_csv": save_csv,
            "save_detailed_stats": save_detailed,
            "save_plots": save_plots,
            "export_formats": additional_formats,
        }

    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting...")
        sys.exit(0)


def get_recommended_samples(graph_type: str, dimension: int, num_vertices: int) -> int:
    """
    Get recommended sample size based on graph type.

    Uses proportional scaling with standardized samples per vertex:
    - Hypercubes: 2000 samples per vertex (all dimensions)
    - Petersen graphs: 1000 samples per vertex (constant degree 3)
    - Triangular lattice: 1000 samples per vertex (constant degree 6)
    - Torus grid: 1000 samples per vertex (constant degree 4)
    - Hexagonal lattice: 1000 samples per vertex (constant degree 3, graphene)

    Statistical validity: 2000 samples/vertex gives SE ≈ σ/44.7 and 95% CI ≈ ±0.04

    Args:
        graph_type: Type of graph ("hypercube", "petersen", "triangular", or "torus").
        dimension: Dimension parameter (d for hypercube, n for Petersen, rows for triangular/torus).
        num_vertices: Number of vertices in the graph.

    Returns:
        Recommended sample count.
    """
    if graph_type == "hypercube":
        # Standardized at 2000 samples per vertex for all dimensions
        # Balances statistical rigor (SE ≈ σ/44.7) with computational efficiency
        # Ensures consistent precision across all hypercube experiments
        samples_per_vertex = 2000

        recommended = int(samples_per_vertex * num_vertices)
        return recommended
    elif graph_type == "petersen":
        # Petersen graphs have constant degree 3, lower complexity
        samples_per_vertex = 1000
        return max(500, int(samples_per_vertex * num_vertices))
    elif graph_type == "triangular":
        # Triangular lattice has constant degree 6, similar complexity to Petersen
        samples_per_vertex = 1000
        return max(1000, int(samples_per_vertex * num_vertices))
    elif graph_type == "torus":
        # Torus grid has constant degree 4, similar complexity to Petersen/triangular
        samples_per_vertex = 1000
        return max(1000, int(samples_per_vertex * num_vertices))
    elif graph_type == "hexagonal":
        # Hexagonal lattice has constant degree 3 (same as Petersen), graphene structure
        samples_per_vertex = 1000
        return max(1000, int(samples_per_vertex * num_vertices))
    elif graph_type == "complete":
        # Complete graphs have degree n-1, high connectivity but simple structure
        samples_per_vertex = 1000
        return max(500, int(samples_per_vertex * num_vertices))
    elif graph_type == "ndgrid":
        # N-dimensional grids have constant degree 2d, similar complexity to lattices
        samples_per_vertex = 1000
        return max(1000, int(samples_per_vertex * num_vertices))
    elif graph_type == "gnp":
        # Random graphs have variable degree depending on p, use conservative estimate
        samples_per_vertex = 2000
        return max(1000, int(samples_per_vertex * num_vertices))
    else:
        # Default fallback
        samples_per_vertex = 1000
        return max(500, int(samples_per_vertex * num_vertices))


def run_gnp_batch_experiment():
    """Run G(n,p) batch experiment with user input."""
    import numpy as np
    from dfs_analyzer.experiments.gnp_batch_runner import GNPBatchRunner

    print("\n" + "=" * 70)
    print("G(n,p) BATCH EXPERIMENT")
    print("=" * 70)
    print("\nNote: G(n,p) graphs are NOT regular or symmetric.")
    print("This experiment explores DFS behavior on random graphs.")
    print("Multiple graphs will be generated and analyzed in batch mode.")
    print("=" * 70)

    # Step 1: Get n (number of vertices)
    print("\n--- Graph Parameters ---")
    n = get_integer_input("Enter n (number of vertices): ", min_val=10, max_val=1000)

    # Step 2: Get p (edge probability) with connectivity warning
    threshold = (np.log(n) + 3) / n if n > 1 else 0.5
    print(f"\nConnectivity threshold: p ≥ {threshold:.4f}")
    print(f"  (recommended minimum for high probability of connectivity)")

    p = get_float_input(
        f"\nEnter p (edge probability, 0 < p < 1): ",
        min_val=0.001,
        max_val=0.999
    )

    if p < threshold:
        print(f"\n[WARNING] Warning: p={p:.4f} is below threshold {threshold:.4f}")
        print(f"  Some graphs may fail to connect and will be discarded.")
        if not ask_yes_no("  Continue anyway?", default=False):
            print("Experiment cancelled.")
            return

    # Step 3: Get number of graphs
    print("\n--- Batch Parameters ---")
    num_graphs = get_integer_input(
        "Enter number of graphs to generate: ",
        min_val=1,
        max_val=10000
    )

    # Step 4: Get samples per graph
    recommended_samples = max(1000, n * 10)
    print(f"\nRecommended RDFS samples per graph: {recommended_samples}")
    num_samples_per_graph = get_integer_input(
        f"Enter RDFS samples per graph: ",
        min_val=100,
        default=recommended_samples
    )

    # Step 5: Advanced options
    print("\n--- Advanced Options ---")
    configure_advanced = ask_yes_no(
        "Configure advanced options (seed, output directory)?", default=False
    )

    if configure_advanced:
        rng_seed = get_integer_input(
            "RNG seed for reproducibility", default=1832479182
        )
        output_dir = input("Output directory (default: data_output): ").strip()
        if not output_dir:
            output_dir = "data_output"
    else:
        rng_seed = 1832479182
        output_dir = "data_output"

    # Step 6: Confirm experiment
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Graph type:             G(n,p) Random Graph")
    print(f"Vertices (n):           {n}")
    print(f"Edge probability (p):   {p:.4f}")
    print(f"Expected edges/graph:   {p * n * (n - 1) / 2:.1f}")
    print(f"Number of graphs:       {num_graphs}")
    print(f"RDFS samples/graph:     {num_samples_per_graph}")
    print(f"Total RDFS runs:        {num_graphs * num_samples_per_graph}")
    print(f"RNG Seed:               {rng_seed}")
    print(f"Output dir:             {output_dir}")
    print("\nFiles to generate:")
    print(f"  summary.txt:            Aggregate statistics")
    print(f"  per_graph_stats.csv:    Statistics for each graph")
    print("=" * 70)

    if not ask_yes_no("\nProceed with batch experiment?", default=True):
        print("Experiment cancelled.")
        return

    # Step 7: Run batch experiment
    print("\n" + "=" * 70)
    print("RUNNING BATCH EXPERIMENT")
    print("=" * 70)
    print("Generating and analyzing multiple G(n,p) graphs...\n")

    # Progress callback
    def progress_callback(current, total, message):
        percent = int(100 * current / total)
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r{message}: [{bar}] {percent}% ({current}/{total})", end="", flush=True)

    runner = GNPBatchRunner()

    try:
        results = runner.run(
            n=n,
            p=p,
            num_graphs=num_graphs,
            num_samples_per_graph=num_samples_per_graph,
            rng_seed=rng_seed,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        print("\n")  # New line after progress bar
    except Exception as e:
        print(f"\n\nError running batch experiment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 8: Display results
    print("\n" + results.get_summary())
    print(f"\nResults saved to: {results.output_path}")

    # Step 9: Post-experiment options
    print("\n" + "=" * 70)
    print("What would you like to do next?")
    print("=" * 70)
    print("1. Run another experiment")
    print("2. Return to main menu")
    print("=" * 70)

    choice = get_user_choice([1, 2])

    if choice == 1:
        run_new_experiment()
    # choice == 2 returns to main menu


def run_new_experiment():
    """Run a new experiment with user input."""
    print("\n" + "=" * 70)
    print("NEW EXPERIMENT")
    print("=" * 70)

    # Step 0: Select analysis type
    print("\n--- Analysis Type ---")
    print("1. Full Graph Analysis - All vertices")
    print("2. Immediate Neighbors - Only vertices adjacent to start")
    print("3. Opposite Vertex - Diagonally opposite vertex (hypercube only)")
    print("4. Custom Vertex Pair - Specify start and target vertices")

    analysis_type = get_user_choice([1, 2, 3, 4])

    if analysis_type == 1:
        print("\nSelected: Full Graph Analysis")
        focus = "full"
    elif analysis_type == 2:
        print("\nSelected: Immediate Neighbors")
        focus = "neighbors"
    elif analysis_type == 3:
        print("\nSelected: Opposite Vertex")
        focus = "opposite"
    else:
        print("\nSelected: Custom Vertex Pair")
        focus = "custom"

    # Step 0b: Analysis method (now RDFS only)
    print("\n--- Analysis Method ---")
    analysis_method = "rdfs"
    print("Using: RDFS (Randomized DFS) - Empirical simulation")
    print("Note: Focus is now exclusively on DFS analysis")

    # Step 1: Select graph type
    print("\n--- Graph Type Selection ---")

    # Validates opposite analysis is only for hypercubes
    if focus == "opposite":
        print("1. Hypercube (required for opposite vertex analysis)")
        graph_choice = 1
        graph_type = "hypercube"
        print("\nNote: Opposite vertex analysis only available for hypercubes")
        print("Selected: Hypercube")
    elif focus == "custom":
        # Custom vertex pair supports hypercube, petersen, triangular, torus, hexagonal
        print("1. Hypercube")
        print("2. Generalized Petersen")
        print("3. Triangular Lattice")
        print("4. Torus Grid")
        print("5. Hexagonal Lattice")

        graph_choice = get_user_choice([1, 2, 3, 4, 5])

        if graph_choice == 1:
            graph_type = "hypercube"
            print("\nSelected: Hypercube")
        elif graph_choice == 2:
            graph_type = "petersen"
            print("\nSelected: Generalized Petersen")
        elif graph_choice == 3:
            graph_type = "triangular"
            print("\nSelected: Triangular Lattice")
        elif graph_choice == 4:
            graph_type = "torus"
            print("\nSelected: Torus Grid")
        else:
            graph_type = "hexagonal"
            print("\nSelected: Hexagonal Lattice")
    else:
        print("1. Hypercube")
        print("2. Generalized Petersen")
        print("3. Triangular Lattice")
        print("4. Torus Grid")
        print("5. Hexagonal Lattice (Graphene)")
        print("6. Complete Graph")
        print("7. N-Dimensional Grid (3D-10D)")
        print("8. G(n,p) Random Graph")

        graph_choice = get_user_choice([1, 2, 3, 4, 5, 6, 7, 8])

        if graph_choice == 1:
            graph_type = "hypercube"
            print("\nSelected: Hypercube")
        elif graph_choice == 2:
            graph_type = "petersen"
            print("\nSelected: Generalized Petersen")
        elif graph_choice == 3:
            graph_type = "triangular"
            print("\nSelected: Triangular Lattice")
        elif graph_choice == 4:
            graph_type = "torus"
            print("\nSelected: Torus Grid")
        elif graph_choice == 5:
            graph_type = "hexagonal"
            print("\nSelected: Hexagonal Lattice (Graphene)")
        elif graph_choice == 6:
            graph_type = "complete"
            print("\nSelected: Complete Graph")
        elif graph_choice == 7:
            graph_type = "ndgrid"
            print("\nSelected: N-Dimensional Grid")
        else:
            graph_type = "gnp"
            print("\nSelected: G(n,p) Random Graph")

    # Step 2: Get graph parameters
    print("\n--- Graph Parameters ---")

    if graph_type == "hypercube":
        dimension = get_integer_input(
            "Enter hypercube dimension (3-20 supported): ", min_val=2, max_val=25
        )
        petersen_k = None
        lattice_rows = None
        lattice_cols = None
        grid_size = None
        gnp_p = None
        num_vertices = 2**dimension
        print(f"\nHypercube {dimension}D has {num_vertices} vertices")

        # Warns about memory usage for very large dimensions
        if dimension > 15:
            est_memory_mb = num_vertices * 8 / (1024 * 1024)  # Rough estimate
            print(f"  [WARNING] Warning: Large graph ({est_memory_mb:.1f} MB estimated memory)")
            print(f"  Recommended for HPC environments with sufficient RAM")
    elif graph_type == "petersen":
        dimension = get_integer_input(
            "Enter n (vertices per ring, n >= 3): ", min_val=3, max_val=20
        )
        petersen_k = get_integer_input(
            f"Enter k (skip parameter, 1 <= k < {dimension}): ",
            min_val=1, max_val=dimension-1
        )
        lattice_rows = None
        lattice_cols = None
        grid_size = None
        gnp_p = None
        num_vertices = 2 * dimension
        print(f"\nPetersen GP({dimension}, {petersen_k}) has {num_vertices} vertices")
    elif graph_type == "triangular":
        lattice_rows = get_integer_input(
            "Enter number of rows (3-50 recommended): ", min_val=3, max_val=100
        )
        lattice_cols = get_integer_input(
            "Enter number of columns (3-50 recommended): ", min_val=3, max_val=100
        )
        dimension = lattice_rows  # Used for later compatibility
        petersen_k = None
        grid_size = None
        gnp_p = None
        num_vertices = lattice_rows * lattice_cols
        print(f"\nTriangular Lattice {lattice_rows}×{lattice_cols} has {num_vertices} vertices")
        print(f"  Degree: 6 (each vertex has 6 neighbors)")
        print(f"  Topology: Periodic boundary conditions")

        # Warns about memory usage for very large lattices
        if num_vertices > 2500:
            print(f"  [WARNING] Warning: Large graph ({num_vertices} vertices)")
            print(f"  Consider smaller dimensions for faster results")
    elif graph_type == "torus":
        lattice_rows = get_integer_input(
            "Enter number of rows (3-50 recommended): ", min_val=3, max_val=100
        )
        lattice_cols = get_integer_input(
            "Enter number of columns (3-50 recommended): ", min_val=3, max_val=100
        )
        dimension = lattice_rows  # Used for later compatibility
        petersen_k = None
        grid_size = None
        gnp_p = None
        num_vertices = lattice_rows * lattice_cols
        print(f"\nTorus Grid {lattice_rows}×{lattice_cols} has {num_vertices} vertices")
        print(f"  Degree: 4 (each vertex has 4 neighbors)")
        print(f"  Topology: Periodic boundary conditions")

        # Warns about memory usage for very large lattices
        if num_vertices > 2500:
            print(f"  [WARNING] Warning: Large graph ({num_vertices} vertices)")
            print(f"  Consider smaller dimensions for faster results")
    elif graph_type == "hexagonal":
        lattice_rows = get_integer_input(
            "Enter number of rows (3-50 recommended): ", min_val=3, max_val=100
        )
        lattice_cols = get_integer_input(
            "Enter number of columns (3-50 recommended): ", min_val=3, max_val=100
        )
        dimension = lattice_rows  # Used for later compatibility
        petersen_k = None
        grid_size = None
        gnp_p = None
        num_vertices = lattice_rows * lattice_cols
        print(f"\nHexagonal Lattice {lattice_rows}×{lattice_cols} has {num_vertices} vertices")
        print(f"  Degree: 3 (honeycomb structure, like graphene)")
        print(f"  Topology: Periodic boundary conditions")

        # Warns about memory usage for very large lattices
        if num_vertices > 2500:
            print(f"  [WARNING] Warning: Large graph ({num_vertices} vertices)")
            print(f"  Consider smaller dimensions for faster results")
    elif graph_type == "complete":
        dimension = get_integer_input(
            "Enter number of vertices (n >= 2): ", min_val=2, max_val=100
        )
        petersen_k = None
        lattice_rows = None
        lattice_cols = None
        grid_size = None
        gnp_p = None
        num_vertices = dimension
        num_edges = num_vertices * (num_vertices - 1) // 2
        print(f"\nComplete Graph K_{num_vertices} has {num_vertices} vertices and {num_edges} edges")
        print(f"  Degree: {num_vertices - 1} (every vertex connects to all others)")
        print(f"  Diameter: 1 (shortest path between any two vertices)")

        # Warns about memory usage for large complete graphs
        if num_vertices > 50:
            print(f"  [WARNING] Warning: Large graph ({num_edges} edges)")
            print(f"  Consider smaller sizes for faster results")
    elif graph_type == "ndgrid":
        dimension = get_integer_input(
            "Enter number of dimensions (2-10 recommended): ", min_val=2, max_val=20
        )
        grid_size = get_integer_input(
            f"Enter grid size (points per dimension, 2-20 recommended): ", min_val=2, max_val=50
        )
        petersen_k = None
        lattice_rows = None
        lattice_cols = None
        gnp_p = None
        num_vertices = grid_size ** dimension
        degree = 2 * dimension
        print(f"\n{dimension}D Torus Grid {grid_size}^{dimension} has {num_vertices} vertices")
        print(f"  Degree: {degree} (constant, {dimension} dimensions × 2 directions)")
        print(f"  Topology: Periodic boundary conditions in all dimensions")

        # Warns about memory usage for large grids
        if num_vertices > 10000:
            print(f"  [WARNING] Warning: Large graph ({num_vertices} vertices)")
            print(f"  Consider smaller dimensions or grid size for faster results")
    elif graph_type == "gnp":
        dimension = get_integer_input(
            "Enter n (number of vertices, n >= 2): ", min_val=2, max_val=1000
        )
        gnp_p = get_float_input(
            "Enter p (edge probability, 0 < p < 1): ", min_val=0.001, max_val=0.999
        )
        petersen_k = None
        lattice_rows = None
        lattice_cols = None
        grid_size = None
        num_vertices = dimension
        expected_degree = (num_vertices - 1) * gnp_p
        print(f"\nG(n,p) Random Graph: n={num_vertices}, p={gnp_p:.3f}")
        print(f"  Expected degree: {expected_degree:.1f}")
        print(f"  Note: Graph structure varies with random seed")
        print(f"  Note: Graph may not be connected if p is too small")

        # Warns about connectivity issues
        if gnp_p < (2 * np.log(num_vertices) / num_vertices):
            print(f"  [WARNING] Warning: Low edge probability may result in disconnected graph")
            print(f"  Recommended: p >= {(2 * np.log(num_vertices) / num_vertices):.3f} for likely connectivity")

    # Step 2b: Get start and target vertices (only for custom mode)
    start_vertex = None
    target_vertex = None
    if focus == "custom":
        print("\n--- Vertex Selection ---")
        print("Specify the start and target vertices for analysis")
        print()

        if graph_type == "hypercube":
            # Hypercube vertices are binary tuples
            print(f"Hypercube vertices are {dimension}-bit binary tuples")
            print(f"Example: (0,0,0,0,0) for {dimension}D hypercube")
            print()

            # Gets start vertex
            while True:
                try:
                    start_input = input(f"Enter start vertex as comma-separated bits (e.g., 0,0,0,...): ").strip()
                    start_bits = [int(b.strip()) for b in start_input.split(",")]
                    if len(start_bits) != dimension:
                        print(f"Error: Must provide exactly {dimension} bits")
                        continue
                    if not all(b in [0, 1] for b in start_bits):
                        print("Error: All bits must be 0 or 1")
                        continue
                    start_vertex = tuple(start_bits)
                    print(f"Start vertex: {start_vertex}")
                    break
                except ValueError:
                    print("Error: Invalid input. Use comma-separated 0s and 1s")

            # Gets target vertex
            while True:
                try:
                    target_input = input(f"Enter target vertex as comma-separated bits (e.g., 1,1,0,...): ").strip()
                    target_bits = [int(b.strip()) for b in target_input.split(",")]
                    if len(target_bits) != dimension:
                        print(f"Error: Must provide exactly {dimension} bits")
                        continue
                    if not all(b in [0, 1] for b in target_bits):
                        print("Error: All bits must be 0 or 1")
                        continue
                    target_vertex = tuple(target_bits)
                    if target_vertex == start_vertex:
                        print("Warning: Start and target are the same vertex")
                        if not ask_yes_no("Continue anyway?", default=False):
                            continue
                    print(f"Target vertex: {target_vertex}")

                    # Computes and displays Hamming distance
                    hamming_dist = sum(s != t for s, t in zip(start_vertex, target_vertex))
                    print(f"Hamming distance: {hamming_dist}")
                    break
                except ValueError:
                    print("Error: Invalid input. Use comma-separated 0s and 1s")

        elif graph_type == "gnp":
            # G(n,p) vertices are integers 0 to n-1
            print(f"G(n,p) vertices are integers from 0 to {num_vertices - 1}")
            print()

            # Gets start vertex
            start_vertex = get_integer_input(
                f"Enter start vertex (0 to {num_vertices - 1}): ",
                min_val=0,
                max_val=num_vertices - 1
            )
            print(f"Start vertex: {start_vertex}")

            # Gets target vertex
            while True:
                target_vertex = get_integer_input(
                    f"Enter target vertex (0 to {num_vertices - 1}): ",
                    min_val=0,
                    max_val=num_vertices - 1
                )
                if target_vertex == start_vertex:
                    print("Warning: Start and target are the same vertex")
                    if not ask_yes_no("Continue anyway?", default=False):
                        continue
                print(f"Target vertex: {target_vertex}")
                break

        else:  # petersen
            # Petersen vertices are ('outer', i) or ('inner', i)
            print(f"Petersen GP({dimension}, {petersen_k}) has vertices as ('ring', index) tuples")
            print(f"  Outer ring: ('outer', 0) to ('outer', {dimension - 1})")
            print(f"  Inner ring: ('inner', 0) to ('inner', {dimension - 1})")
            print()

            # Gets start vertex
            while True:
                try:
                    ring = input("Enter start ring (outer/inner): ").strip().lower()
                    if ring not in ['outer', 'inner']:
                        print("Error: Ring must be 'outer' or 'inner'")
                        continue
                    idx = get_integer_input(
                        f"Enter start index (0 to {dimension - 1}): ",
                        min_val=0,
                        max_val=dimension - 1
                    )
                    start_vertex = (ring, idx)
                    print(f"Start vertex: {start_vertex}")
                    break
                except Exception as e:
                    print(f"Error: {e}")

            # Gets target vertex
            while True:
                try:
                    ring = input("Enter target ring (outer/inner): ").strip().lower()
                    if ring not in ['outer', 'inner']:
                        print("Error: Ring must be 'outer' or 'inner'")
                        continue
                    idx = get_integer_input(
                        f"Enter target index (0 to {dimension - 1}): ",
                        min_val=0,
                        max_val=dimension - 1
                    )
                    target_vertex = (ring, idx)

                    if target_vertex == start_vertex:
                        print("Warning: Start and target are the same vertex")
                        if not ask_yes_no("Continue anyway?", default=False):
                            continue

                    print(f"Target vertex: {target_vertex}")
                    break
                except Exception as e:
                    print(f"Error: {e}")

        else:  # triangular or torus
            # Both use (q, r) or (row, col) coordinate tuples
            if graph_type == "triangular":
                print(f"Triangular lattice vertices are (q, r) coordinate tuples")
                coord_names = ("q (column)", "r (row)")
            else:  # torus
                print(f"Torus grid vertices are (row, col) coordinate tuples")
                coord_names = ("row", "col")

            print(f"  {coord_names[0]}: 0 to {lattice_cols - 1}")
            print(f"  {coord_names[1]}: 0 to {lattice_rows - 1}")
            print()

            # Gets start vertex
            while True:
                try:
                    start_coord1 = get_integer_input(
                        f"Enter start {coord_names[0]} (0 to {lattice_cols - 1}): ",
                        min_val=0,
                        max_val=lattice_cols - 1
                    )
                    start_coord2 = get_integer_input(
                        f"Enter start {coord_names[1]} (0 to {lattice_rows - 1}): ",
                        min_val=0,
                        max_val=lattice_rows - 1
                    )
                    start_vertex = (start_coord1, start_coord2)
                    print(f"Start vertex: {start_vertex}")
                    break
                except Exception as e:
                    print(f"Error: {e}")

            # Gets target vertex
            while True:
                try:
                    target_coord1 = get_integer_input(
                        f"Enter target {coord_names[0]} (0 to {lattice_cols - 1}): ",
                        min_val=0,
                        max_val=lattice_cols - 1
                    )
                    target_coord2 = get_integer_input(
                        f"Enter target {coord_names[1]} (0 to {lattice_rows - 1}): ",
                        min_val=0,
                        max_val=lattice_rows - 1
                    )
                    target_vertex = (target_coord1, target_coord2)

                    if target_vertex == start_vertex:
                        print("Warning: Start and target are the same vertex")
                        if not ask_yes_no("Continue anyway?", default=False):
                            continue

                    print(f"Target vertex: {target_vertex}")
                    break
                except Exception as e:
                    print(f"Error: {e}")

    # Step 3: Get sample size
    print("\n--- Sampling Configuration ---")
    recommended = get_recommended_samples(graph_type, dimension, num_vertices)
    print(f"Recommended samples for {num_vertices} vertices:")
    print(f"  Quick test:    {max(1000, recommended // 10)}")
    print(f"  Standard:      {recommended}")
    print(f"  High accuracy: {recommended * 2}")

    num_samples = get_integer_input(
        f"\nEnter number of samples (recommended: {recommended}): ", min_val=100
    )

    # Step 4: Output options
    print("\n--- Output Options ---")
    output_options = select_output_options()

    # Step 5: Advanced options
    print("\n--- Advanced Options ---")
    configure_advanced = ask_yes_no(
        "Configure advanced options (seed, output directory)?", default=False
    )

    if configure_advanced:
        rng_seed = get_integer_input(
            "RNG seed for reproducibility", default=1832479182
        )
        output_dir = input("Output directory (default: data_output): ").strip()
        if not output_dir:
            output_dir = "data_output"
    else:
        rng_seed = 1832479182
        output_dir = "data_output"

    # Step 6: Confirm and run
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    if graph_type == "hypercube":
        print(f"Graph:          Hypercube {dimension}D ({num_vertices} vertices)")
    elif graph_type == "petersen":
        print(f"Graph:          Petersen GP({dimension}, {petersen_k}) ({num_vertices} vertices)")
    elif graph_type == "triangular":
        print(f"Graph:          Triangular Lattice {lattice_rows}×{lattice_cols} ({num_vertices} vertices)")
    elif graph_type == "torus":
        print(f"Graph:          Torus Grid {lattice_rows}×{lattice_cols} ({num_vertices} vertices)")
    elif graph_type == "hexagonal":
        print(f"Graph:          Hexagonal Lattice {lattice_rows}×{lattice_cols} ({num_vertices} vertices)")
    elif graph_type == "complete":
        num_edges = num_vertices * (num_vertices - 1) // 2
        print(f"Graph:          Complete Graph K_{num_vertices} ({num_vertices} vertices, {num_edges} edges)")
    elif graph_type == "ndgrid":
        degree = 2 * dimension
        print(f"Graph:          {dimension}D Torus Grid {grid_size}^{dimension} ({num_vertices} vertices, degree {degree})")
    elif graph_type == "gnp":
        expected_degree = (num_vertices - 1) * gnp_p
        print(f"Graph:          G(n,p) Random Graph: n={num_vertices}, p={gnp_p:.3f} (expected degree {expected_degree:.1f})")
    else:
        print(f"Graph:          {graph_type} ({num_vertices} vertices)")
    print(f"Samples:        {num_samples}")
    print(f"RNG Seed:       {rng_seed}")
    print(f"Output dir:     {output_dir}")
    print("\nFiles to generate:")
    print(f"  summary.txt:        Always")
    print(f"  data.csv:           {'Yes' if output_options['save_csv'] else 'No'}")
    print(f"  detailed_stats.txt: {'Yes' if output_options['save_detailed_stats'] else 'No'}")
    print(f"  visualization.png:  {'Yes' if output_options['save_plots'] else 'No'}")
    if output_options['export_formats']:
        print(f"  Additional:         {', '.join(output_options['export_formats'])}")
    print("=" * 70)

    if not ask_yes_no("\nProceed with experiment?", default=True):
        print("Experiment cancelled.")
        return

    # Step 7: Create config and run
    # Determine grid_size parameter
    if graph_type == "ndgrid":
        # grid_size already set from user input
        pass
    else:
        grid_size = None

    config = ExperimentConfig(
        graph_type=graph_type,
        dimension=dimension,
        petersen_k=petersen_k,
        lattice_rows=lattice_rows,
        lattice_cols=lattice_cols,
        grid_size=grid_size,
        gnp_p=gnp_p,
        num_samples=num_samples,
        rng_seed=rng_seed,
        output_dir=output_dir,
        save_plots=output_options['save_plots'],
        save_detailed_stats=output_options['save_detailed_stats'],
        save_csv=output_options['save_csv'],
        export_formats=output_options['export_formats'],
    )

    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT")
    print("=" * 70)
    print("This may take a moment...")
    print("Press Ctrl+C to cancel at any time.\n")

    # Runs experiment based on focus and method
    if focus == "opposite":
        # Opposite vertex analysis (hypercube only)
        from dfs_analyzer.experiments.opposite_runner import OppositeAnalysisRunner

        runner = OppositeAnalysisRunner()

        # Progress callback for RDFS
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

        try:
            results = runner.run(config, method="rdfs", progress_callback=progress_callback)
            print("\n")  # New line after progress bar
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("EXPERIMENT CANCELLED")
            print("=" * 70)
            print("Experiment was cancelled by user (Ctrl+C).")
            print("No results were saved.")
            return
        except Exception as e:
            print(f"\n\nError running experiment: {e}")
            import traceback
            traceback.print_exc()
            return

    elif focus == "neighbors":
        # Neighbor analysis
        from dfs_analyzer.experiments.neighbor_runner import NeighborAnalysisRunner

        runner = NeighborAnalysisRunner()

        # Progress callback for RDFS
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

        try:
            results = runner.run(config, method="rdfs", progress_callback=progress_callback)
            print("\n")  # New line after progress bar
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("EXPERIMENT CANCELLED")
            print("=" * 70)
            print("Experiment was cancelled by user (Ctrl+C).")
            print("No results were saved.")
            return
        except Exception as e:
            print(f"\n\nError running experiment: {e}")
            import traceback
            traceback.print_exc()
            return

    elif focus == "custom":
        # Custom vertex pair analysis
        from dfs_analyzer.experiments.custom_vertex_runner import CustomVertexRunner
        from dfs_analyzer.core.graphs import Hypercube, GeneralizedPetersen, TriangularLattice, TorusGrid, HexagonalLattice, CompleteGraph, NDGrid
        from dfs_analyzer.core.gnp_graph import generate_connected_gnp

        # Creates graph instance
        if graph_type == "hypercube":
            graph = Hypercube(dimension)
        elif graph_type == "petersen":
            graph = GeneralizedPetersen(dimension, petersen_k)
        elif graph_type == "triangular":
            graph = TriangularLattice(lattice_rows, lattice_cols)
        elif graph_type == "torus":
            graph = TorusGrid(lattice_rows, lattice_cols)
        elif graph_type == "hexagonal":
            graph = HexagonalLattice(lattice_rows, lattice_cols)
        elif graph_type == "complete":
            graph = CompleteGraph(dimension)
        elif graph_type == "ndgrid":
            graph = NDGrid(dimension, grid_size)
        else:  # gnp
            print(f"Generating connected G({dimension}, {gnp_p:.3f}) graph...")
            graph = generate_connected_gnp(dimension, gnp_p, rng_seed=rng_seed)

        runner = CustomVertexRunner()

        # Progress callback for RDFS
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

        try:
            results = runner.run(
                graph=graph,
                start_vertex=start_vertex,
                target_vertex=target_vertex,
                num_samples=num_samples,
                method="rdfs",
                rng_seed=rng_seed,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            print("\n")  # New line after progress bar
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("EXPERIMENT CANCELLED")
            print("=" * 70)
            print("Experiment was cancelled by user (Ctrl+C).")
            print("No results were saved.")
            return
        except Exception as e:
            print(f"\n\nError running experiment: {e}")
            import traceback
            traceback.print_exc()
            return

    else:
        # Full graph analysis
        # Progress callback for RDFS
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

        from dfs_analyzer.experiments.runner import ExperimentRunner
        runner = ExperimentRunner()
        try:
            results = runner.run(config, progress_callback=progress_callback)
            print("\n")  # New line after progress bar
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("EXPERIMENT CANCELLED")
            print("=" * 70)
            print("Experiment was cancelled by user (Ctrl+C).")
            print("No results were saved.")
            return
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
This tool validates the expected behavior for symmetric regular graphs.

EXPECTED BEHAVIOR:
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

A tool for empirically analyzing DFS behavior on symmetric
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
