"""
Streamlit web GUI for DFS Graph Analyzer.

Interactive web interface for running DFS experiments on symmetric regular graphs.
Works alongside the CLI - both interfaces use the same core logic.

HPC Usage:
    ssh -L 8501:localhost:8501 user@hpc.edu
    streamlit run dfs_analyzer/ui/streamlit_app.py
    # Then open http://localhost:8501 in your browser
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Adds parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dfs_analyzer.core.graphs import Hypercube, GeneralizedPetersen, CompleteGraph
from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner
# Removed: RandomWalkRunner (Laplacian analysis - out of scope)
from dfs_analyzer.experiments.neighbor_runner import NeighborAnalysisRunner
from dfs_analyzer.experiments.opposite_runner import OppositeAnalysisRunner
from dfs_analyzer.experiments.custom_vertex_runner import CustomVertexRunner
from dfs_analyzer.experiments.gnp_batch_runner import GNPBatchRunner


# Page configuration
st.set_page_config(
    page_title="DFS Graph Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application."""

    # Header
    st.title("📊 DFS Graph Analyzer")
    st.markdown("### Interactive tool for analyzing DFS behavior")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Analysis Type
        st.subheader("1. Analysis Type")
        analysis_type = st.selectbox(
            "Select analysis focus",
            [
                "Full Graph Analysis",
                "Immediate Neighbors",
                "Opposite Vertex (Hypercube only)",
                "Custom Vertex Pair"
            ],
            help="Choose what to analyze: all vertices, neighbors, opposite vertex, or custom pair"
        )

        # Analysis Method (simplified - RDFS only)
        st.subheader("2. Analysis Method")
        method = "rdfs"
        st.info("[OK] Using RDFS (Randomized DFS) sampling - empirical method")

        # Graph Type
        st.subheader("3. Graph Type")
        if "Opposite Vertex" in analysis_type:
            graph_type = "Hypercube"
            st.info("Opposite vertex analysis requires hypercube")
        elif "Custom Vertex" in analysis_type:
            graph_type = st.selectbox(
                "Select graph type",
                ["Hypercube", "Generalized Petersen", "Triangular Lattice", "Torus Grid", "Hexagonal Lattice", "Complete Graph", "N-Dimensional Grid", "G(n,p) Random"]
            )
        else:
            graph_type = st.selectbox(
                "Select graph type",
                ["Hypercube", "Generalized Petersen", "Triangular Lattice", "Torus Grid", "Hexagonal Lattice", "Complete Graph", "N-Dimensional Grid", "G(n,p) Random"]
            )

        # Graph Parameters
        st.subheader("4. Graph Parameters")

        if graph_type == "G(n,p) Random":
            # G(n,p) parameters
            dimension = st.number_input("Number of vertices (n)", min_value=2, max_value=1000, value=30)
            gnp_p = st.slider("Edge probability (p)", min_value=0.001, max_value=0.999, value=0.300, step=0.001, format="%.3f")
            threshold = (np.log(dimension) + 3) / dimension if dimension > 1 else 0.5
            st.info(f"Connectivity threshold: p ≥ {threshold:.4f}")
            if gnp_p < threshold:
                st.warning("[WARNING] Low p may result in disconnected graph")

            petersen_k = None
            lattice_rows = None
            lattice_cols = None
            grid_size = None
            num_vertices = dimension
            expected_degree = (num_vertices - 1) * gnp_p
            st.info(f"Expected degree: {expected_degree:.1f}")

        elif graph_type == "Hypercube":
            dimension = st.slider("Dimension (d)", min_value=2, max_value=20, value=5)
            num_vertices = 2 ** dimension
            st.info(f"Vertices: {num_vertices}")
            petersen_k = None
            lattice_rows = None
            lattice_cols = None
            grid_size = None
            gnp_p = None

        elif graph_type == "Generalized Petersen":
            dimension = st.slider("Ring size (n)", min_value=3, max_value=20, value=5)
            petersen_k = st.slider("Skip parameter (k)", min_value=1, max_value=dimension-1, value=2)
            num_vertices = 2 * dimension
            st.info(f"Vertices: {num_vertices}")
            lattice_rows = None
            lattice_cols = None
            grid_size = None
            gnp_p = None

        elif graph_type == "Triangular Lattice":
            lattice_rows = st.slider("Number of rows", min_value=3, max_value=50, value=5)
            lattice_cols = st.slider("Number of columns", min_value=3, max_value=50, value=5)
            num_vertices = lattice_rows * lattice_cols
            st.info(f"Vertices: {num_vertices} (degree 6, torus topology)")
            dimension = lattice_rows  # For compatibility
            petersen_k = None
            grid_size = None
            gnp_p = None
            
            

        elif graph_type == "Torus Grid":
            lattice_rows = st.slider("Number of rows", min_value=3, max_value=50, value=5)
            lattice_cols = st.slider("Number of columns", min_value=3, max_value=50, value=5)
            num_vertices = lattice_rows * lattice_cols
            st.info(f"Vertices: {num_vertices} (degree 4, torus topology)")
            dimension = lattice_rows  # For compatibility
            petersen_k = None
            grid_size = None
            gnp_p = None
            
            

        elif graph_type == "Hexagonal Lattice":
            lattice_rows = st.slider("Number of rows", min_value=3, max_value=50, value=5)
            lattice_cols = st.slider("Number of columns", min_value=3, max_value=50, value=5)
            num_vertices = lattice_rows * lattice_cols
            st.info(f"Vertices: {num_vertices} (degree 3, honeycomb/graphene structure)")
            dimension = lattice_rows  # For compatibility
            petersen_k = None
            grid_size = None
            gnp_p = None
            
            

        elif graph_type == "Complete Graph":
            dimension = st.slider("Number of vertices (n)", min_value=2, max_value=100, value=10)
            num_vertices = dimension
            num_edges = num_vertices * (num_vertices - 1) // 2
            st.info(f"Vertices: {num_vertices}, Edges: {num_edges} (degree {num_vertices - 1}, diameter 1)")
            petersen_k = None
            lattice_rows = None
            lattice_cols = None
            grid_size = None
            gnp_p = None
            
            

        else:  # N-Dimensional Grid
            dimension = st.slider("Number of dimensions (d)", min_value=2, max_value=20, value=3)
            grid_size = st.slider("Grid size (points per dimension)", min_value=2, max_value=50, value=10)
            num_vertices = grid_size ** dimension
            degree = 2 * dimension
            st.info(f"Vertices: {num_vertices} ({grid_size}^{dimension}), Degree: {degree}, Torus topology")
            petersen_k = None
            lattice_rows = None
            lattice_cols = None
            gnp_p = None
            
            

        # Custom Vertex Selection
        start_vertex = None
        target_vertex = None
        if "Custom Vertex" in analysis_type:
            st.subheader("5. Vertex Selection")

            if graph_type == "Hypercube":
                st.markdown("**Enter vertices as comma-separated bits**")
                start_input = st.text_input("Start vertex (e.g., 0,0,0,0,0)", value="0," * dimension)
                target_input = st.text_input("Target vertex (e.g., 1,0,1,0,1)", value="1,0," + "1,0," * (dimension//2))

                try:
                    start_vertex = tuple(int(b.strip()) for b in start_input.strip(',').split(','))
                    target_vertex = tuple(int(b.strip()) for b in target_input.strip(',').split(','))

                    if len(start_vertex) == dimension and len(target_vertex) == dimension:
                        hamming = sum(s != t for s, t in zip(start_vertex, target_vertex))
                        st.success(f"[OK] Hamming distance: {hamming}")
                    else:
                        st.error(f"[ERROR] Need exactly {dimension} bits per vertex")
                        start_vertex = None
                        target_vertex = None
                except:
                    st.error("[ERROR] Invalid format. Use comma-separated 0s and 1s")
                    start_vertex = None
                    target_vertex = None

            elif graph_type == "Generalized Petersen":
                st.markdown("**Select vertices from rings**")
                col1, col2 = st.columns(2)
                with col1:
                    start_ring = st.selectbox("Start ring", ["outer", "inner"], key="start_ring")
                    start_idx = st.number_input("Start index", 0, dimension-1, 0, key="start_idx")
                with col2:
                    target_ring = st.selectbox("Target ring", ["outer", "inner"], key="target_ring")
                    target_idx = st.number_input("Target index", 0, dimension-1, 0, key="target_idx")

                start_vertex = (start_ring, int(start_idx))
                target_vertex = (target_ring, int(target_idx))
                st.success(f"[OK] Start: {start_vertex}, Target: {target_vertex}")

            elif graph_type == "Triangular Lattice":
                st.markdown("**Enter coordinates (q, r)**")
                st.info(f"q: 0 to {lattice_cols-1} (columns), r: 0 to {lattice_rows-1} (rows)")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Start Vertex**")
                    start_q = st.number_input("Start q (column)", 0, lattice_cols-1, 0, key="start_q")
                    start_r = st.number_input("Start r (row)", 0, lattice_rows-1, 0, key="start_r")
                with col2:
                    st.markdown("**Target Vertex**")
                    target_q = st.number_input("Target q (column)", 0, lattice_cols-1, lattice_cols-1, key="target_q")
                    target_r = st.number_input("Target r (row)", 0, lattice_rows-1, lattice_rows-1, key="target_r")

                start_vertex = (int(start_q), int(start_r))
                target_vertex = (int(target_q), int(target_r))
                st.success(f"[OK] Start: {start_vertex}, Target: {target_vertex}")

            elif graph_type == "Torus Grid":
                st.markdown("**Enter coordinates (row, col)**")
                st.info(f"row: 0 to {lattice_rows-1}, col: 0 to {lattice_cols-1}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Start Vertex**")
                    start_row = st.number_input("Start row", 0, lattice_rows-1, 0, key="start_row_torus")
                    start_col = st.number_input("Start col", 0, lattice_cols-1, 0, key="start_col_torus")
                with col2:
                    st.markdown("**Target Vertex**")
                    target_row = st.number_input("Target row", 0, lattice_rows-1, lattice_rows-1, key="target_row_torus")
                    target_col = st.number_input("Target col", 0, lattice_cols-1, lattice_cols-1, key="target_col_torus")

                start_vertex = (int(start_row), int(start_col))
                target_vertex = (int(target_row), int(target_col))
                st.success(f"[OK] Start: {start_vertex}, Target: {target_vertex}")

            elif graph_type == "Hexagonal Lattice":
                st.markdown("**Enter coordinates (row, col)**")
                st.info(f"row: 0 to {lattice_rows-1}, col: 0 to {lattice_cols-1}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Start Vertex**")
                    start_row = st.number_input("Start row", 0, lattice_rows-1, 0, key="start_row_hex")
                    start_col = st.number_input("Start col", 0, lattice_cols-1, 0, key="start_col_hex")
                with col2:
                    st.markdown("**Target Vertex**")
                    target_row = st.number_input("Target row", 0, lattice_rows-1, lattice_rows-1, key="target_row_hex")
                    target_col = st.number_input("Target col", 0, lattice_cols-1, lattice_cols-1, key="target_col_hex")

                start_vertex = (int(start_row), int(start_col))
                target_vertex = (int(target_row), int(target_col))
                st.success(f"[OK] Start: {start_vertex}, Target: {target_vertex}")

            else:  # Complete Graph, N-Dimensional Grid, or G(n,p) Random
                st.markdown("**Enter vertex labels (integers)**")
                st.info(f"Vertices labeled: 0 to {num_vertices-1}")
                col1, col2 = st.columns(2)
                with col1:
                    start_vertex = st.number_input("Start vertex", 0, num_vertices-1, 0, key="start_int")
                with col2:
                    target_vertex = st.number_input("Target vertex", 0, num_vertices-1, min(1, num_vertices-1), key="target_int")

                st.success(f"[OK] Start: {start_vertex}, Target: {target_vertex}")

        # Sampling Configuration
        st.subheader("6. Sampling" if "Custom" not in analysis_type else "6. Sampling")
        if method == "rdfs":
            if "G(n,p)" in analysis_type and "Batch" in analysis_type:
                samples_per_graph = st.number_input(
                    "Samples per graph",
                    min_value=100,
                    max_value=100000,
                    value=max(1000, num_vertices * 10)
                )
                num_samples = samples_per_graph
            else:
                recommended = num_vertices * 2000 if graph_type == "Hypercube" else num_vertices * 1000
                num_samples = st.number_input(
                    "Number of RDFS samples",
                    min_value=100,
                    max_value=10000000,
                    value=min(recommended, 10000),
                    help=f"Recommended: {recommended}"
                )
        else:
            num_samples = st.number_input(
                "Number of samples",
                min_value=100,
                max_value=10000000,
                value=10000,
                help="Number of RDFS runs"
            )

        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            rng_seed = st.number_input("RNG Seed", value=1832479182)
            output_dir = st.text_input("Output Directory", value="data_output")

            st.markdown("**Output Files**")
            save_csv = st.checkbox("Save CSV", value=False)
            save_detailed = st.checkbox("Save detailed stats", value=False)
            save_plots = st.checkbox("Save plots", value=False)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈 Experiment")

        # Displays current configuration
        with st.expander("📋 Current Configuration", expanded=True):
            if graph_type == "G(n,p)" or graph_type == "G(n,p) Random":
                st.markdown(f"""
                - **Graph**: G({n_gnp}, {p_gnp:.4f})
                - **Expected edges**: {p_gnp * n_gnp * (n_gnp - 1) / 2:.1f}
                """)
                if "Batch" in analysis_type:
                    st.markdown(f"- **Number of graphs**: {num_graphs}")
                    st.markdown(f"- **Total RDFS runs**: {num_graphs * num_samples}")
            else:
                if graph_type == 'Hypercube':
                    graph_desc = f"Hypercube ({dimension}D)"
                elif graph_type == 'Generalized Petersen':
                    graph_desc = f"Petersen GP({dimension},{petersen_k})"
                else:  # Triangular Lattice
                    graph_desc = f"Triangular Lattice {lattice_rows}×{lattice_cols}"

                st.markdown(f"""
                - **Graph**: {graph_desc}
                - **Vertices**: {num_vertices}
                - **Analysis**: {analysis_type}
                - **Method**: {method.upper()}
                - **Samples**: {num_samples}
                """)

            if start_vertex is not None:
                st.markdown(f"- **Start**: {start_vertex}")
                st.markdown(f"- **Target**: {target_vertex}")

        # Runs button
        run_button = st.button("🚀 Run Experiment", type="primary", use_container_width=True)

        if run_button:
            # Validates custom vertex inputs
            if "Custom Vertex" in analysis_type and (start_vertex is None or target_vertex is None):
                st.error("[ERROR] Please provide valid start and target vertices")
                return

            # Creates results container
            results_container = st.container()

            with results_container:
                st.markdown("---")

                # Note: Progress callbacks don't work reliably in Streamlit with multiprocessing
                # Using spinner instead for cleaner UX without threading warnings

                try:
                    with st.spinner("[RUNNING] Running experiment... Click 'Stop' button (top right) to cancel."):
                        # Runs appropriate experiment based on configuration
                        if "G(n,p)" in analysis_type and "Batch" in analysis_type:
                            # G(n,p) batch mode
                            runner = GNPBatchRunner()

                            # Note: No progress callback to avoid threading warnings
                            results = runner.run(
                                n=n_gnp,
                                p=p_gnp,
                                num_graphs=num_graphs,
                                num_samples_per_graph=num_samples,
                                rng_seed=rng_seed,
                                output_dir=output_dir,
                                progress_callback=None
                            )

                        elif "Custom Vertex" in analysis_type:
                            # Custom vertex pair
                            if graph_type == "Hypercube":
                                graph = Hypercube(dimension)
                            elif graph_type == "Generalized Petersen":
                                graph = GeneralizedPetersen(dimension, petersen_k)
                            elif graph_type == "Triangular Lattice":
                                from dfs_analyzer.core.graphs import TriangularLattice
                                graph = TriangularLattice(lattice_rows, lattice_cols)
                            elif graph_type == "Torus Grid":
                                from dfs_analyzer.core.graphs import TorusGrid
                                graph = TorusGrid(lattice_rows, lattice_cols)
                            elif graph_type == "Hexagonal Lattice":
                                from dfs_analyzer.core.graphs import HexagonalLattice
                                graph = HexagonalLattice(lattice_rows, lattice_cols)
                            elif graph_type == "Complete Graph":
                                from dfs_analyzer.core.graphs import CompleteGraph
                                graph = CompleteGraph(dimension)
                            elif graph_type == "N-Dimensional Grid":
                                from dfs_analyzer.core.graphs import NDGrid
                                graph = NDGrid(dimension, grid_size)
                            else:  # G(n,p) Random
                                from dfs_analyzer.core.gnp_graph import generate_connected_gnp
                                with st.spinner(f"Generating connected G({dimension}, {gnp_p:.3f}) graph..."):
                                    graph = generate_connected_gnp(dimension, gnp_p, rng_seed=rng_seed)

                            runner = CustomVertexRunner()

                            # Note: No progress callback to avoid threading warnings
                            results = runner.run(
                                graph=graph,
                                start_vertex=start_vertex,
                                target_vertex=target_vertex,
                                num_samples=num_samples,
                                method=method,
                                rng_seed=rng_seed,
                                output_dir=output_dir,
                                progress_callback=None
                            )

                        elif "Opposite" in analysis_type:
                            # Opposite vertex (hypercube only)
                            config = ExperimentConfig(
                                graph_type="hypercube",
                                dimension=dimension,
                                num_samples=num_samples,
                                rng_seed=rng_seed,
                                output_dir=output_dir,
                                save_csv=save_csv,
                                save_detailed_stats=save_detailed,
                                save_plots=save_plots
                            )

                            runner = OppositeAnalysisRunner()

                            # Note: No progress callback to avoid threading warnings
                            results = runner.run(
                                config,
                                method=method,
                                progress_callback=None
                            )

                        elif "Neighbors" in analysis_type:
                            # Immediate neighbors
                            if graph_type == "Hypercube":
                                config_graph_type = "hypercube"
                            elif graph_type == "Generalized Petersen":
                                config_graph_type = "petersen"
                            elif graph_type == "Triangular Lattice":
                                config_graph_type = "triangular"
                            elif graph_type == "Torus Grid":
                                config_graph_type = "torus"
                            elif graph_type == "Hexagonal Lattice":
                                config_graph_type = "hexagonal"
                            elif graph_type == "Complete Graph":
                                config_graph_type = "complete"
                            elif graph_type == "N-Dimensional Grid":
                                config_graph_type = "ndgrid"
                            else:  # G(n,p) Random
                                config_graph_type = "gnp"

                            config = ExperimentConfig(
                                graph_type=config_graph_type,
                                dimension=dimension,
                                petersen_k=petersen_k,
                                lattice_rows=lattice_rows,
                                lattice_cols=lattice_cols,
                                grid_size=grid_size if config_graph_type == "ndgrid" else None,
                                gnp_p=gnp_p if config_graph_type == "gnp" else None,
                                num_samples=num_samples,
                                rng_seed=rng_seed,
                                output_dir=output_dir,
                                save_csv=save_csv,
                                save_detailed_stats=save_detailed,
                                save_plots=save_plots
                            )

                            runner = NeighborAnalysisRunner()

                            # Note: No progress callback to avoid threading warnings
                            results = runner.run(
                                config,
                                method=method,
                                progress_callback=None
                            )

                        else:
                            # Full graph analysis
                            if graph_type == "Hypercube":
                                config_graph_type = "hypercube"
                            elif graph_type == "Generalized Petersen":
                                config_graph_type = "petersen"
                            elif graph_type == "Triangular Lattice":
                                config_graph_type = "triangular"
                            elif graph_type == "Torus Grid":
                                config_graph_type = "torus"
                            elif graph_type == "Hexagonal Lattice":
                                config_graph_type = "hexagonal"
                            elif graph_type == "Complete Graph":
                                config_graph_type = "complete"
                            elif graph_type == "N-Dimensional Grid":
                                config_graph_type = "ndgrid"
                            else:  # G(n,p) Random
                                config_graph_type = "gnp"

                            config = ExperimentConfig(
                                graph_type=config_graph_type,
                                dimension=dimension,
                                petersen_k=petersen_k,
                                lattice_rows=lattice_rows,
                                lattice_cols=lattice_cols,
                                grid_size=grid_size if config_graph_type == "ndgrid" else None,
                                gnp_p=gnp_p if config_graph_type == "gnp" else None,
                                num_samples=num_samples,
                                rng_seed=rng_seed,
                                output_dir=output_dir,
                                save_csv=save_csv,
                                save_detailed_stats=save_detailed,
                                save_plots=save_plots
                            )

                            # Always use RDFS
                            runner = ExperimentRunner()

                            # Note: No progress callback to avoid threading warnings
                            results = runner.run(config, progress_callback=None)

                    # Displays results
                    st.success("✅ Experiment Complete!")
                    st.markdown("---")

                    # Results summary
                    st.subheader("📊 Results")
                    st.text(results.get_summary())

                    # Output path
                    st.info(f"💾 Results saved to: `{results.output_path}`")

                except KeyboardInterrupt:
                    st.warning("[WARNING] Experiment was cancelled by user.")
                    st.info("No results were saved.")
                except Exception as e:
                    st.error(f"[ERROR] Error running experiment: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        st.header("📚 Info")

        st.markdown("""
        ### Expected DFS Behavior

        For large symmetric regular graphs, the average discovery number
        of a vertex in randomized DFS tends to **(n-1)/2**, where n is
        the number of vertices.

        ### Graph Types

        **Hypercube**: d-dimensional binary graph
        - Vertices: 2^d binary tuples
        - Edges: Hamming distance 1
        - Highly symmetric

        **Petersen**: GP(n,k) graphs
        - Two rings of n vertices
        - Regular degree 3
        - Skip parameter k

        **Triangular Lattice**: 2D tiling
        - Rows × Cols vertices
        - Regular degree 6
        - Torus topology (periodic boundaries)

        **Torus Grid**: 2D grid
        - Rows × Cols vertices
        - Regular degree 4
        - Torus topology (periodic boundaries)

        **G(n,p)**: Random graphs
        - NOT regular/symmetric
        - Each edge appears with probability p
        - Exploratory analysis

        ### Method

        **RDFS (Randomized DFS)**: Runs multiple randomized DFS simulations
        - Empirical sampling approach
        - Tests the expected behavior through repeated trials
        - Provides mean, variance, and statistical validation

        ### HPC Usage

        ```bash
        ssh -L 8501:localhost:8501 user@hpc.edu
        streamlit run run_gui.py
        # Open: http://localhost:8501
        ```
        """)


if __name__ == "__main__":
    main()
