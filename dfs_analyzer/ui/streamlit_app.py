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

from dfs_analyzer.core.graphs import Hypercube, GeneralizedPetersen, CompleteGraph, RandomRegularGraph
from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner
# Removed: RandomWalkRunner (Laplacian analysis - out of scope)
from dfs_analyzer.experiments.neighbor_runner import NeighborAnalysisRunner
from dfs_analyzer.experiments.opposite_runner import OppositeAnalysisRunner
from dfs_analyzer.experiments.custom_vertex_runner import CustomVertexRunner
from dfs_analyzer.experiments.gnp_batch_runner import GNPBatchRunner


# Page configuration
st.set_page_config(
    page_title="Random DFS Graph Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application."""

    # Header
    st.title("📊 Random DFS Graph Analyzer")
    st.markdown("### Interactive tool for studying randomized DFS behaviour on graphs")
    st.markdown("---")

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🧪 Experiments", "📈 Analysis Tools", "📚 Documentation"])

    with tab1:
        run_experiments_tab()

    with tab2:
        run_analysis_tools_tab()

    with tab3:
        show_documentation_tab()


def run_experiments_tab():
    """Main experiments tab content."""
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
                "Custom Vertex Pair",
                "Three-Vertex Distribution",
            ],
            help="Choose what to analyze: all vertices, neighbors, opposite vertex, custom pair, or three-vertex histograms"
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
                ["Hypercube", "Generalized Petersen", "Triangular Lattice", "Torus Grid", "Hexagonal Lattice", "Complete Graph", "N-Dimensional Grid", "G(n,p) Random", "Random d-Regular"]
            )
        else:
            graph_type = st.selectbox(
                "Select graph type",
                ["Hypercube", "Generalized Petersen", "Triangular Lattice", "Torus Grid", "Hexagonal Lattice", "Complete Graph", "N-Dimensional Grid", "G(n,p) Random", "Random d-Regular"]
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
            
            

        elif graph_type == "Random d-Regular":
            dimension = st.number_input("Number of vertices (n)", min_value=2, max_value=1000, value=128)
            degree = st.slider("Degree (d)", min_value=2, max_value=min(dimension-1, 20), value=6)

            # Validate n*d is even
            if (dimension * degree) % 2 != 0:
                st.error(f"⚠️ n×d must be even! Current: {dimension}×{degree}={dimension*degree}")
                st.info("Fix: Use even n with any d, or odd n with even d")

            petersen_k = degree
            lattice_rows = None
            lattice_cols = None
            grid_size = None
            gnp_p = None
            num_vertices = dimension
            num_edges = (num_vertices * degree) // 2
            st.info(f"Random {degree}-regular graph → {num_vertices} vertices, {num_edges} edges")

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

        # Three-vertex vertex selection
        three_vertices = []
        three_labels   = []
        if "Three-Vertex" in analysis_type:
            st.subheader("5. Three Vertices")
            st.markdown("Pick three vertices at different BFS distances (near, mid, far).")
            for i in range(3):
                st.markdown(f"**Vertex {i+1}**")
                if graph_type == "Hypercube":
                    raw = st.text_input(
                        f"Vertex {i+1} bits (comma-separated, length {dimension})",
                        value=",".join(["0"] * dimension),
                        key=f"tv_hc_{i}"
                    )
                    try:
                        bits = tuple(int(b.strip()) for b in raw.split(','))
                        if len(bits) == dimension:
                            three_vertices.append(bits)
                        else:
                            three_vertices.append(None)
                    except Exception:
                        three_vertices.append(None)
                elif graph_type in ("Triangular Lattice", "Torus Grid", "Hexagonal Lattice"):
                    c1, c2 = st.columns(2)
                    with c1:
                        row_v = st.number_input(f"Row", 0, lattice_rows-1, 0, key=f"tv_row_{i}")
                    with c2:
                        col_v = st.number_input(f"Col", 0, lattice_cols-1, i, key=f"tv_col_{i}")
                    three_vertices.append((int(row_v), int(col_v)))
                elif graph_type == "Generalized Petersen":
                    ring_v = st.selectbox(f"Ring", ["outer", "inner"], key=f"tv_ring_{i}")
                    idx_v  = st.number_input(f"Index", 0, dimension-1, 0, key=f"tv_idx_{i}")
                    three_vertices.append((ring_v, int(idx_v)))
                else:
                    v = st.number_input(f"Vertex index", 0, num_vertices-1, min(i, num_vertices-1),
                                        key=f"tv_int_{i}")
                    three_vertices.append(int(v))
                lbl = st.text_input(f"Label", value=f"Vertex {i+1}", key=f"tv_lbl_{i}")
                three_labels.append(lbl)

        # Sampling Configuration
        st.subheader("6. Sampling")
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
                num_samples = st.number_input(
                    "Number of RDFS samples",
                    min_value=100,
                    max_value=10000000,
                    value=10000,
                    step=1000,
                    help="Quick: 1k-5k | Standard: 10k-50k | High accuracy: 100k+"
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
                        if "Three-Vertex" in analysis_type:
                            # Build graph
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
                            elif graph_type == "Random d-Regular":
                                from dfs_analyzer.core.graphs import RandomRegularGraph
                                graph = RandomRegularGraph(n=dimension, degree=petersen_k, seed=rng_seed)
                            else:
                                from dfs_analyzer.core.gnp_graph import generate_connected_gnp
                                graph = generate_connected_gnp(dimension, gnp_p, rng_seed=rng_seed)

                            if None in three_vertices:
                                st.error("[ERROR] One or more vertex inputs are invalid.")
                                st.stop()

                            from dfs_analyzer.experiments.three_vertex_runner import ThreeVertexRunner
                            tv_runner = ThreeVertexRunner()
                            tv_result = tv_runner.run(
                                graph=graph,
                                vertices=three_vertices,
                                labels=three_labels,
                                n_samples=num_samples,
                                rng_seed=rng_seed,
                                output_dir=output_dir,
                                graph_name=graph_type,
                            )

                            st.success("✅ Three-Vertex Distribution Complete!")
                            st.markdown("---")
                            st.subheader("📊 Summary Statistics")
                            for lbl, stats in tv_result['summary_stats'].items():
                                st.markdown(
                                    f"**{lbl}** — mean={stats['mean']:.1f}, "
                                    f"std={stats['std']:.1f}, "
                                    f"min={stats['min']}, "
                                    f"deviation from (n-1)/2 = {stats['deviation_from_half']:+.1f}"
                                )
                            img_path = Path(output_dir) / 'three_vertex_distributions.png'
                            if img_path.exists():
                                st.image(str(img_path), use_column_width=True)
                            st.info(f"💾 Saved to: `{output_dir}/three_vertex_distributions.png`")

                        elif "G(n,p)" in analysis_type and "Batch" in analysis_type:
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
        st.header("📚 Quick Info")
        st.markdown("""
        ### Expected Behavior
        For large symmetric regular graphs, average discovery
        number tends to **(n-1)/2**.

        ### Method
        **RDFS**: Runs multiple randomized DFS simulations
        - Empirical sampling approach
        - Provides mean, variance, statistical validation

        See **Documentation** tab for complete info.
        """)


def run_analysis_tools_tab():
    """Analysis tools tab for formula predictions and comparisons."""
    import subprocess
    from pathlib import Path

    st.header("📈 Analysis Tools")
    st.markdown("""
    Advanced analysis capabilities beyond basic experiments.
    All tools use existing experimental data or predictive formulas.
    """)
    st.markdown("---")

    # Get base directory
    base_dir = Path.cwd()

    # Tool selection
    tool = st.selectbox(
        "Select Analysis Tool",
        [
            "Formula Predictions (Hypercube)",
            "BFS vs DFS Comparison",
            "Confidence Intervals",
            "Visualize Formula System",
            "Validate Formulas",
            "Layer Exclusion Analysis",
            "─── Post-Experiment Analyses ───",
            "Deviation Analysis",
            "Sigmoid Model Fitting",
            "Layer Variance Analysis",
        ]
    )

    if tool == "Formula Predictions (Hypercube)":
        st.subheader("🔮 Formula Predictions")
        st.markdown("""
        Instantly predict layer-specific mean discovery numbers for hypercubes
        **without running experiments** using our validated general formula.

        **Formula**: `Mean_layer(L, d) = (n-1)/2 + a(d)·L³ + b(d)·L² + c(d)·L + d(d)`

        **Validated Range**: 3D-13D (6.52% average error, 95% CI: ±0.21%)
        """)

        col1, col2 = st.columns(2)

        with col1:
            dimension = st.number_input("Dimension (d)", min_value=3, max_value=20, value=10)
            layer = st.number_input("Layer (L)", min_value=1, max_value=dimension, value=5)

        with col2:
            if dimension < 3 or dimension > 13:
                st.warning(f"⚠️ {dimension}D is outside validated range (3D-13D). Prediction may be less accurate.")
            else:
                st.success(f"✅ {dimension}D is within validated range")

        if st.button("🔮 Predict", type="primary"):
            try:
                # Import formula function
                sys.path.insert(0, str(base_dir))
                from hypercube_formula_split import hypercube_coefficients

                # Get coefficients
                a, b, c, d = hypercube_coefficients(dimension)

                # Calculate prediction
                n = 2**dimension
                expected = (n-1)/2
                deviation = a*layer**3 + b*layer**2 + c*layer + d
                predicted = expected + deviation

                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Mean", f"{expected:.2f}")
                with col2:
                    st.metric("Deviation", f"{deviation:+.2f}")
                with col3:
                    st.metric("Predicted Mean", f"{predicted:.2f}")

                st.markdown("---")
                st.info(f"💡 This prediction was instant - no RDFS sampling needed!")

                # Show coefficients
                with st.expander("📐 Coefficient Values"):
                    st.code(f"""
a(d={dimension}) = {a:.10e}
b(d={dimension}) = {b:.10e}
c(d={dimension}) = {c:.10e}
d(d={dimension}) = {d:.10e}
                    """)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure hypercube_formula_split.py is in the project directory.")

        st.markdown("---")
        st.markdown("**Want to see full demonstration?** Run the script below:")
        if st.button("Run demo_formula_prediction.py"):
            with st.spinner("Running analysis..."):
                result = subprocess.run(
                    ["python3", "demo_formula_prediction.py"],
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✓ Analysis complete!")
                    with st.expander("Show Output"):
                        st.text(result.stdout)
                else:
                    st.error("Analysis encountered an error")
                    st.code(result.stderr)

    elif tool == "BFS vs DFS Comparison":
        st.subheader("🔀 BFS vs DFS Numbering Comparison")
        st.markdown("""
        Compares BFS (breadth-first) and DFS (depth-first) numbering schemes
        across 7 different graph types.

        **Tests**: Complete, Star, Path, Hypercube, Petersen, Torus, Triangular

        **Key Finding**: Divergence from **branching**, not diameter!
        - Path graph: diameter 9, 0% divergence (no branching)
        - Hypercube: diameter 4, 56% divergence (high branching)
        """)

        if st.button("▶️ Run Comparison", type="primary"):
            with st.spinner("Running comparison (may take 1-2 minutes)..."):
                result = subprocess.run(
                    ["python3", "compare_bfs_dfs_numbering.py"],
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✓ Analysis complete!")
                    st.info("Results saved to: analysis/bfs_dfs_comparison.png")

                    # Try to display the image
                    img_path = base_dir / "analysis" / "bfs_dfs_comparison.png"
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)

                    with st.expander("Show Full Output"):
                        st.text(result.stdout)
                else:
                    st.error("Analysis encountered an error")
                    st.code(result.stderr)

    elif tool == "Confidence Intervals":
        st.subheader("📊 Confidence Intervals")
        st.markdown("""
        Calculates prediction uncertainty using error propagation from
        coefficient formula uncertainties.

        **Shows**: 68%, 95%, and 99% confidence intervals for predictions

        **Result**: 95% confidence within ±0.21% of expected mean (excellent!)
        """)

        if st.button("📊 Calculate Intervals", type="primary"):
            with st.spinner("Calculating confidence intervals..."):
                result = subprocess.run(
                    ["python3", "calculate_confidence_intervals.py"],
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✓ Analysis complete!")
                    with st.expander("Show Output"):
                        st.text(result.stdout)
                else:
                    st.error("Analysis encountered an error")
                    st.code(result.stderr)

    elif tool == "Visualize Formula System":
        st.subheader("📈 Visualize Formula System")
        st.markdown("""
        Generates comprehensive 6-panel visualization showing:
        - Coefficient formulas across dimensions
        - Validation errors by dimension
        - Confidence intervals by layer
        - Predicted vs actual comparison
        """)

        if st.button("🎨 Generate Visualization", type="primary"):
            with st.spinner("Generating visualization..."):
                result = subprocess.run(
                    ["python3", "visualize_formula_system.py"],
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✓ Visualization generated!")
                    st.info("Saved to: analysis/general_formula_system.png")

                    # Try to display the image
                    img_path = base_dir / "analysis" / "general_formula_system.png"
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)
                else:
                    st.error("Visualization encountered an error")
                    st.code(result.stderr)

    elif tool == "Validate Formulas":
        st.subheader("✅ Validate Formulas")
        st.markdown("""
        Validates formula predictions against ALL experimental data.

        Computes errors by dimension and generates validation report.
        """)

        if st.button("✅ Run Validation", type="primary"):
            with st.spinner("Running validation..."):
                result = subprocess.run(
                    ["python3", "validate_split_formula.py"],
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✓ Validation complete!")
                    with st.expander("Show Output"):
                        st.text(result.stdout)
                else:
                    st.error("Validation encountered an error")
                    st.code(result.stderr)

    elif tool == "Layer Exclusion Analysis":
        st.subheader("🔬 Layer Exclusion Analysis")
        st.markdown("""
        Analyzes impact of excluding Layer 1 (immediate neighbors) from
        deviation analysis.

        **Key Finding**: Layer 1 should be INCLUDED
        - WITH Layer 1: R² = 0.9976 ✅
        - WITHOUT Layer 1: R² = 0.8798 ❌
        """)

        if st.button("🔬 Run Analysis", type="primary"):
            with st.spinner("Running layer exclusion analysis (may take 1-2 minutes)..."):
                result = subprocess.run(
                    ["python3", "compare_layer1_exclusion.py"],
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✓ Analysis complete!")
                    st.info("Results saved to: analysis/layer1_exclusion/")

                    with st.expander("Show Output"):
                        st.text(result.stdout)
                else:
                    st.error("Analysis encountered an error")
                    st.code(result.stderr)

    elif tool == "Deviation Analysis":
        st.subheader("📉 Deviation Analysis")
        st.markdown("""
        Fits a **cubic polynomial** to the per-layer deviation from (n-1)/2.

        **Formula**: `deviation(L) = a·L³ + b·L² + c·L + d`

        Reads `layer_statistics_bfs.csv` from an existing experiment directory.
        """)

        exp_dir = st.text_input("Experiment directory", value="data_output",
                                help="Path to a data_output/... subdirectory containing layer_statistics_bfs.csv")

        if st.button("📉 Run Deviation Analysis", type="primary"):
            try:
                from pathlib import Path as _Path
                matches = sorted(
                    [str(d) for d in _Path(exp_dir).rglob('layer_statistics_bfs.csv')
                     if d.parent != _Path(exp_dir)],
                    reverse=True
                )
                target = str(_Path(exp_dir)) if (_Path(exp_dir) / 'layer_statistics_bfs.csv').exists() else (matches[0].replace('/layer_statistics_bfs.csv', '') if matches else None)
                if target is None:
                    st.error("No layer_statistics_bfs.csv found. Run a Full Graph Analysis first.")
                else:
                    with st.spinner("Running deviation analysis..."):
                        from dfs_analyzer.core.deviation_analysis import run_deviation_analysis
                        fig, result, layers, means, n = run_deviation_analysis(
                            target, graph_name=_Path(target).name, output_dir=target)
                    r = result
                    st.success("✓ Deviation analysis complete!")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("a", f"{r['a']:.4e}")
                    col2.metric("b", f"{r['b']:.4e}")
                    col3.metric("c", f"{r['c']:.4e}")
                    col4.metric("R²", f"{r['r_squared']:.4f}")
                    img_path = _Path(target) / 'deviation_analysis.png'
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif tool == "Sigmoid Model Fitting":
        st.subheader("〽️ Sigmoid Model Fitting")
        st.markdown("""
        Fits **sigmoid** curves with three different transformations of the layer variable:

        | Model | f(L) |
        |-------|------|
        | Plain | L |
        | Square root | √L |
        | Logarithmic | log(L) |

        **Best model** (highest R²) is highlighted with ★.
        """)

        exp_dir = st.text_input("Experiment directory", value="data_output",
                                key="sig_dir",
                                help="Path containing layer_statistics_bfs.csv")

        if st.button("〽️ Fit Sigmoid Models", type="primary"):
            try:
                from pathlib import Path as _Path
                target = str(_Path(exp_dir)) if (_Path(exp_dir) / 'layer_statistics_bfs.csv').exists() else None
                if target is None:
                    matches = sorted([str(d.parent) for d in _Path(exp_dir).rglob('layer_statistics_bfs.csv')], reverse=True)
                    target  = matches[0] if matches else None
                if target is None:
                    st.error("No layer_statistics_bfs.csv found. Run a Full Graph Analysis first.")
                else:
                    with st.spinner("Fitting sigmoid models..."):
                        from dfs_analyzer.core.sigmoid_fitting import run_sigmoid_fitting
                        best, results, fig, layers, means, n = run_sigmoid_fitting(
                            target, graph_name=_Path(target).name, output_dir=target)
                    st.success(f"✓ Best model: **{best}**")
                    rows = []
                    for name, r in results.items():
                        if r['success']:
                            rows.append({"Transform": name, "A": f"{r['A']:.0f}",
                                         "k": f"{r['k']:.4f}", "R²": f"{r['r2']:.4f}",
                                         "Best": "★" if name == best else ""})
                    import pandas as pd
                    st.table(pd.DataFrame(rows))
                    img_path = _Path(target) / 'sigmoid_fitting.png'
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif tool == "Layer Variance Analysis":
        st.subheader("📊 Layer Variance Analysis")
        st.markdown("""
        Plots the **within-layer spread** of discovery numbers as a function of BFS layer.

        - **Left panel**: std dev of vertex means within each layer
        - **Right panel**: coefficient of variation CV = std/mean (normalised)

        Answers: does DFS become more or less uniform farther from the root?
        """)

        exp_dir = st.text_input("Experiment directory", value="data_output",
                                key="var_dir",
                                help="Path containing layer_statistics_bfs.csv")

        if st.button("📊 Run Layer Variance Analysis", type="primary"):
            try:
                from pathlib import Path as _Path
                target = str(_Path(exp_dir)) if (_Path(exp_dir) / 'layer_statistics_bfs.csv').exists() else None
                if target is None:
                    matches = sorted([str(d.parent) for d in _Path(exp_dir).rglob('layer_statistics_bfs.csv')], reverse=True)
                    target  = matches[0] if matches else None
                if target is None:
                    st.error("No layer_statistics_bfs.csv found. Run a Full Graph Analysis first.")
                else:
                    with st.spinner("Running layer variance analysis..."):
                        from dfs_analyzer.core.layer_variance import run_layer_variance_analysis
                        result = run_layer_variance_analysis(
                            target, graph_name=_Path(target).name, output_dir=target)
                    st.success("✓ Layer variance analysis complete!")
                    img_path = _Path(target) / 'layer_variance.png'
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif tool == "─── Post-Experiment Analyses ───":
        st.info("Select one of the post-experiment analyses from the dropdown above.")


def show_documentation_tab():
    """Documentation tab with complete information."""
    st.header("📚 Documentation")

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
    - Formula predictions available (3D-13D)

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

    ### Methods

    **RDFS (Randomized DFS)**: Runs multiple randomized DFS simulations
    - Empirical sampling approach
    - Tests the expected behavior through repeated trials
    - Provides mean, variance, and statistical validation

    ### Analysis Tools

    **Formula Predictions**: Instant predictions for hypercubes (3D-13D)
    - No experiments needed
    - 6.52% average error
    - 95% confidence: ±0.21%

    **BFS vs DFS**: Compare numbering schemes
    - Tests 7 graph types
    - Identifies branching effects
    - "Skip and catch up" pattern in hypercubes

    **Confidence Intervals**: Quantify prediction uncertainty
    - Error propagation from coefficients
    - 68%, 95%, 99% CI
    - Uncertainty grows with layer number

    ### HPC Usage

    ```bash
    # SSH with port forwarding
    ssh -L 8501:localhost:8501 user@hpc.edu

    # Load environment
    conda activate dfs-analyzer

    # Run GUI
    streamlit run run_gui.py

    # Open in browser: http://localhost:8501
    ```

    ### File Outputs

    Each experiment generates:
    - `summary.txt` - Human-readable summary
    - `data.csv` - Per-vertex statistics
    - `layer_statistics_bfs.csv` - Layer-by-layer stats
    - `visualization.png` - Bar chart
    - `histogram.png` - Distribution
    - `layer_analysis.png` - Distance analysis

    ### Documentation Files

    - `README.md` - Main documentation
    - `QUICKSTART.md` - Quick start guide
    - `ALL_ANALYSIS_FEATURES.md` - Complete analysis guide
    - `FORMULA_QUICK_REFERENCE.md` - Formula cheat sheet
    - `BFS_DFS_COMPARISON_RESULTS.md` - BFS vs DFS findings

    ### Citation

    If you use this tool in your research:
    ```
    Random DFS Graph Analyzer v0.6.0
    Venkat Mahesh Mandava, Ashoka University
    https://github.com/theking7415/capstone-project-dfs-analyzer
    ```
    """)


if __name__ == "__main__":
    main()
