"""
Experiment configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for a DFS analysis experiment.

    Attributes:
        graph_type: Type of graph to analyze (hypercube, petersen, triangular, torus, hexagonal, complete, ndgrid, gnp).
        dimension: Dimension parameter for the graph (e.g., d for hypercube, n for gnp).
        num_samples: Number of RDFS runs to perform.
        rng_seed: Random number generator seed for reproducibility.
        output_dir: Directory to save results (default: "data_output").
        experiment_name: Optional custom name for the experiment.
        save_plots: Whether to generate visualization.png (default: False).
        save_detailed_stats: Whether to save detailed_stats.txt (default: False).
        save_csv: Whether to save data.csv (default: False).
        export_formats: Additional export formats (json, pickle) (default: ['pickle']).
        petersen_k: Skip parameter k for Petersen graphs.
        lattice_rows: Number of rows for triangular lattice, torus grid, or hexagonal lattice.
        lattice_cols: Number of columns for triangular lattice, torus grid, or hexagonal lattice.
        grid_size: Number of points per dimension for ndgrid (n-dimensional torus grid).
        gnp_p: Edge probability p for G(n,p) random graphs (0 < p < 1).

    Note:
        summary.txt is always generated regardless of settings.

    Example:
        >>> config = ExperimentConfig(
        ...     graph_type="hypercube",
        ...     dimension=5,
        ...     num_samples=10000,
        ...     save_csv=True
        ... )
    """

    graph_type: str = "hypercube"
    dimension: int = 3
    petersen_k: Optional[int] = None  # Only used for Petersen graphs
    lattice_rows: Optional[int] = None  # Only used for triangular lattice
    lattice_cols: Optional[int] = None  # Only used for triangular lattice
    grid_size: Optional[int] = None  # Only used for ndgrid (points per dimension)
    gnp_p: Optional[float] = None  # Only used for G(n,p) random graphs (edge probability)
    num_samples: int = 1000
    rng_seed: int = 1832479182
    output_dir: str = "data_output"
    experiment_name: Optional[str] = None
    save_plots: bool = False  # Changed default to False to save storage
    save_detailed_stats: bool = False  # New option for detailed statistics
    save_csv: bool = False  # New option for CSV export
    export_formats: list[str] = field(default_factory=lambda: ['pickle'])  # Pickle enabled by default for plot regeneration

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.graph_type not in ["hypercube", "petersen", "triangular", "torus", "hexagonal", "complete", "ndgrid", "gnp"]:
            raise ValueError(
                f"Unsupported graph type: {self.graph_type}. "
                f"Currently supported: hypercube, petersen, triangular, torus, hexagonal, complete, ndgrid, gnp"
            )

        if self.graph_type == "hypercube":
            if self.dimension < 1:
                raise ValueError("Dimension must be at least 1")
        elif self.graph_type == "petersen":
            if self.dimension < 3:
                raise ValueError("n must be at least 3 for Petersen graphs")
            if self.petersen_k is None:
                raise ValueError("petersen_k must be specified for Petersen graphs")
            if self.petersen_k < 1 or self.petersen_k >= self.dimension:
                raise ValueError(f"k must be in range [1, n-1] where n={self.dimension}")
        elif self.graph_type in ["triangular", "torus", "hexagonal"]:
            if self.lattice_rows is None or self.lattice_cols is None:
                raise ValueError(f"lattice_rows and lattice_cols must be specified for {self.graph_type}")
            if self.lattice_rows < 3 or self.lattice_cols < 3:
                raise ValueError(f"{self.graph_type.capitalize()} lattice requires at least 3 rows and 3 columns")
        elif self.graph_type == "complete":
            if self.dimension < 2:
                raise ValueError("Complete graph requires at least 2 vertices (dimension >= 2)")
        elif self.graph_type == "ndgrid":
            if self.dimension < 2:
                raise ValueError("NDGrid requires at least 2 dimensions")
            if self.grid_size is None:
                raise ValueError("grid_size must be specified for ndgrid")
            if self.grid_size < 2:
                raise ValueError("NDGrid requires at least 2 points per dimension")
        elif self.graph_type == "gnp":
            if self.dimension < 2:
                raise ValueError("G(n,p) requires at least 2 vertices (n >= 2)")
            if self.gnp_p is None:
                raise ValueError("gnp_p must be specified for G(n,p) random graphs")
            if not (0 < self.gnp_p < 1):
                raise ValueError(f"Edge probability p must be in (0, 1), got {self.gnp_p}")

        if self.num_samples < 1:
            raise ValueError("Number of samples must be at least 1")

        valid_formats = ["csv", "json", "txt", "pickle"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Unsupported export format: {fmt}. "
                    f"Valid formats: {valid_formats}"
                )

    def get_graph_description(self) -> str:
        """
        Get a human-readable description of the graph.

        Returns:
            Description string like "Hypercube 5D (32 vertices)".
        """
        if self.graph_type == "hypercube":
            num_vertices = 2**self.dimension
            return f"Hypercube {self.dimension}D ({num_vertices} vertices)"
        elif self.graph_type == "petersen":
            num_vertices = 2 * self.dimension
            return f"Petersen GP({self.dimension}, {self.petersen_k}) ({num_vertices} vertices)"
        elif self.graph_type == "triangular":
            num_vertices = self.lattice_rows * self.lattice_cols
            return f"Triangular Lattice {self.lattice_rows}×{self.lattice_cols} ({num_vertices} vertices)"
        elif self.graph_type == "torus":
            num_vertices = self.lattice_rows * self.lattice_cols
            return f"Torus Grid {self.lattice_rows}×{self.lattice_cols} ({num_vertices} vertices)"
        elif self.graph_type == "hexagonal":
            num_vertices = self.lattice_rows * self.lattice_cols
            return f"Hexagonal Lattice {self.lattice_rows}×{self.lattice_cols} ({num_vertices} vertices, graphene)"
        elif self.graph_type == "complete":
            num_vertices = self.dimension
            num_edges = num_vertices * (num_vertices - 1) // 2
            return f"Complete Graph K_{num_vertices} ({num_vertices} vertices, {num_edges} edges)"
        elif self.graph_type == "ndgrid":
            num_vertices = self.grid_size ** self.dimension
            degree = 2 * self.dimension
            return f"{self.dimension}D Torus Grid {self.grid_size}^{self.dimension} ({num_vertices} vertices, degree {degree})"
        elif self.graph_type == "gnp":
            num_vertices = self.dimension
            expected_degree = (num_vertices - 1) * self.gnp_p
            return f"G(n,p) Random Graph: n={num_vertices}, p={self.gnp_p:.3f} ({num_vertices} vertices, expected degree {expected_degree:.1f})"
        return f"{self.graph_type} (dim={self.dimension})"

    def get_auto_experiment_name(self) -> str:
        """
        Generate an automatic experiment name based on configuration.

        Returns:
            Name like "hypercube-5d-10000-samples" or "petersen-5-2-10000-samples".
        """
        if self.graph_type == "hypercube":
            return f"hypercube-{self.dimension}d-{self.num_samples}-samples"
        elif self.graph_type == "petersen":
            return f"petersen-{self.dimension}-{self.petersen_k}-{self.num_samples}-samples"
        elif self.graph_type == "triangular":
            return f"triangular-{self.lattice_rows}x{self.lattice_cols}-{self.num_samples}-samples"
        elif self.graph_type == "torus":
            return f"torus-{self.lattice_rows}x{self.lattice_cols}-{self.num_samples}-samples"
        elif self.graph_type == "hexagonal":
            return f"hexagonal-{self.lattice_rows}x{self.lattice_cols}-{self.num_samples}-samples"
        elif self.graph_type == "complete":
            return f"complete-k{self.dimension}-{self.num_samples}-samples"
        elif self.graph_type == "ndgrid":
            return f"ndgrid-{self.dimension}d-{self.grid_size}^{self.dimension}-{self.num_samples}-samples"
        elif self.graph_type == "gnp":
            return f"gnp-n{self.dimension}-p{self.gnp_p:.3f}-{self.num_samples}-samples"
        return f"{self.graph_type}-{self.dimension}-{self.num_samples}-samples"

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "graph_type": self.graph_type,
            "dimension": self.dimension,
            "petersen_k": self.petersen_k,
            "lattice_rows": self.lattice_rows,
            "lattice_cols": self.lattice_cols,
            "grid_size": self.grid_size,
            "gnp_p": self.gnp_p,
            "num_samples": self.num_samples,
            "rng_seed": self.rng_seed,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "save_plots": self.save_plots,
            "save_detailed_stats": self.save_detailed_stats,
            "save_csv": self.save_csv,
            "export_formats": self.export_formats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            ExperimentConfig instance.
        """
        return cls(**data)
