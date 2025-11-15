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
        graph_type: Type of graph to analyze (currently only "hypercube").
        dimension: Dimension parameter for the graph (e.g., d for hypercube).
        num_samples: Number of RDFS runs to perform.
        rng_seed: Random number generator seed for reproducibility.
        output_dir: Directory to save results (default: "data_output").
        experiment_name: Optional custom name for the experiment.
        save_plots: Whether to generate and save visualizations.
        export_formats: List of export formats (csv, json, txt, pickle).

    Example:
        >>> config = ExperimentConfig(
        ...     graph_type="hypercube",
        ...     dimension=5,
        ...     num_samples=10000
        ... )
    """

    graph_type: str = "hypercube"
    dimension: int = 3
    num_samples: int = 1000
    rng_seed: int = 1832479182
    output_dir: str = "data_output"
    experiment_name: Optional[str] = None
    save_plots: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["csv", "txt"])

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.graph_type not in ["hypercube"]:
            raise ValueError(
                f"Unsupported graph type: {self.graph_type}. "
                f"Currently supported: hypercube"
            )

        if self.dimension < 1:
            raise ValueError("Dimension must be at least 1")

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
        return f"{self.graph_type} (dim={self.dimension})"

    def get_auto_experiment_name(self) -> str:
        """
        Generate an automatic experiment name based on configuration.

        Returns:
            Name like "hypercube-5d-10000-samples".
        """
        if self.graph_type == "hypercube":
            return f"hypercube-{self.dimension}d-{self.num_samples}-samples"
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
            "num_samples": self.num_samples,
            "rng_seed": self.rng_seed,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "save_plots": self.save_plots,
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
