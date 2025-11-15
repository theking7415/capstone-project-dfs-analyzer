"""
Graph abstractions for DFS analysis.

Provides an abstract Graph interface and concrete implementations
for various symmetric regular graphs used in validating the (n-1)/2 conjecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np

# Defines generic vertex type for type-safe graph implementations
Vertex = TypeVar("Vertex")


class Graph(ABC, Generic[Vertex]):
    """
    Defines abstract base class for graphs used in DFS analysis.

    Allows the RDFS algorithm to work with any graph type without modification.
    Concrete graph implementations must provide methods for traversal,
    vertex enumeration, and visualization.

    Type Parameters:
        Vertex: Specifies the type used to represent vertices in the graph.
    """

    @abstractmethod
    def get_start_vertex(self) -> Vertex:
        """Returns the starting vertex for DFS traversal."""
        pass

    @abstractmethod
    def get_adj_list(self, v: Vertex) -> list[Vertex]:
        """
        Returns the adjacency list for a given vertex.

        Args:
            v: Specifies the vertex whose neighbors to retrieve.

        Returns:
            List of vertices adjacent to v.
        """
        pass

    @abstractmethod
    def plot_means_vars(self, summary_stats: dict[Vertex, Any], *, fname=None):
        """
        Visualizes the mean discovery numbers and variances for vertices.

        Args:
            summary_stats: Maps vertices to their statistical summaries.
            fname: Specifies optional filename to save the plot.
        """
        pass

    @abstractmethod
    def desc(self) -> str:
        """Returns a short description of the graph for file naming."""
        pass

    @abstractmethod
    def number_vertices(self) -> int:
        """Returns the total number of vertices in the graph."""
        pass


# Defines vertex type as tuple of integers for hypercube graphs
HypercubeVertexType = tuple[int, ...]


class Hypercube(Graph[HypercubeVertexType]):
    """
    Represents a d-dimensional hypercube graph.

    Vertices are represented as tuples of 0s and 1s of length d.
    Two vertices are adjacent if they differ in exactly one coordinate.
    The graph is d-regular with 2^d total vertices.

    Attributes:
        d: Stores the dimension of the hypercube.
    """

    def __init__(self, d: int):
        """
        Creates a d-dimensional hypercube.

        Args:
            d: Specifies the dimension (must be >= 1).

        Raises:
            ValueError: Occurs when d < 1.
        """
        # Validates dimension is positive
        if d < 1:
            raise ValueError("Dimension d must be a positive integer.")
        self.d = d

    def get_start_vertex(self) -> HypercubeVertexType:
        """Returns the all-zeros vertex as symmetric starting point."""
        return (0,) * self.d

    def get_adj_list(self, v: HypercubeVertexType) -> list[HypercubeVertexType]:
        """
        Returns neighbors of vertex v in the hypercube.

        Neighbors differ in exactly one position (Hamming distance 1).

        Args:
            v: Represents a vertex as tuple of 0s and 1s.

        Returns:
            List of neighboring vertices.
        """
        neighbors = []
        # Iterates through each dimension to find neighbors
        for i in range(self.d):
            # Converts tuple to list for modification
            neighbor_list = list(v)
            # Flips bit at position i (0 becomes 1, 1 becomes 0)
            neighbor_list[i] = 1 - neighbor_list[i]
            # Converts back to tuple and adds to neighbors
            neighbors.append(tuple(neighbor_list))
        return neighbors

    def desc(self) -> str:
        """Returns description string for filename generation."""
        return f"hypercube-{self.d}d"

    def number_vertices(self) -> int:
        """Returns 2^d, the total number of vertices."""
        return 2**self.d

    def plot_means_vars(
        self, summary_stats: dict[HypercubeVertexType, Any], *, fname=None
    ):
        """
        Creates visualization of mean DFS numbers for hypercube vertices.

        Generates bar chart showing mean discovery number for each vertex
        with error bars representing standard deviation.

        Args:
            summary_stats: Maps vertices to scipy.stats.describe results.
            fname: Specifies optional filename to save plot.

        Returns:
            Matplotlib figure object containing the visualization.
        """
        # Sorts vertices by binary value for consistent ordering
        sorted_vertices = sorted(summary_stats.keys())

        # Converts vertex tuples to binary string labels
        labels = ["".join(map(str, v)) for v in sorted_vertices]
        # Extracts mean values for each vertex
        means = [summary_stats[v].mean for v in sorted_vertices]
        # Calculates standard deviations from variances
        sds = [np.sqrt(summary_stats[v].variance) for v in sorted_vertices]

        # Creates figure with width scaled by number of vertices
        fig, ax = plt.subplots(figsize=(max(10, self.number_vertices() * 0.5), 6))

        # Draws bars with error bars representing standard deviation
        ax.bar(labels, means, yerr=sds, capsize=5, color="skyblue", ecolor="gray")

        # Sets axis labels and title
        ax.set_ylabel("Mean DFS Number", fontsize=12)
        ax.set_xlabel("Vertex (Binary Representation)", fontsize=12)
        ax.set_title(
            f"Mean DFS Number for a {self.d}-Dimensional Hypercube", fontsize=14
        )

        # Calculates theoretical (n-1)/2 value
        theoretical_value = (self.number_vertices() - 1) / 2
        # Adds horizontal line showing theoretical prediction
        ax.axhline(
            y=theoretical_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"(n-1)/2 = {theoretical_value:.2f}",
        )
        ax.legend()

        # Rotates x-axis labels to prevent overlap
        plt.xticks(rotation=60, ha="right")

        # Adds filename to title if provided
        if fname:
            fname_str = f": {fname}"
        else:
            fname_str = ""

        # Sets figure title with dimension and vertex count
        fig.suptitle(
            rf"{self.d}-D Hypercube ({2**self.d} vertices){fname_str}", fontsize=16
        )
        fig.tight_layout()

        return fig

    def __repr__(self) -> str:
        """Returns string representation showing dimension and vertex count."""
        return f"Hypercube(d={self.d}, vertices={self.number_vertices()})"
