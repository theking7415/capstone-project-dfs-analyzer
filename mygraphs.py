from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np

Vertex = TypeVar("Vertex")


class Graph(ABC, Generic[Vertex]):
    @abstractmethod
    def get_start_vertex(self) -> Vertex:
        pass

    @abstractmethod
    def get_adj_list(self, v) -> list[Vertex]:
        pass

    @abstractmethod
    def plot_means_vars(self, summary_stats: dict[Vertex, Any], *, fname=None):
        pass

    @abstractmethod
    def desc(self) -> str:
        pass

    @abstractmethod
    def number_vertices(self) -> int:
        pass

HypercubeVertexType = tuple[int, ...]

class Hypercube(Graph[HypercubeVertexType]):
    def __init__(self, d: int):
        """Creates a hypercube of dimension d.

        Vertices are represented by tuples of 0s and 1s of length d.
        For example, a 3D hypercube has vertices like (0,0,0), (0,1,0), etc.
        """
        if d < 1:
            raise ValueError("Dimension d must be a positive integer.")
        self.d = d

    def get_start_vertex(self) -> HypercubeVertexType:
        # The all-zeros vertex is a standard and symmetric starting point.
        return (0,) * self.d

    def get_adj_list(self, v: HypercubeVertexType) -> list[HypercubeVertexType]:
        """Neighbors are vertices that differ in exactly one position (Hamming distance 1)."""
        neighbors = []
        # We can find all neighbors by flipping each bit one at a time.
        for i in range(self.d):
            # Convert to list to modify the bit, then convert back to tuple
            neighbor_list = list(v)
            neighbor_list[i] = 1 - neighbor_list[i]  # Flips 0 to 1 and 1 to 0
            neighbors.append(tuple(neighbor_list))
        return neighbors

    def desc(self) -> str:
        return f"hypercube-{self.d}d"

    def number_vertices(self) -> int:
        # A d-dimensional hypercube has 2^d vertices.
        return 2**self.d

    def plot_means_vars(self, summary_stats: dict[HypercubeVertexType, Any], *, fname=None):
        """
        Visualizing data on a hypercube is non-trivial as it lacks a natural 2D layout.
        A bar chart is an effective way to display the statistics for each vertex.
        """
        # Sort vertices for a consistent plotting order.
        # Here we sort them as if they are binary numbers.
        sorted_vertices = sorted(summary_stats.keys())

        # Create string labels for the x-axis, e.g., '010'
        labels = ["".join(map(str, v)) for v in sorted_vertices]
        means = [summary_stats[v].mean for v in sorted_vertices]
        sds = [np.sqrt(summary_stats[v].variance) for v in sorted_vertices]

        fig, ax = plt.subplots(figsize=(max(10, self.number_vertices() * 0.5), 6))

        ax.bar(labels, means, yerr=sds, capsize=5, color='skyblue', ecolor='gray')

        ax.set_ylabel('Mean DFS Number')
        ax.set_xlabel('Vertex (Binary Representation)')
        ax.set_title(f'Mean DFS Number for a {self.d}-Dimensional Hypercube')
        plt.xticks(rotation=60, ha="right") # Rotate labels if they overlap

        if fname:
            fname_str = f": {fname}"
        else:
            fname_str = ""

        fig.suptitle(rf"{self.d}-D Hypercube ({2**self.d} vertices){fname_str}")
        fig.tight_layout()
        plt.show()
