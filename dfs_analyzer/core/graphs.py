"""
Graph abstractions for DFS analysis.

Provides an abstract Graph interface and concrete implementations
for various symmetric regular graphs used in analyzing DFS behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Tuple

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
            f"Mean DFS Number for a {self.d}-Dimensional Hypercube",
            fontsize=14
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


# Defines vertex type as tuple of (ring, index) for Petersen graphs
PetersenVertexType = tuple[str, int]


class GeneralizedPetersen(Graph[PetersenVertexType]):
    """
    Represents a Generalized Petersen Graph GP(n, k).

    Has 2n vertices total with two rings of n vertices each.
    Outer ring vertices connect sequentially, inner ring vertices
    connect with skip parameter k, and spokes connect corresponding vertices.

    Attributes:
        n: Stores number of vertices in each ring.
        k: Stores skip parameter for inner ring connections.
    """

    def __init__(self, n: int, k: int):
        """
        Creates a Generalized Petersen Graph GP(n, k).

        Args:
            n: Specifies number of vertices in each ring (total = 2n).
            k: Specifies skip parameter for inner ring.

        Raises:
            ValueError: Occurs when n < 3 or k not in range [1, n-1].
        """
        # Validates n is at least 3
        if n < 3:
            raise ValueError("n must be at least 3")
        # Validates k is in valid range
        if k < 1 or k >= n:
            raise ValueError("k must be in range [1, n-1]")

        self.n = n
        self.k = k

    def get_start_vertex(self) -> PetersenVertexType:
        """Returns first outer vertex as symmetric starting point."""
        return ('outer', 0)

    def get_adj_list(self, v: PetersenVertexType) -> list[PetersenVertexType]:
        """
        Returns neighbors of vertex v in the Petersen graph.

        Outer ring vertices connect to adjacent outer vertices and inner spoke.
        Inner ring vertices connect to outer spoke and adjacent inner vertices.

        Args:
            v: Represents vertex as ('outer'|'inner', index).

        Returns:
            List of neighboring vertices.
        """
        ring, i = v
        neighbors = []

        if ring == 'outer':
            # Adds previous vertex in outer ring
            neighbors.append(('outer', (i - 1) % self.n))
            # Adds next vertex in outer ring
            neighbors.append(('outer', (i + 1) % self.n))
            # Adds spoke connection to inner ring
            neighbors.append(('inner', i))
        else:  # ring == 'inner'
            # Adds spoke connection to outer ring
            neighbors.append(('outer', i))
            # Adds previous vertex in inner ring (with skip k)
            neighbors.append(('inner', (i - self.k) % self.n))
            # Adds next vertex in inner ring (with skip k)
            neighbors.append(('inner', (i + self.k) % self.n))

        return neighbors

    def desc(self) -> str:
        """Returns description string for filename generation."""
        return f"petersen-{self.n}-{self.k}"

    def number_vertices(self) -> int:
        """Returns 2n, the total number of vertices."""
        return 2 * self.n

    def get_coordinates(self, v: PetersenVertexType) -> tuple[float, float]:
        """
        Returns 2D coordinates for a Petersen graph vertex.

        Places outer ring on a circle of radius 2, inner ring on radius 1.
        Uses standard polar coordinates converted to Cartesian.

        Args:
            v: Vertex as ('outer'|'inner', index)

        Returns:
            (x, y) coordinates as floats
        """
        import math
        ring, i = v

        # Angle for vertex i (evenly spaced around circle)
        angle = 2 * math.pi * i / self.n

        if ring == 'outer':
            # Outer ring at radius 2
            radius = 2.0
        else:  # inner
            # Inner ring at radius 1
            radius = 1.0

        # Converts polar to Cartesian
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        return (x, y)

    def plot_means_vars(
        self, summary_stats: dict[PetersenVertexType, Any], *, fname=None
    ):
        """
        Creates visualization of mean DFS numbers for Petersen graph vertices.

        Generates dual bar charts showing outer and inner ring statistics separately.

        Args:
            summary_stats: Maps vertices to scipy.stats.describe results.
            fname: Specifies optional filename to save plot.

        Returns:
            Matplotlib figure object containing the visualization.
        """
        # Separates outer and inner ring vertices
        outer_vertices = sorted(
            [(ring, i) for ring, i in summary_stats.keys() if ring == 'outer'],
            key=lambda x: x[1]
        )
        inner_vertices = sorted(
            [(ring, i) for ring, i in summary_stats.keys() if ring == 'inner'],
            key=lambda x: x[1]
        )

        # Extracts statistics for outer ring
        outer_means = [summary_stats[v].mean for v in outer_vertices]
        outer_sds = [np.sqrt(summary_stats[v].variance) for v in outer_vertices]
        # Extracts statistics for inner ring
        inner_means = [summary_stats[v].mean for v in inner_vertices]
        inner_sds = [np.sqrt(summary_stats[v].variance) for v in inner_vertices]

        # Creates side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plots outer ring with error bars
        x_outer = list(range(self.n))
        ax1.bar(x_outer, outer_means, yerr=outer_sds, capsize=5,
                color='skyblue', ecolor='gray')
        ax1.set_ylabel('Mean DFS Number', fontsize=12)
        ax1.set_xlabel('Vertex Index (Outer Ring)', fontsize=12)
        ax1.set_title(f'Outer Ring - GP({self.n}, {self.k})', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plots inner ring with error bars
        x_inner = list(range(self.n))
        ax2.bar(x_inner, inner_means, yerr=inner_sds, capsize=5,
                color='lightcoral', ecolor='gray')
        ax2.set_ylabel('Mean DFS Number', fontsize=12)
        ax2.set_xlabel('Vertex Index (Inner Ring)', fontsize=12)
        ax2.set_title(f'Inner Ring - GP({self.n}, {self.k})', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Adds theoretical (n-1)/2 line to both plots
        theoretical_value = (self.number_vertices() - 1) / 2
        ax1.axhline(y=theoretical_value, color='red', linestyle='--',
                   linewidth=2, label=f'(n-1)/2 = {theoretical_value:.2f}')
        ax2.axhline(y=theoretical_value, color='red', linestyle='--',
                   linewidth=2, label=f'(n-1)/2 = {theoretical_value:.2f}')
        ax1.legend()
        ax2.legend()

        # Sets figure title
        fig.suptitle(
            f'Mean DFS Numbers - Generalized Petersen Graph GP({self.n}, {self.k})',
            fontsize=16
        )
        fig.tight_layout()

        return fig

    def __repr__(self) -> str:
        """Returns string representation showing parameters and vertex count."""
        return f"GeneralizedPetersen(n={self.n}, k={self.k}, vertices={self.number_vertices()})"


# Defines vertex type as tuple of two integers for triangular lattice
TriangularLatticeVertexType = tuple[int, int]


class TriangularLattice(Graph[TriangularLatticeVertexType]):
    """
    Represents a triangular lattice graph with periodic boundary conditions.

    A triangular lattice is a 2D tiling where each vertex has 6 neighbors,
    forming a regular pattern of equilateral triangles. This implementation
    uses axial coordinates (q, r) where vertices are positioned in a
    rhombus-shaped grid.

    Properties:
    - Each vertex has degree 6 (hexagonal coordination)
    - Regular and symmetric structure
    - Periodic boundary conditions (torus topology)
    - Total vertices: rows × cols

    Attributes:
        rows: Number of rows in the lattice.
        cols: Number of columns in the lattice.

    Example:
        >>> lattice = TriangularLattice(5, 5)  # 5x5 triangular lattice
        >>> lattice.number_vertices()
        25
        >>> len(lattice.get_adj_list((0, 0)))
        6
    """

    def __init__(self, rows: int, cols: int):
        """
        Creates a triangular lattice with periodic boundary conditions.

        Args:
            rows: Number of rows (must be >= 3 for valid lattice).
            cols: Number of columns (must be >= 3 for valid lattice).

        Raises:
            ValueError: If rows or cols < 3.
        """
        # Validates dimensions
        if rows < 3 or cols < 3:
            raise ValueError("Triangular lattice requires at least 3 rows and 3 columns.")

        self.rows = rows
        self.cols = cols

    def get_start_vertex(self) -> TriangularLatticeVertexType:
        """Returns (0, 0) as the starting vertex."""
        return (0, 0)

    def get_adj_list(self, v: TriangularLatticeVertexType) -> list[TriangularLatticeVertexType]:
        """
        Returns the 6 neighbors of vertex v in triangular lattice.

        In axial coordinates, the 6 neighbors are at offsets:
        - (±1, 0): horizontal neighbors
        - (0, ±1): diagonal neighbors (one direction)
        - (±1, ∓1): diagonal neighbors (other direction)

        Uses periodic boundary conditions (wraps around at edges).

        Args:
            v: Vertex as (q, r) coordinate tuple.

        Returns:
            List of 6 neighboring vertices.
        """
        q, r = v

        # Defines the 6 neighbor directions in axial coordinates
        directions = [
            (1, 0),   # right
            (-1, 0),  # left
            (0, 1),   # down-right
            (0, -1),  # up-left
            (1, -1),  # up-right
            (-1, 1),  # down-left
        ]

        neighbors = []
        for dq, dr in directions:
            # Computes new coordinates with periodic boundary conditions
            new_q = (q + dq) % self.cols
            new_r = (r + dr) % self.rows
            neighbors.append((new_q, new_r))

        return neighbors

    def get_coordinates(self, v: TriangularLatticeVertexType) -> tuple[float, float]:
        """
        Returns 2D Cartesian coordinates for visualization and distance computation.

        Converts axial coordinates (q, r) to Cartesian (x, y) using standard
        triangular lattice geometry with unit spacing.

        Args:
            v: Vertex as (q, r) coordinate tuple.

        Returns:
            Tuple of (x, y) Cartesian coordinates.
        """
        q, r = v
        # Standard triangular lattice coordinate conversion
        x = q + 0.5 * r
        y = r * np.sqrt(3) / 2
        return (x, y)

    def l1_distance(self, v1: TriangularLatticeVertexType, v2: TriangularLatticeVertexType) -> int:
        """
        Computes L1 distance on triangular lattice with torus wrapping.

        Uses axial coordinate system (q, r). On a torus, the distance in each
        dimension is the minimum of direct and wrapped distances.

        Args:
            v1: First vertex as (q, r) tuple.
            v2: Second vertex as (q, r) tuple.

        Returns:
            L1 distance in axial coordinates accounting for torus topology.

        Example:
            >>> lattice = TriangularLattice(10, 10)
            >>> lattice.l1_distance((0, 0), (1, 1))
            2
            >>> lattice.l1_distance((0, 0), (9, 9))  # Wraps around
            2
        """
        q1, r1 = v1
        q2, r2 = v2

        # Computes minimum distance in q dimension (accounting for wrap)
        q_dist = abs(q1 - q2)
        q_dist = min(q_dist, self.cols - q_dist)

        # Computes minimum distance in r dimension (accounting for wrap)
        r_dist = abs(r1 - r2)
        r_dist = min(r_dist, self.rows - r_dist)

        return q_dist + r_dist

    def l2_distance(self, v1: TriangularLatticeVertexType, v2: TriangularLatticeVertexType) -> float:
        """
        Computes L2 (Euclidean) distance on triangular lattice with torus wrapping.

        Converts to Cartesian coordinates and computes Euclidean distance,
        accounting for torus wrapping in both dimensions.

        Args:
            v1: First vertex as (q, r) tuple.
            v2: Second vertex as (q, r) tuple.

        Returns:
            L2 distance in Cartesian coordinates accounting for torus topology.

        Example:
            >>> lattice = TriangularLattice(10, 10)
            >>> lattice.l2_distance((0, 0), (1, 0))
            1.0
        """
        q1, r1 = v1
        q2, r2 = v2

        # Computes minimum distance in q dimension (accounting for wrap)
        q_dist = abs(q1 - q2)
        q_dist = min(q_dist, self.cols - q_dist)
        if abs(q1 - q2) > self.cols - abs(q1 - q2):
            q_dist = -q_dist  # Preserves sign for direction

        # Computes minimum distance in r dimension (accounting for wrap)
        r_dist = abs(r1 - r2)
        r_dist = min(r_dist, self.rows - r_dist)
        if abs(r1 - r2) > self.rows - abs(r1 - r2):
            r_dist = -r_dist  # Preserves sign for direction

        # Converts axial distance to Cartesian
        x_dist = q_dist + 0.5 * r_dist
        y_dist = r_dist * np.sqrt(3) / 2

        # Returns Euclidean distance
        return float(np.sqrt(x_dist**2 + y_dist**2))

    def desc(self) -> str:
        """Returns description string for filename generation."""
        return f"triangular-{self.rows}x{self.cols}"

    def number_vertices(self) -> int:
        """Returns total number of vertices (rows × cols)."""
        return self.rows * self.cols

    def plot_means_vars(
        self, summary_stats: dict[TriangularLatticeVertexType, Any], *, fname=None
    ):
        """
        Creates heatmap visualization of mean DFS numbers for triangular lattice.

        Displays the lattice as a 2D grid where color represents mean discovery number.
        Shows the spatial distribution of DFS traversal patterns.

        Args:
            summary_stats: Maps vertices to scipy.stats.describe results.
            fname: Optional filename to save plot.

        Returns:
            Matplotlib figure object containing the visualization.
        """
        # Creates 2D array for heatmap
        data = np.zeros((self.rows, self.cols))

        # Fills array with mean values
        for (q, r), stats in summary_stats.items():
            data[r, q] = stats.mean

        # Creates figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Creates heatmap
        im = ax.imshow(data, cmap='viridis', aspect='equal', origin='lower')

        # Adds colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean DFS Number', fontsize=12)

        # Sets labels and title
        ax.set_xlabel('Column (q)', fontsize=12)
        ax.set_ylabel('Row (r)', fontsize=12)
        ax.set_title(
            f'Mean DFS Numbers - Triangular Lattice {self.rows}×{self.cols}\n'
            f'({self.number_vertices()} vertices, degree 6)',
            fontsize=14
        )

        # Adds theoretical (n-1)/2 value as text
        theoretical_value = (self.number_vertices() - 1) / 2
        ax.text(
            0.02, 0.98,
            f'(n-1)/2 = {theoretical_value:.2f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        fig.tight_layout()

        return fig

    def __repr__(self) -> str:
        """Returns string representation showing dimensions and vertex count."""
        return f"TriangularLattice(rows={self.rows}, cols={self.cols}, vertices={self.number_vertices()})"


# Defines vertex type as tuple of two integers for torus grid
TorusGridVertexType = tuple[int, int]


class TorusGrid(Graph[TorusGridVertexType]):
    """
    Represents a 2D grid graph with periodic boundary conditions (torus topology).

    A torus grid is a regular 2D lattice where each vertex has 4 neighbors
    (up, down, left, right), with edges wrapping around at boundaries to
    form a torus topology.

    Properties:
    - Each vertex has degree 4 (4-regular graph)
    - Regular and symmetric structure
    - Periodic boundary conditions (torus topology)
    - Total vertices: rows × cols

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.

    Example:
        >>> grid = TorusGrid(5, 5)  # 5x5 torus grid
        >>> grid.number_vertices()
        25
        >>> len(grid.get_adj_list((0, 0)))
        4
    """

    def __init__(self, rows: int, cols: int):
        """
        Creates a torus grid with periodic boundary conditions.

        Args:
            rows: Number of rows (must be >= 3 for valid torus).
            cols: Number of columns (must be >= 3 for valid torus).

        Raises:
            ValueError: If rows or cols < 3.
        """
        # Validates dimensions
        if rows < 3 or cols < 3:
            raise ValueError("Torus grid requires at least 3 rows and 3 columns.")

        self.rows = rows
        self.cols = cols

    def get_start_vertex(self) -> TorusGridVertexType:
        """Returns (0, 0) as the starting vertex."""
        return (0, 0)

    def get_adj_list(self, v: TorusGridVertexType) -> list[TorusGridVertexType]:
        """
        Returns the 4 neighbors of vertex v in torus grid.

        Neighbors are in cardinal directions (up, down, left, right).
        Uses periodic boundary conditions (wraps around at edges).

        Args:
            v: Vertex as (row, col) coordinate tuple.

        Returns:
            List of 4 neighboring vertices.
        """
        row, col = v

        # Defines the 4 neighbor directions
        neighbors = [
            ((row - 1) % self.rows, col),  # up
            ((row + 1) % self.rows, col),  # down
            (row, (col - 1) % self.cols),  # left
            (row, (col + 1) % self.cols),  # right
        ]

        return neighbors

    def get_coordinates(self, v: TorusGridVertexType) -> tuple[float, float]:
        """
        Returns 2D Cartesian coordinates for visualization and distance computation.

        Args:
            v: Vertex as (row, col) coordinate tuple.

        Returns:
            Tuple of (x, y) Cartesian coordinates.
        """
        row, col = v
        return (float(col), float(row))

    def l1_distance(self, v1: TorusGridVertexType, v2: TorusGridVertexType) -> int:
        """
        Computes L1 (Manhattan) distance on torus with wrapping.

        On a torus, the distance in each dimension is the minimum of:
        - Direct distance: |a - b|
        - Wrapped distance: dimension_size - |a - b|

        Args:
            v1: First vertex as (row, col) tuple.
            v2: Second vertex as (row, col) tuple.

        Returns:
            L1 distance accounting for torus topology.

        Example:
            >>> grid = TorusGrid(10, 10)
            >>> grid.l1_distance((0, 0), (1, 1))
            2
            >>> grid.l1_distance((0, 0), (9, 9))  # Wraps around
            2
        """
        row1, col1 = v1
        row2, col2 = v2

        # Computes minimum distance in row dimension (accounting for wrap)
        row_dist = abs(row1 - row2)
        row_dist = min(row_dist, self.rows - row_dist)

        # Computes minimum distance in col dimension (accounting for wrap)
        col_dist = abs(col1 - col2)
        col_dist = min(col_dist, self.cols - col_dist)

        return row_dist + col_dist

    def l2_distance(self, v1: TorusGridVertexType, v2: TorusGridVertexType) -> float:
        """
        Computes L2 (Euclidean) distance on torus with wrapping.

        On a torus, the distance in each dimension is the minimum of:
        - Direct distance: |a - b|
        - Wrapped distance: dimension_size - |a - b|

        Args:
            v1: First vertex as (row, col) tuple.
            v2: Second vertex as (row, col) tuple.

        Returns:
            L2 distance accounting for torus topology.

        Example:
            >>> grid = TorusGrid(10, 10)
            >>> grid.l2_distance((0, 0), (3, 4))
            5.0
            >>> grid.l2_distance((0, 0), (9, 9))  # Wraps around
            1.4142135623730951
        """
        row1, col1 = v1
        row2, col2 = v2

        # Computes minimum distance in row dimension (accounting for wrap)
        row_dist = abs(row1 - row2)
        row_dist = min(row_dist, self.rows - row_dist)

        # Computes minimum distance in col dimension (accounting for wrap)
        col_dist = abs(col1 - col2)
        col_dist = min(col_dist, self.cols - col_dist)

        # Returns Euclidean distance
        return (row_dist**2 + col_dist**2) ** 0.5

    def desc(self) -> str:
        """Returns description string for filename generation."""
        return f"torus-{self.rows}x{self.cols}"

    def number_vertices(self) -> int:
        """Returns total number of vertices (rows × cols)."""
        return self.rows * self.cols

    def plot_means_vars(
        self, summary_stats: dict[TorusGridVertexType, Any], *, fname=None
    ):
        """
        Creates heatmap visualization of mean DFS numbers for torus grid.

        Displays the grid as a 2D heatmap where color represents mean discovery number.
        Shows the spatial distribution of DFS traversal patterns.

        Args:
            summary_stats: Maps vertices to scipy.stats.describe results.
            fname: Optional filename to save plot.

        Returns:
            Matplotlib figure object containing the visualization.
        """
        # Creates 2D array for heatmap
        data = np.zeros((self.rows, self.cols))

        # Fills array with mean values
        for (row, col), stats in summary_stats.items():
            data[row, col] = stats.mean

        # Creates figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Creates heatmap
        im = ax.imshow(data, cmap='viridis', aspect='equal', origin='lower')

        # Adds colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean DFS Number', fontsize=12)

        # Sets labels and title
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(
            f'Mean DFS Numbers - Torus Grid {self.rows}×{self.cols}\n'
            f'({self.number_vertices()} vertices, degree 4)',
            fontsize=14
        )

        # Adds theoretical (n-1)/2 value as text
        theoretical_value = (self.number_vertices() - 1) / 2
        ax.text(
            0.02, 0.98,
            f'(n-1)/2 = {theoretical_value:.2f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        fig.tight_layout()

        return fig

    def __repr__(self) -> str:
        """Returns string representation showing dimensions and vertex count."""
        return f"TorusGrid(rows={self.rows}, cols={self.cols}, vertices={self.number_vertices()})"


# Defines vertex type as tuple of two integers for hexagonal lattice
HexagonalLatticeVertexType = Tuple[int, int]


class HexagonalLattice(Graph[HexagonalLatticeVertexType]):
    """
    Represents a hexagonal (honeycomb) lattice graph with periodic boundary conditions.

    A hexagonal lattice is the graph structure of graphene, where each vertex has
    exactly 3 neighbors, forming a pattern of regular hexagons. The lattice consists
    of two interpenetrating triangular sublattices (A and B).

    Properties:
    - Each vertex has degree 3 (unlike triangular lattice which has degree 6)
    - Bipartite graph structure (A and B sublattices)
    - Forms regular hexagons (honeycomb pattern)
    - Periodic boundary conditions (torus topology)
    - Total vertices: 2 × rows × cols

    Attributes:
        rows: Number of rows in the base lattice.
        cols: Number of columns in the base lattice.

    Example:
        >>> lattice = HexagonalLattice(5, 5)  # 5x5 honeycomb lattice
        >>> lattice.number_vertices()
        50  # 2 atoms per unit cell
        >>> len(lattice.get_adj_list((0, 0, 'A')))
        3
    """

    def __init__(self, rows: int, cols: int):
        """
        Creates a hexagonal lattice with periodic boundary conditions.

        Args:
            rows: Number of rows (must be >= 3 for valid lattice).
            cols: Number of columns (must be >= 3 for valid lattice).

        Raises:
            ValueError: If rows or cols < 3.
        """
        # Validates dimensions
        if rows < 3 or cols < 3:
            raise ValueError("Hexagonal lattice requires at least 3 rows and 3 columns.")

        self.rows = rows
        self.cols = cols

    def _get_sublattice(self, row: int, col: int) -> str:
        """
        Determines which sublattice (A or B) a vertex belongs to.

        Args:
            row: Row coordinate.
            col: Column coordinate.

        Returns:
            'A' or 'B' indicating the sublattice.
        """
        # Alternates sublattices in checkerboard pattern
        return 'A' if (row + col) % 2 == 0 else 'B'

    def get_start_vertex(self) -> HexagonalLatticeVertexType:
        """Returns (0, 0) as the starting vertex."""
        return (0, 0)

    def get_adj_list(self, v: HexagonalLatticeVertexType) -> list[HexagonalLatticeVertexType]:
        """
        Returns the 3 neighbors of vertex v in hexagonal lattice.

        In a honeycomb lattice, each vertex connects to exactly 3 neighbors
        on the opposite sublattice. The connection pattern depends on which
        sublattice the vertex belongs to.

        Uses periodic boundary conditions (wraps around at edges).

        Args:
            v: Vertex as (row, col) coordinate tuple.

        Returns:
            List of 3 neighboring vertices.
        """
        row, col = v
        sublattice = self._get_sublattice(row, col)

        neighbors = []

        if sublattice == 'A':
            # A sublattice connects to 3 B neighbors
            # Right, down-left, up-left
            offsets = [
                (0, 1),   # right
                (1, 0),   # down
                (-1, 0),  # up
            ]
        else:  # sublattice == 'B'
            # B sublattice connects to 3 A neighbors
            # Left, down-right, up-right
            offsets = [
                (0, -1),  # left
                (1, 0),   # down
                (-1, 0),  # up
            ]

        for dr, dc in offsets:
            # Computes new coordinates with periodic boundary conditions
            new_row = (row + dr) % self.rows
            new_col = (col + dc) % self.cols
            neighbors.append((new_row, new_col))

        return neighbors

    def get_coordinates(self, v: HexagonalLatticeVertexType) -> tuple[float, float]:
        """
        Returns 2D Cartesian coordinates for visualization.

        Converts lattice coordinates to physical positions using hexagonal geometry.

        Args:
            v: Vertex as (row, col) tuple.

        Returns:
            (x, y) coordinates for plotting.
        """
        row, col = v
        sublattice = self._get_sublattice(row, col)

        # Hexagonal lattice spacing
        # Horizontal spacing between columns
        dx = 1.5
        # Vertical spacing between rows
        dy = np.sqrt(3) / 2

        x = col * dx
        y = row * dy

        # Offset B sublattice by 0.5 in x direction
        if sublattice == 'B':
            x += 0.5

        return (x, y)

    def number_vertices(self) -> int:
        """Returns total number of vertices (rows × cols)."""
        return self.rows * self.cols

    def desc(self) -> str:
        """Returns description string for file naming."""
        return f"hexagonal-{self.rows}x{self.cols}"

    def plot_means_vars(self, summary_stats: dict[HexagonalLatticeVertexType, Any], *, fname=None):
        """
        Visualizes mean DFS numbers as a 2D hexagonal lattice heatmap.

        Creates a scatter plot where each hexagon represents a vertex,
        colored by its mean discovery number. A and B sublattices are
        shown in their proper geometric arrangement.

        Args:
            summary_stats: Maps vertices to their statistical summaries.
            fname: Optional filename to save the plot.

        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Extracts vertex positions and mean values
        x_coords = []
        y_coords = []
        mean_values = []

        for v in summary_stats:
            x, y = self.get_coordinates(v)
            x_coords.append(x)
            y_coords.append(y)
            mean_values.append(summary_stats[v].mean)

        # Creates scatter plot with hexagonal markers
        scatter = ax.scatter(
            x_coords, y_coords, c=mean_values,
            cmap='viridis', s=200, marker='h',
            edgecolors='black', linewidth=0.5
        )

        # Adds colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mean DFS Number', fontsize=12)

        # Sets labels and title
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(
            f'Mean DFS Numbers - Hexagonal Lattice {self.rows}×{self.cols}\n'
            f'({self.number_vertices()} vertices, degree 3, graphene structure)',
            fontsize=14
        )
        ax.set_aspect('equal')

        # Adds theoretical (n-1)/2 value as text
        theoretical_value = (self.number_vertices() - 1) / 2
        ax.text(
            0.02, 0.98,
            f'(n-1)/2 = {theoretical_value:.2f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        fig.tight_layout()

        # Saves figure if filename provided
        if fname:
            fig.savefig(fname, dpi=150, bbox_inches='tight')

        return fig

    def __repr__(self) -> str:
        """Returns string representation showing dimensions and vertex count."""
        return f"HexagonalLattice(rows={self.rows}, cols={self.cols}, vertices={self.number_vertices()})"

# Complete Graph Implementation
CompleteGraphVertexType = int

class CompleteGraph(Graph[CompleteGraphVertexType]):
    """
    Complete graph K_n where every vertex connects to every other vertex.
    
    Properties:
    - n vertices labeled 0 to n-1
    - Every vertex connects to all other vertices
    - Degree: n-1 (maximum possible)
    - Diameter: 1 (all vertices are neighbors)
    - Total edges: n(n-1)/2
    """
    
    def __init__(self, n: int):
        """
        Initializes complete graph with n vertices.
        
        Args:
            n: Number of vertices (must be >= 2)
        """
        if n < 2:
            raise ValueError("Complete graph requires at least 2 vertices.")
        self.n = n
    
    def get_start_vertex(self) -> CompleteGraphVertexType:
        """Returns starting vertex (vertex 0)."""
        return 0
    
    def get_adj_list(self, v: CompleteGraphVertexType) -> list[CompleteGraphVertexType]:
        """
        Returns all vertices except v (since all vertices are connected).
        
        Args:
            v: Current vertex
            
        Returns:
            List of all other vertices
        """
        # Returns all vertices except v
        return [i for i in range(self.n) if i != v]
    
    def number_vertices(self) -> int:
        """Returns number of vertices."""
        return self.n
    
    def plot_means_vars(self, summary_stats, *, fname=None):
        """
        Creates visualization of average discovery numbers.
        
        For complete graphs, uses simple bar chart with vertex indices.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extracts data
        vertices = sorted(summary_stats.keys())
        means = [summary_stats[v].mean for v in vertices]
        variances = [summary_stats[v].variance for v in vertices]
        stds = [np.sqrt(var) for var in variances]
        
        # Calculates expected value
        n = self.number_vertices()
        expected = (n - 1) / 2
        
        # Creates figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plots 1: Mean discovery numbers
        x_pos = np.arange(len(vertices))
        ax1.bar(x_pos, means, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axhline(y=expected, color='red', linestyle='--', linewidth=2, 
                    label=f'Expected (n-1)/2 = {expected:.1f}')
        ax1.set_xlabel('Vertex', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Discovery Number', fontsize=12, fontweight='bold')
        ax1.set_title(f'Complete Graph K_{n}: Mean Discovery Numbers', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plots 2: Standard deviations
        ax2.bar(x_pos, stds, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Vertex', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax2.set_title('Discovery Number Variability', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Saves figure if filename provided
        if fname:
            fig.savefig(fname, dpi=150, bbox_inches='tight')
        
        return fig

    def desc(self) -> str:
        """Returns short description for file naming."""
        return f"complete-k{self.n}"

    def __repr__(self) -> str:
        """Returns string representation."""
        return f"CompleteGraph(n={self.n}, vertices={self.n}, edges={self.n*(self.n-1)//2})"


# N-Dimensional Torus Grid Implementation
NDGridVertexType = tuple[int, ...]  # d-tuple of integers

class NDGrid(Graph[NDGridVertexType]):
    """
    N-dimensional torus grid graph (generalization of 2D torus to arbitrary dimensions).

    Properties:
    - d-dimensional grid with size points per dimension
    - Vertices: size^d total (all d-tuples with coordinates 0 to size-1)
    - Degree: 2d (constant, regardless of size)
    - Edges: Connect vertices differing by ±1 in exactly one coordinate (with wraparound)
    - Topology: Periodic boundary conditions in all dimensions (torus)

    Examples:
    - NDGrid(3, 10): 3D grid, 10×10×10 = 1000 vertices, degree 6
    - NDGrid(4, 5): 4D grid, 5×5×5×5 = 625 vertices, degree 8
    - NDGrid(10, 3): 10D grid, 3^10 = 59,049 vertices, degree 20
    """

    def __init__(self, dimension: int, size: int):
        """
        Initializes n-dimensional torus grid.

        Args:
            dimension: Number of dimensions (d >= 2)
            size: Number of points per dimension (size >= 2)
        """
        if dimension < 2:
            raise ValueError("NDGrid requires at least 2 dimensions.")
        if size < 2:
            raise ValueError("NDGrid requires at least 2 points per dimension.")

        self.dimension = dimension
        self.size = size
        self.d = dimension  # Alias for compatibility

    def get_start_vertex(self) -> NDGridVertexType:
        """Returns starting vertex (origin: all zeros)."""
        return tuple(0 for _ in range(self.dimension))

    def get_adj_list(self, v: NDGridVertexType) -> list[NDGridVertexType]:
        """
        Returns neighbors of vertex v.

        In a d-dimensional torus, each vertex has 2d neighbors:
        - d neighbors by incrementing one coordinate (with wraparound)
        - d neighbors by decrementing one coordinate (with wraparound)

        Args:
            v: Current vertex (d-tuple)

        Returns:
            List of 2d neighbor vertices
        """
        neighbors = []

        for dim in range(self.dimension):
            # Increment in this dimension (with wraparound)
            neighbor_plus = list(v)
            neighbor_plus[dim] = (v[dim] + 1) % self.size
            neighbors.append(tuple(neighbor_plus))

            # Decrement in this dimension (with wraparound)
            neighbor_minus = list(v)
            neighbor_minus[dim] = (v[dim] - 1) % self.size
            neighbors.append(tuple(neighbor_minus))

        return neighbors

    def number_vertices(self) -> int:
        """Returns total number of vertices."""
        return self.size ** self.dimension

    def plot_means_vars(self, summary_stats, *, fname=None):
        """
        Creates visualization of average discovery numbers.

        For high-dimensional grids, uses a simple bar chart sorted by mean value.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Extracts data
        vertices = sorted(summary_stats.keys())
        means = [summary_stats[v].mean for v in vertices]
        variances = [summary_stats[v].variance for v in vertices]
        stds = [np.sqrt(var) for var in variances]

        # Sorts by mean value for better visualization
        sorted_indices = np.argsort(means)
        sorted_means = [means[i] for i in sorted_indices]
        sorted_stds = [stds[i] for i in sorted_indices]

        # Calculates expected value
        n = self.number_vertices()
        expected = (n - 1) / 2

        # Creates figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plots 1: Mean discovery numbers (sorted)
        x_pos = np.arange(len(vertices))
        ax1.bar(x_pos, sorted_means, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axhline(y=expected, color='red', linestyle='--', linewidth=2,
                    label=f'Expected (n-1)/2 = {expected:.1f}')
        ax1.set_xlabel('Vertex (sorted by mean)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Discovery Number', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'{self.dimension}D Torus Grid ({self.size}^{self.dimension} = {n} vertices, degree {2*self.dimension})\n'
            f'Mean Discovery Numbers (sorted)',
            fontsize=14, fontweight='bold'
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plots 2: Standard deviations (sorted by mean)
        ax2.bar(x_pos, sorted_stds, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Vertex (sorted by mean)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax2.set_title('Discovery Number Variability', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Saves figure if filename provided
        if fname:
            fig.savefig(fname, dpi=150, bbox_inches='tight')

        return fig

    def desc(self) -> str:
        """Returns short description for file naming."""
        return f"ndgrid-{self.dimension}d-{self.size}^{self.dimension}"

    def __repr__(self) -> str:
        """Returns string representation."""
        return f"NDGrid(dimension={self.dimension}, size={self.size}, vertices={self.number_vertices()}, degree={2*self.dimension})"
