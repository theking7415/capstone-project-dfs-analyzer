import pickle
from collections import defaultdict
from typing import Any, TypeVar

import numpy as np
from scipy import stats

from mygraphs import Graph

RNG = np.random.default_rng(1832479182)
Vertex = TypeVar("Vertex")

# Note tha the storage on the stack in the implementation below can grow to |E|,
# the number of edges.  Usual DFS can be implemented with O(|V|) stack space.
# However, for Randomized DFS, we will have to pay with time if we do not store
# the randomized order of neighbors of each vertex.  In the implementation belo,
# we are using the stack to store this.
def rdfs(
    G: Graph[Vertex],
    v: Vertex,
    *,
    dist_stats: dict[Vertex, list[int]] | None = None,
    rng=RNG,
):
    visited: set[Vertex] = set()
    process_stack: list[tuple[Vertex, Vertex]] = []
    index: int = 0
    if dist_stats is None:
        visited_order: list[list[Vertex]] = []

    process_stack.append((v, v))

    while process_stack:
        current_node, parent = process_stack.pop()

        if current_node not in visited:
            visited.add(current_node)
            if dist_stats is None:
                visited_order.append([parent, current_node])
            else:
                dist_stats[current_node].append(index)
            index += 1

            neighbors = G.get_adj_list(current_node)
            unvisited_neighbors = [
                neighbor for neighbor in neighbors if neighbor not in visited
            ]

            rng.shuffle(unvisited_neighbors)

            for neighbor in unvisited_neighbors:
                process_stack.append((neighbor, current_node))

    if dist_stats is None:
        return visited_order


def collect_statistics(
    G: Graph[Vertex], num_samples: int, *, rng=RNG
) -> defaultdict[Vertex, Any]:
    dist_stats: defaultdict[Vertex, Any] = defaultdict(list)
    start_vertex = G.get_start_vertex()
    for i in range(num_samples):
        rdfs(G, start_vertex, dist_stats=dist_stats, rng=rng)

    return dist_stats


def get_summary_stats(dist_stats: defaultdict[Vertex, Any]) -> dict[Vertex, Any]:
    summary_stats = {}
    for v, vals in dist_stats.items():
        summary_stats[v] = stats.describe(vals)
    return summary_stats


def save_statistics(G: Graph[Vertex], *, samples: int):
    ans = collect_statistics(G, samples)
    with open(f"data/rdfs-{G.desc()}-{samples}-samples.pickle", "wb") as f:
        pickle.dump((G, ans), f)


def rdfs_till_first_backtrack(G: Graph[Vertex], v: Vertex, *, rng=RNG) -> int:
    """Return the number of vertices visited before first backtrack."""
    visited: set[Vertex] = set()
    process_stack: list[Vertex] = []
    index: int = 0

    process_stack.append(v)

    while process_stack:
        current_node = process_stack.pop()

        if current_node not in visited:
            visited.add(current_node)
            index += 1

            neighbors = G.get_adj_list(current_node)
            unvisited_neighbors = [
                neighbor for neighbor in neighbors if neighbor not in visited
            ]

            if not unvisited_neighbors:
                return index

            rng.shuffle(unvisited_neighbors)

            for neighbor in unvisited_neighbors:
                process_stack.append(neighbor)

    return index
