import gzip
from pathlib import Path
import numpy as np


def parse_topology_sol_file(sol_file_path: str) -> dict[str, object]:
    """Parse a topology solution file (.gph format).

    Args:
        sol_file_path (str): Path to the solution file (.gph or .gph.gz)

    Returns:
        dict: Dictionary containing solution variables and metadata
    """
    path = Path(sol_file_path)

    # Handle both compressed and uncompressed files
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            lines = f.readlines()
    else:
        with open(path, "r") as f:
            lines = f.readlines()

    # Parse header information
    diameter = None
    num_nodes = None
    num_edges = None
    edges = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Parse comment lines
        if line.startswith("c"):
            # Extract diameter from comment if available
            if "diameter" in line.lower():
                parts = line.lower().split()
                # Extract diameter value, which should be the last part
                diameter = int(parts[-1])

        # Parse problem line: "p edge nodes edges"
        elif line.startswith("p edge"):
            parts = line.split()
            if len(parts) >= 4:
                num_nodes = int(parts[2])
                num_edges = int(parts[3])

        # Parse edge lines: "e node1 node2"
        elif line.startswith("e"):
            parts = line.split()
            if len(parts) >= 3:
                # Convert node indices from 1-indexed to 0-indexed
                node1 = int(parts[1]) - 1
                node2 = int(parts[2]) - 1
                edges.append((node1, node2))

    return {
        "diameter": diameter,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edges": edges,
    }


def convert_topology_solution_to_jijmodeling_format(
    solution_data: dict[str, object], instance_data: dict[str, object]
) -> dict[str, object]:
    """Convert parsed topology solution data to JijModeling variable format for seidel_linear.

    Args:
        solution_data: Parsed solution data from parse_topology_sol_file
        instance_data: Instance data containing problem structure

    Returns:
        Dictionary in JijModeling format with seidel_linear variables.
    """
    nodes = instance_data["nodes"]
    max_diameter = instance_data["maxDiameter"]
    edges_list = solution_data["edges"]
    diameter = solution_data["diameter"]

    # Use Floyd-Warshall algorithm to compute all-pairs shortest paths
    # Initialize distance matrix using NumPy
    all_distances = np.full((nodes, nodes), np.inf)
    # Set distance for direct edges
    for i, j in edges_list:
        all_distances[i, j] = 1
        all_distances[j, i] = 1
    # Set diagonal to 0 (distance from node to itself)
    np.fill_diagonal(all_distances, 0)
    # Vectorized Floyd-Warshall algorithm
    for k in range(nodes):
        all_distances = np.minimum(
            all_distances, all_distances[:, k : k + 1] + all_distances[k : k + 1, :]
        )

    # Convert edges_list to set for O(1) lookup
    edges_set = set()
    for i, j in edges_list:
        edges_set.add((i, j))
        edges_set.add((j, i))

    # Initialize seidel_linear decision variables
    dist = [
        [[0 for _ in range(max_diameter)] for _ in range(nodes)] for _ in range(nodes)
    ]

    # Set dist variables according to Seidel semantics
    for s in range(nodes):
        for t in range(s + 1, nodes):
            is_adjacent = (s, t) in edges_set
            dist[s][t][0] = 1 if is_adjacent else 0

            if all_distances[s, t] != np.inf:
                actual_distance = int(all_distances[s, t])
                for d in range(1, max_diameter):
                    if d + 1 >= actual_distance:
                        dist[s][t][d] = 1
                    else:
                        dist[s][t][d] = 0
            else:
                for d in range(1, max_diameter):
                    dist[s][t][d] = 0

    # Initialize y variables (linearization variables)
    y = [
        [[[0 for _ in range(max_diameter)] for _ in range(nodes)] for _ in range(nodes)]
        for _ in range(nodes)
    ]

    # Compute y variables based on linearization definition
    for s in range(nodes):
        for t in range(s + 1, nodes):
            for u in range(nodes):
                if u != s and u != t:
                    for j in range(max_diameter - 1):
                        if s < u:
                            dist_su_j = dist[s][u][j]
                        elif u < s:
                            dist_su_j = dist[u][s][j]
                        else:
                            dist_su_j = 0

                        if u < t:
                            dist_ut_0 = dist[u][t][0]
                        elif t < u:
                            dist_ut_0 = dist[t][u][0]
                        else:
                            dist_ut_0 = 0

                        y[s][t][u][j] = dist_su_j * dist_ut_0

    return {"diameter": diameter, "dist": dist, "y": y}


def read_topology_solution_file_as_jijmodeling_format(
    sol_file_path: str, instance_data: dict[str, object]
) -> dict[str, object]:
    """Complete solution reading pipeline for Topology problems with seidel_linear formulation.

    Args:
        sol_file_path: Path to the solution file
        instance_data: Instance data from dat_reader

    Returns:
        Solution in JijModeling format ready for evaluation
    """

    # Parse the solution file
    solution_data = parse_topology_sol_file(sol_file_path)

    # Convert to JijModeling format
    jm_solution = convert_topology_solution_to_jijmodeling_format(
        solution_data, instance_data
    )

    return jm_solution
