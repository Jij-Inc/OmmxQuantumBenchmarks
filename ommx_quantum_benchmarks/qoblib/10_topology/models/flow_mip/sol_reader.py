from collections import deque
import gzip
from pathlib import Path


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
    """Convert parsed topology solution data to JijModeling variable format.

    Args:
        solution_data: Parsed solution data from parse_topology_sol_file
        instance_data: Instance data containing problem structure

    Returns:
        Dictionary in JijModeling format.
    """
    nodes = instance_data["nodes"]
    edges_list = solution_data["edges"]
    diameter = solution_data["diameter"]

    # Initialize decision variables
    # z[i,j] - binary variable for edge existence (symmetric matrix)
    z = [[0 for _ in range(nodes)] for _ in range(nodes)]

    # Fill in the edges from solution and build adjacency list
    adjacency = [[] for _ in range(nodes)]
    for edge in edges_list:
        i, j = edge[0], edge[1]
        z[i][j] = 1
        z[j][i] = 1  # Symmetric
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Compute shortest paths using BFS for each source node
    shortest_path = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for s in range(nodes):
        # BFS from source s
        distances = [-1] * nodes
        distances[s] = 0
        queue = deque([s])

        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if distances[neighbor] == -1:  # Not visited
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # Fill SP matrix
        for t in range(nodes):
            if s != t:
                shortest_path[s][t] = (
                    distances[t] if distances[t] != -1 else float("inf")
                )

    # x[s,t,i,j] - flow variables based on shortest paths
    x = [
        [[[0 for _ in range(nodes)] for _ in range(nodes)] for _ in range(nodes)]
        for _ in range(nodes)
    ]

    # For each source-target pair, compute the flow that realizes the shortest path
    for s in range(nodes):
        for t in range(nodes):
            if s != t and shortest_path[s][t] != float("inf"):
                # BFS to find shortest path and construct flow
                distances = [-1] * nodes
                parent = [-1] * nodes
                distances[s] = 0
                queue = deque([s])

                while queue:
                    current = queue.popleft()
                    if current == t:
                        break
                    for neighbor in adjacency[current]:
                        if distances[neighbor] == -1:
                            distances[neighbor] = distances[current] + 1
                            parent[neighbor] = current
                            queue.append(neighbor)

                # Trace back path and set flow variables
                if distances[t] != -1:
                    current = t
                    while parent[current] != -1:
                        prev = parent[current]
                        # Set flow from prev to current for commodity (s,t)
                        x[s][t][prev][current] = 1
                        current = prev

    return {"diameter": diameter, "z": z, "SP": shortest_path, "x": x}


def read_topology_solution_file_as_jijmodeling_format(
    sol_file_path: str, instance_data: dict[str, object]
) -> dict[str, object]:
    """Complete solution reading pipeline for Topology problems.

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
