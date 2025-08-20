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
    # Initialize distance matrix
    all_distances = [[float('inf')] * nodes for _ in range(nodes)]
    
    # Set distance for direct edges
    for i, j in edges_list:
        all_distances[i][j] = 1
        all_distances[j][i] = 1
    
    # Set diagonal to 0 (distance from node to itself)
    for i in range(nodes):
        all_distances[i][i] = 0
    
    # Floyd-Warshall algorithm
    for k in range(nodes):
        for i in range(nodes):
            for j in range(nodes):
                if all_distances[i][k] != float('inf') and all_distances[k][j] != float('inf'):
                    all_distances[i][j] = min(all_distances[i][j], all_distances[i][k] + all_distances[k][j])

    # Initialize seidel_linear decision variables
    # dist[s,t,d] - binary variable: 1 if shortest path of length exactly d between s,t
    dist = [
        [[0 for _ in range(max_diameter)] for _ in range(nodes)] for _ in range(nodes)
    ]

    # Set dist[s,t,d] = 1 based on ZIMPL definition
    # dist[s,t,d] = 1 if there is a shortest path of length d between s,t
    for s in range(nodes):
        for t in range(nodes):
            if s < t:  # Only for s < t (set F)
                if all_distances[s][t] != float('inf'):
                    actual_distance = int(all_distances[s][t])
                    if actual_distance < max_diameter:
                        dist[s][t][actual_distance] = 1
                
                # For adjacency (direct edges), set dist[s,t,0] = 1
                # This represents that nodes s,t are directly connected
                for edge in edges_list:
                    edge_s, edge_t = edge
                    if (edge_s == s and edge_t == t) or (edge_s == t and edge_t == s):
                        dist[s][t][0] = 1
                        break

    # Initialize y variables (linearization variables)
    # y[s,t,u,j] = dist[s,u,j] * dist[u,t,1] according to README.md
    y = [
        [[[0 for _ in range(max_diameter)] for _ in range(nodes)] for _ in range(nodes)]
        for _ in range(nodes)
    ]

    # Compute y variables based on the linearization definition: y[s,t,u,j] = dist[s,u,j] * dist[u,t,0]
    for s in range(nodes):
        for t in range(nodes):
            if s < t:  # Only for s < t (set F)
                for u in range(nodes):
                    if u != s and u != t:  # u in V \ {s,t}
                        for j in range(max_diameter - 1):  # j in {0, ..., max_diameter-2}
                            # y[s,t,u,j] = dist[min(s,u),max(s,u),j] * dist[min(u,t),max(u,t),0]
                            # Handle the min/max logic for s,u
                            if s < u:
                                dist_su = dist[s][u][j]
                            else:
                                dist_su = dist[u][s][j]
                            
                            # Handle the min/max logic for u,t  
                            if u < t:
                                dist_ut = dist[u][t][0]
                            else:
                                dist_ut = dist[t][u][0]
                                
                            y[s][t][u][j] = dist_su * dist_ut

    # Debug: Print some solution values
    print(f"DEBUG: diameter = {diameter}")
    print(f"DEBUG: nodes = {nodes}, max_diameter = {max_diameter}")
    print(f"DEBUG: edges = {edges_list}")
    
    # Count non-zero dist values
    dist_count = 0
    for s in range(nodes):
        for t in range(nodes):
            for d in range(max_diameter):
                if dist[s][t][d] == 1:
                    dist_count += 1
                    print(f"DEBUG: dist[{s},{t},{d}] = 1")
    print(f"DEBUG: Total non-zero dist values: {dist_count}")
    
    # Count non-zero y values
    y_count = 0
    for s in range(nodes):
        for t in range(nodes):
            for u in range(nodes):
                for j in range(max_diameter):
                    if y[s][t][u][j] == 1:
                        y_count += 1
    print(f"DEBUG: Total non-zero y values: {y_count}")
    
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
