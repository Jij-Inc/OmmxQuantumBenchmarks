"""Data loader for Steiner Tree Packing instances."""

from collections import Counter
from pathlib import Path


def load_steiner_instance(instance_path: str | Path) -> dict[str, object]:
    """Load a Steiner Tree Packing instance from the given directory.
    Node and net starts from 1, not 0.

    Args:
        instance_path (str | Path): Path to the instance directory

    Returns:
        dict[str, object]: Dictionary containing all the data needed for JijModeling
    """
    # Define the start indices for nodes and nets.
    node_start_index = 1
    net_start_index = 1

    instance_path = Path(instance_path)

    # Load parameters.
    # For instance, param.dat contains:
    # nodes 800
    # nets 8
    param_data = {}
    with open(instance_path / "param.dat", "r") as f:
        param_data = dict(
            line.strip().split()[:2]
            for line in f
            if line.strip()  # Only process non-empty lines
            and not line.startswith("#")  # Skip comments
            and len(line.strip().split()) >= 2  # Ensure at least two parts
        )
        param_data = {k: int(v) for k, v in param_data.items()}
    nodes = param_data["nodes"]
    nets = param_data["nets"]

    # Load terminals and their net assignments.
    # For instance, terms.dat contains:
    # # Node Net
    # 141   1
    # 220   1
    # 8   2
    # 300   2
    terms_data = []
    innet_data = {}
    with open(instance_path / "terms.dat", "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines.
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # Ensure there are at least two parts (node and net).
            if len(parts) >= 2:
                node = int(parts[0])
                net = int(parts[1])
                terms_data.append(node)
                innet_data[node] = net

    # Load roots and their net assignments
    # For instance, roots.dat contains:
    # # Node Net
    # 220   1
    #   8   2
    roots_data = []
    with open(instance_path / "roots.dat", "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines.
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # Ensure there are at least two parts (node and net).
            if len(parts) >= 2:
                node = int(parts[0])
                net = int(parts[1])
                roots_data.append(node)
                innet_data[node] = net

    # Load arcs
    # For instance, arcs.dat contains:
    # # Tail Head Cost
    # 1 2 10
    # 2 1 15
    arcs_data = []
    cost_dict = {}
    with open(instance_path / "arcs.dat", "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines.
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # Ensure there are at least three parts (tail, head, cost).
            if len(parts) >= 3:
                tail = int(parts[0])
                head = int(parts[1])
                cost = int(parts[2])
                arcs_data.append((tail, head))
                cost_dict[(tail, head)] = cost

    # Create derived sets
    # terms_data should contains all roots_data as well, but just in case we combine them.
    special_nodes = sorted(set(terms_data + roots_data))  # S = T + R
    terminal_nodes = sorted(set(terms_data) - set(roots_data))  # T = S - R
    normal_nodes = sorted(
        set(range(node_start_index, nodes + 1)) - set(special_nodes)
    )  # N = V - S

    # Create cost matrix
    cost_matrix = [[0 for _ in range(nodes + 1)] for _ in range(nodes + 1)]
    for (i, j), c in cost_dict.items():
        cost_matrix[i][j] = c

    # Create innet array for special nodes
    innet_array = [innet_data[node] for node in special_nodes]

    # Create nets count array.
    net_counts = Counter(innet_data[node] for node in terms_data)
    nets_count = [net_counts.get(net, 0) for net in range(net_start_index, nets + 1)]

    # Create network assignment arrays (optimized: list comprehensions)
    R_innet_array = [innet_data[root] for root in roots_data]
    T_innet_array = [innet_data[terminal] for terminal in terminal_nodes]

    return {
        "L": list(range(net_start_index, nets + 1)),  # Net indices
        "V": list(range(node_start_index, nodes + 1)),  # Vertex indices
        "S": special_nodes,  # Special nodes (terms + roots)
        "R": roots_data,  # Root nodes
        "A": arcs_data,  # Arc pairs
        "T": terminal_nodes,  # Terminal nodes (S - R)
        "N": normal_nodes,  # Normal nodes (V - S)
        "innet": innet_array,  # Net assignment for special nodes
        "R_innet": R_innet_array,  # Net assignment for root nodes
        "T_innet": T_innet_array,  # Net assignment for terminal nodes
        "cost": cost_matrix,  # Cost matrix
        "nets": nets_count,  # Number of terminals per net
        "nodes": nodes,  # Number of nodes
        "num_nets": nets,  # Number of nets
    }
