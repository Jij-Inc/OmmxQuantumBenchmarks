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
    node_start_index = 0
    net_start_index = 0

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
    num_nodes = param_data["nodes"]
    num_nets = param_data["nets"]

    # Load terminals and their net assignments.
    # For instance, terms.dat contains:
    # # Node Net
    # 141   1
    # 220   1
    # 8   2
    # 300   2
    specials = []  # terminal nodes + root nodes
    special_to_net = {}
    with open(instance_path / "terms.dat", "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines.
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # Ensure there are at least two parts (terminal node and net).
            if len(parts) >= 2:
                node = int(parts[0]) - 1
                net = int(parts[1]) - 1
                specials.append(node)
                special_to_net[node] = net

    # Load roots and their net assignments
    # For instance, roots.dat contains:
    # # Node Net
    # 220   1
    #   8   2
    roots = []
    with open(instance_path / "roots.dat", "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines.
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # Ensure there are at least two parts (root node and net).
            if len(parts) >= 2:
                root = int(parts[0]) - 1
                net = int(parts[1]) - 1
                roots.append(root)
                # terms.dat should contain all roots and terminals. Thus, it should be already stored in special_to_net.
                assert special_to_net[root] == net

    # Load arcs
    # For instance, arcs.dat contains:
    # # Tail Head Cost
    # 1 2 10
    # 2 1 15
    arcs = []
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
                tail = int(parts[0]) - 1
                head = int(parts[1]) - 1
                cost = int(parts[2])
                arcs.append((tail, head))
                cost_dict[(tail, head)] = cost

    # Create derived sets
    terminals = sorted(set(specials) - set(roots))  # T = S - R
    all_nodes = list(range(node_start_index, num_nodes + node_start_index))  # V
    normals = sorted(set(all_nodes) - set(specials))  # N = V - S
    nodes_without_roots = sorted(set(all_nodes) - set(roots))  # VNR = V - R

    # Create cost matrix
    cost_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for (i, j), c in cost_dict.items():
        cost_matrix[i][j] = c

    # Create nets count array.
    nets = list(range(net_start_index, num_nets + net_start_index))
    net_counts = Counter(special_to_net[special] for special in specials)
    net_cardinality = [(net, net_counts.get(net, 0)) for net in nets]

    # Create network assignment arrays (optimized: list comprehensions)
    root_to_net = [(root, special_to_net[root]) for root in roots]
    terminal_to_net = [(terminal, special_to_net[terminal]) for terminal in terminals]

    return {
        "L": nets,  # Net indices
        "V": all_nodes,  # Vertex indices
        "R": roots,  # Root nodes
        "A": arcs,  # Arc pairs
        "T": terminals,  # Terminal nodes (S - R)
        "N": normals,  # Normal nodes (V - S)
        "VNR": nodes_without_roots,
        "innetR": root_to_net,  # Net assignment for root nodes
        "innetT": terminal_to_net,  # Net assignment for terminal nodes
        "cost": cost_matrix,  # Cost matrix
        "netCardinality": net_cardinality,  # Number of terminals per net
    }
