"""Data loader for Steiner Tree Packing instances."""

from pathlib import Path


def load_steiner_instance(instance_path):
    """Load a Steiner Tree Packing instance from the given directory.
    
    Args:
        instance_path (str): Path to the instance directory
    
    Returns:
        dict: Dictionary containing all the data needed for JijModeling
    """
    instance_path = Path(instance_path)
    
    # Load parameters
    param_data = {}
    with open(instance_path / "param.dat", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                param_data[parts[0]] = int(parts[1])
    
    nodes = param_data["nodes"]
    nets = param_data["nets"]
    
    # Load terminals and their net assignments
    terms_data = []
    innet_data = {}
    with open(instance_path / "terms.dat", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                node = int(parts[0])
                net = int(parts[1])
                terms_data.append(node)
                innet_data[node] = net
    
    # Load roots and their net assignments
    roots_data = []
    with open(instance_path / "roots.dat", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                node = int(parts[0])
                net = int(parts[1])
                roots_data.append(node)
                innet_data[node] = net
    
    # Load arcs
    arcs_data = []
    cost_dict = {}
    with open(instance_path / "arcs.dat", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                tail = int(parts[0])
                head = int(parts[1]) 
                cost = int(parts[2])
                arcs_data.append((tail, head))
                cost_dict[(tail, head)] = cost
    
    # Create derived sets
    special_nodes = sorted(set(terms_data + roots_data))
    terminal_nodes = sorted(set(terms_data) - set(roots_data))  # T = S - R
    normal_nodes = sorted(set(range(1, nodes + 1)) - set(special_nodes))  # N = V - S
    
    # Create cost matrix
    cost_matrix = [[0 for _ in range(nodes + 1)] for _ in range(nodes + 1)]  # 1-indexed
    for (i, j), c in cost_dict.items():
        cost_matrix[i][j] = c
    
    # Create innet array for special nodes (terms + roots)
    innet_array = []
    for node in special_nodes:
        innet_array.append(innet_data[node])
    
    # Count terminals per net
    nets_count = []
    for net in range(1, nets + 1):
        count = sum(1 for node in terms_data if innet_data[node] == net)
        nets_count.append(count)
    
    # Create network assignment arrays for roots and terminals
    R_innet_array = []
    for root in roots_data:
        R_innet_array.append(innet_data[root])
    
    T_innet_array = []
    for terminal in terminal_nodes:
        T_innet_array.append(innet_data[terminal])
    
    return {
        "L": list(range(1, nets + 1)),  # Net indices
        "V": list(range(1, nodes + 1)),  # Vertex indices
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
        "nodes": nodes,
        "num_nets": nets
    }


if __name__ == "__main__":
    # Test the loader
    instance_path = "/Users/keisukesato/dev/git/OMMX-OBLIB/04-steiner/instances/stp_s020_l2_t3_h2_rs24098"
    data = load_steiner_instance(instance_path)
    
    print(f"Nodes: {data['nodes']}")
    print(f"Nets: {data['num_nets']}")
    print(f"Special nodes (S): {len(data['S'])}")
    print(f"Root nodes (R): {len(data['R'])}")
    print(f"Terminal nodes (T): {len(data['T'])}")
    print(f"Normal nodes (N): {len(data['N'])}")
    print(f"Arcs (A): {len(data['A'])}")
    print(f"Terminals per net: {data['nets']}")