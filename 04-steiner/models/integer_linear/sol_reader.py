"""Solution reader for Steiner Tree Packing Problem."""

import re


def parse_steiner_sol_file(sol_file_path: str) -> dict[str, object]:
    """Parse a Steiner Tree Packing solution file.

    Args:
        sol_file_path (str): Path to the solution file

    Returns:
        dict: Dictionary containing solution variables and metadata
    """

    solution_data = {
        "objective": None,
        "used_arcs": [],  # List of (tail, head, net) tuples from solution
    }

    with open(sol_file_path, "r") as file:
        content = file.read().strip()

        # Parse objective value (if present) - look for "Cost:" pattern
        obj_match = re.search(r"cost[:\s]+([0-9.-]+)", content, re.IGNORECASE)
        if obj_match:
            solution_data["objective"] = float(obj_match.group(1))

        # Parse solution arcs
        # Format: "tail head net" (e.g., "31 11 1")
        for line in content.split("\n"):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse arc data: tail head net
            parts = line.split()
            if len(parts) >= 3:
                try:
                    tail = int(parts[0]) - 1
                    head = int(parts[1]) - 1
                    net = int(parts[2]) - 1
                    solution_data["used_arcs"].append((tail, head, net))
                except ValueError:
                    # Skip lines that don't contain valid integers
                    continue

    return solution_data


def convert_steiner_solution_to_jijmodeling_format(
    solution_data: dict[str, object], instance_data: dict[str, object]
) -> dict[str, object]:
    """Convert parsed Steiner solution data to JijModeling variable format.

    Computes all decision variables from the solution file, including auxiliary variables
    that are determined by the instance data and primary variables (x, y).

    Args:
        solution_data: Parsed solution data from parse_steiner_sol_file
        instance_data: Instance data containing problem structure

    Returns:
        Dictionary in JijModeling format containing:
        - x: arc-terminal flow variables
        - y: arc-net usage variables
    """

    # Get problem dimensions
    nodes = instance_data["V"]  # List of nodes
    terminals = instance_data["T"]  # List of terminal nodes
    nets = instance_data["L"]  # List of net indices
    roots = instance_data["R"]  # List of root nodes

    # Build terminal to net mapping
    terminal_to_net = {}  # Map terminal to its net
    for i, terminal in enumerate(terminals):
        terminal_to_net[terminal] = instance_data["innetT"][i][1]

    # Build root to net mapping
    root_to_net = {}  # Map root to its net
    for i, root in enumerate(roots):
        root_to_net[root] = instance_data["innetR"][i][1]

    num_nodes = len(nodes)
    num_terminals = len(terminals)
    num_nets = len(nets)
    num_roots = len(roots)

    # Initialize x, y, and z arrays to match model.py format
    x_values = [
        [[0.0 for _ in range(num_terminals)] for _ in range(num_nodes)]
        for _ in range(num_nodes)
    ]
    y_values = [
        [[0.0 for _ in range(num_nets)] for _ in range(num_nodes)]
        for _ in range(num_nodes)
    ]
    z_values = [[0.0 for _ in range(num_terminals)] for _ in range(num_roots)]

    # Build adjacency lists for each net to determine reachability
    net_graphs = {}  # net -> {node: [neighbors]}
    for tail, head, net in solution_data["used_arcs"]:
        if net not in net_graphs:
            net_graphs[net] = {}
        if tail not in net_graphs[net]:
            net_graphs[net][tail] = []
        if head not in net_graphs[net]:
            net_graphs[net][head] = []
        net_graphs[net][tail].append(head)

    # For each net, determine which terminals each arc serves
    def find_reachable_terminals_from_arc(net, tail, head):
        """Find terminals reachable from the head of arc (tail, head) in the given net."""
        if net not in net_graphs:
            return []
        
        reachable_terminals = []
        visited = set()
        stack = [head]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            
            # Check if this node is a terminal in the current net
            if node in terminals and terminal_to_net[node] == net:
                reachable_terminals.append(node)
            
            # Continue traversal
            if node in net_graphs[net]:
                for neighbor in net_graphs[net][node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return reachable_terminals

    # Process used arcs from solution
    for tail, head, net in solution_data["used_arcs"]:
        # Set y variable: y[tail, head, net] = 1.0
        y_values[tail][head][net] = 1.0

        # For x variables, determine which terminals this arc actually serves
        # by finding terminals reachable from the head of this arc
        reachable_terminals = find_reachable_terminals_from_arc(net, tail, head)
        
        for terminal in reachable_terminals:
            t_idx = terminals.index(terminal)
            x_values[tail][head][t_idx] = 1.0

    # Calculate z values: z[r, t] = 1 if root_innet[r] == terminal_innet[t]
    for r_idx, root in enumerate(roots):
        for t_idx, terminal in enumerate(terminals):
            if root_to_net[root] == terminal_to_net[terminal]:
                z_values[r_idx][t_idx] = 1.0

    jm_solution = {"x": x_values, "y": y_values, "z": z_values}

    return jm_solution


def read_steiner_solution_file_as_jijmodeling_format(
    sol_file_path: str, instance_data: dict[str, object]
) -> dict[str, object]:
    """Complete solution reading pipeline for Steiner Tree Packing problems.

    Args:
        sol_file_path: Path to the solution file
        instance_data: Instance data from dat_reader

    Returns:
        Solution in JijModeling format ready for evaluation
    """

    # Parse the solution file
    solution_data = parse_steiner_sol_file(sol_file_path)

    # Convert to JijModeling format
    jm_solution = convert_steiner_solution_to_jijmodeling_format(
        solution_data, instance_data
    )

    return jm_solution
