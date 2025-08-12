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
        "total_vars": 0,
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
                    tail = int(parts[0])
                    head = int(parts[1])
                    net = int(parts[2])
                    solution_data["used_arcs"].append((tail, head, net))
                except ValueError:
                    # Skip lines that don't contain valid integers
                    continue
        # Count total variables used
        solution_data["total_vars"] = len(solution_data["used_arcs"])

    return solution_data


def convert_steiner_solution_to_jijmodeling_format(
    solution_data: dict[str, object], instance_data: dict[str, object]
) -> dict[str, object]:
    """Convert parsed Steiner solution data to JijModeling variable format.

    Args:
        solution_data: Parsed solution data from parse_steiner_sol_file
        instance_data: Instance data containing problem structure

    Returns:
        Dictionary in JijModeling format for solution evaluation
    """

    # Get problem dimensions
    arcs = instance_data["A"]  # List of (tail, head) tuples
    terminals = instance_data["T"]  # List of terminal nodes
    nets = instance_data["L"]  # List of net indices

    # Build terminal to net mapping
    terminal_to_net = {}  # Map terminal to its net
    for i, terminal in enumerate(terminals):
        terminal_to_net[terminal] = instance_data["T_innet"][i]

    num_arcs = len(arcs)
    num_terminals = len(terminals)
    num_nets = len(nets)

    # Initialize x and y arrays
    x_values = [[0.0 for _ in range(num_terminals)] for _ in range(num_arcs)]
    y_values = [[0.0 for _ in range(num_nets)] for _ in range(num_arcs)]

    # Process used arcs from solution
    for tail, head, net in solution_data["used_arcs"]:
        # Find arc index
        try:
            arc_idx = arcs.index((tail, head))
        except ValueError:
            # Arc not found in the arc list, skip
            continue

        # Set y variable: y[arc_idx, net-1] = 1 (net is 1-indexed)
        if 1 <= net <= num_nets:
            y_values[arc_idx][net - 1] = 1.0

        # Set x variables for all terminals in this net
        for t_idx, terminal in enumerate(terminals):
            if terminal_to_net[terminal] == net:
                x_values[arc_idx][t_idx] = 1.0

    jm_solution = {"x": x_values, "y": y_values}

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
