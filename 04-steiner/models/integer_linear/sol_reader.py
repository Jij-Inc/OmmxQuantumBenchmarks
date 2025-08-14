"""Optimized solution reader for Steiner Tree Packing Problem."""

from collections import defaultdict
from io import StringIO
import re

import numpy as np
import pandas as pd


def parse_steiner_sol_file(sol_file_path: str) -> dict[str, object]:
    """Parse a Steiner Tree Packing solution file using optimized pandas operations.

    Args:
        sol_file_path (str): Path to the solution file

    Returns:
        dict: Dictionary containing solution variables and metadata
    """

    solution_data = {
        "objective": None,
        "used_arcs": [],  # List of (tail, head, net) tuples from solution
    }

    # Read the entire file to extract objective and process arcs
    with open(sol_file_path, "r") as file:
        content = file.read().strip()

    # Parse objective value (if present) - look for "Cost:" pattern
    obj_match = re.search(r"cost[:\s]+([0-9.-]+)", content, re.IGNORECASE)
    if obj_match:
        solution_data["objective"] = float(obj_match.group(1))

    # Extract arc data lines (skip comments and empty lines)
    lines = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.startswith("#") and len(line.split()) >= 3
    ]

    if lines:
        # Use pandas for fast parsing of arc data
        csv_buffer = StringIO("\n".join(lines))

        arcs_df = pd.read_csv(
            csv_buffer,
            sep=r"\s+",
            names=["tail", "head", "net"],
            dtype={"tail": int, "head": int, "net": int},
        )

        # Convert to 0-based indexing and create tuples
        arcs_df["tail"] -= 1
        arcs_df["head"] -= 1
        arcs_df["net"] -= 1

        solution_data["used_arcs"] = list(
            zip(arcs_df["tail"], arcs_df["head"], arcs_df["net"])
        )

    return solution_data


def convert_steiner_solution_to_jijmodeling_format(
    solution_data: dict[str, object], instance_data: dict[str, object]
) -> dict[str, object]:
    """Convert parsed Steiner solution data to JijModeling variable format using optimized operations.

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

    # Build terminal to net mapping using vectorized operations
    terminal_to_net = dict(zip(terminals, [x[1] for x in instance_data["innetT"]]))

    # Build root to net mapping using vectorized operations
    root_to_net = dict(zip(roots, [x[1] for x in instance_data["innetR"]]))

    num_nodes = len(nodes)
    num_terminals = len(terminals)
    num_nets = len(nets)
    num_roots = len(roots)

    # Initialize arrays using numpy for better performance
    x_values = np.zeros((num_nodes, num_nodes, num_terminals), dtype=float)
    y_values = np.zeros((num_nodes, num_nodes, num_nets), dtype=float)
    z_values = np.zeros((num_roots, num_terminals), dtype=float)

    # Build adjacency lists for each net using defaultdict for efficiency
    net_graphs = defaultdict(lambda: defaultdict(list))
    for tail, head, net in solution_data["used_arcs"]:
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

    # Create terminal index lookup for faster access
    terminal_to_idx = {terminal: idx for idx, terminal in enumerate(terminals)}

    # Process used arcs from solution
    for tail, head, net in solution_data["used_arcs"]:
        # Set y variable: y[tail, head, net] = 1.0
        y_values[tail, head, net] = 1.0

        # For x variables, determine which terminals this arc actually serves
        # by finding terminals reachable from the head of this arc
        reachable_terminals = find_reachable_terminals_from_arc(net, tail, head)

        for terminal in reachable_terminals:
            t_idx = terminal_to_idx[terminal]
            x_values[tail, head, t_idx] = 1.0

    # Calculate z values using vectorized operations
    root_nets = np.array([root_to_net[root] for root in roots])
    terminal_nets = np.array([terminal_to_net[terminal] for terminal in terminals])

    # Use broadcasting to create a matrix comparison
    z_values = (root_nets[:, np.newaxis] == terminal_nets[np.newaxis, :]).astype(float)

    # Convert numpy arrays back to lists for compatibility
    jm_solution = {
        "x": x_values.tolist(),
        "y": y_values.tolist(),
        "z": z_values.tolist(),
    }

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
