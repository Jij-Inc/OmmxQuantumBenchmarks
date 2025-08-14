"""Optimized data loader for Steiner Tree Packing instances."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


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

    # Load parameters using pandas
    param_df = pd.read_csv(
        instance_path / "param.dat",
        sep=r"\s+",
        comment="#",
        names=["param", "value"],
        dtype={"param": str, "value": int},
    )
    param_data = dict(zip(param_df["param"], param_df["value"]))
    num_nodes = param_data["nodes"]
    num_nets = param_data["nets"]

    # Load terminals and their net assignments using pandas
    terms_df = pd.read_csv(
        instance_path / "terms.dat",
        sep=r"\s+",
        comment="#",
        names=["node", "net"],
        dtype={"node": int, "net": int},
    )
    # Convert to 0-based indexing
    terms_df["node"] -= 1
    terms_df["net"] -= 1

    specials = terms_df["node"].tolist()
    special_to_net = dict(zip(terms_df["node"], terms_df["net"]))

    # Load roots and their net assignments using pandas
    roots_df = pd.read_csv(
        instance_path / "roots.dat",
        sep=r"\s+",
        comment="#",
        names=["node", "net"],
        dtype={"node": int, "net": int},
    )
    # Convert to 0-based indexing
    roots_df["node"] -= 1
    roots_df["net"] -= 1

    roots = roots_df["node"].tolist()
    # Verify consistency with terms.dat
    for _, row in roots_df.iterrows():
        assert special_to_net[row["node"]] == row["net"]

    # Load arcs using pandas
    arcs_df = pd.read_csv(
        instance_path / "arcs.dat",
        sep=r"\s+",
        comment="#",
        names=["tail", "head", "cost"],
        dtype={"tail": int, "head": int, "cost": int},
    )
    # Convert to 0-based indexing
    arcs_df["tail"] -= 1
    arcs_df["head"] -= 1

    arcs = list(zip(arcs_df["tail"], arcs_df["head"]))
    cost_dict = dict(zip(zip(arcs_df["tail"], arcs_df["head"]), arcs_df["cost"]))

    # Create derived sets using vectorized operations
    terminals = sorted(set(specials) - set(roots))  # T = S - R
    all_nodes = list(range(node_start_index, num_nodes + node_start_index))  # V
    normals = sorted(set(all_nodes) - set(specials))  # N = V - S
    nodes_without_roots = sorted(set(all_nodes) - set(roots))  # VNR = V - R

    # Create cost matrix using numpy for better performance
    cost_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    if cost_dict:
        indices = list(cost_dict.keys())
        values = list(cost_dict.values())
        rows, cols = zip(*indices)
        cost_matrix[rows, cols] = values
    cost_matrix = cost_matrix.tolist()

    # Create nets count array using vectorized operations
    nets = list(range(net_start_index, num_nets + net_start_index))
    net_counts = Counter(special_to_net[special] for special in specials)
    net_cardinality = [(net, net_counts.get(net, 0)) for net in nets]

    # Create network assignment arrays
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
