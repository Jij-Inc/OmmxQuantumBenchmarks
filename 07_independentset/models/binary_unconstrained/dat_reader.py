def read_dimacs_gph(path: str):
    """
    Read an undirected graph in DIMACS format (.gph).
    The function converts vertices to 0-based indexing and removes self-loops
    or duplicate edges. Edge endpoints are normalized such that the smaller
    index comes first.

    Args:
        path (str): Path to the `.gph` DIMACS graph file.

    Returns:
        tuple[int, list[list[int]]]:
            - N: The number of vertices.
            - E: A list of edges, where each edge is represented as a
              2-element list `[u, v]` with 0-based vertex indices.
    """
    N = None
    E = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                # Skip empty and comment lines
                continue
            if line.startswith("p"):
                # Example: "p edge 17 39"
                parts = line.split()
                if len(parts) < 4 or parts[1] != "edge":
                    raise ValueError(f"Invalid p-line: {line}")
                N = int(parts[2])
                # M = int(parts[3])  # Optional: use this for edge count validation
            elif line.startswith("e"):
                # Example: "e 7 17"
                _, u, v = line.split()
                u0 = int(u) - 1  # Convert to 0-based
                v0 = int(v) - 1
                if u0 == v0:
                    # Ignore self-loops if present
                    continue
                # Normalize to (smaller, larger) to avoid duplicates
                if v0 < u0:
                    u0, v0 = v0, u0
                E.append([u0, v0])

    if N is None:
        raise ValueError("Missing 'p edge N M' header.")

    # Deduplicate edges just in case
    E = sorted(set(tuple(e) for e in E))
    E = [list(e) for e in E]

    # Basic validation: ensure all endpoints are in range
    if any(u < 0 or v < 0 or u > N - 1 or v > N - 1 for u, v in E):
        raise ValueError("Edge endpoint out of range for declared N.")

    return N, E
