import re


def parse_vrp_solution_file(
    file_path: str, depot: int, demand: list[int], n: int
) -> tuple[float, dict[int, float]]:
    """
    Parse a VRP solution file containing routes and cost into a solution dictionary.

    The solution dictionary assigns values to decision variables under the assumed
    ordering:
      - x[i,j] flattened in row-major order: id = i * n + j
      - y[i] sequential after x: id = n * n + i

    Args:
        file_path (str): Path to the solution file.
        depot (int): Index of the depot node.
        demand (list[int]): Demand values for each node (length = n).
        n (int): Total number of nodes, including the depot.

    Returns:
        tuple[float, dict[int, float]]:
            - objective_value (float): The objective cost parsed from the file.
            - solution_dict (dict[int, float]): Mapping from variable ID to its value.
              IDs are assigned as:
                * 0 .. n*n-1 → x[i,j]
                * n*n .. n*n+n-1 → y[i]
    """
    sol = {i: 0.0 for i in range(n * n + n)}
    objective_value = 0.0

    def xid(i, j):
        return i * n + j

    def yid(i):
        return n * n + i

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Route line
            m = re.match(r"^Route\s+#\d+:\s+(.*)$", line)
            if m:
                nodes = [int(x) for x in m.group(1).split()]
                prev = depot
                load = 0
                for node in nodes:
                    sol[xid(prev, node)] = 1.0
                    load += demand[node]
                    sol[yid(node)] = load
                    prev = node
                sol[xid(prev, depot)] = 1.0
                continue

            # Cost line
            m = re.match(r"^Cost\s+(\d+)", line)
            if m:
                objective_value = float(m.group(1))
                continue

    return objective_value, sol
