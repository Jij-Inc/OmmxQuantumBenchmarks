import re


def parse_solution_as_dict(
    file_path: str, n: int, n_slots: int
) -> tuple[float, dict[int, float]]:
    """Parses a `.sol` solution file into an objective value and a variable assignment dictionary.

    The variables are returned in a strict order:
        - ``bh[i,s]`` for all teams and all slots → size = n * n_slots
        - ``ba[i,s]`` for all teams and all slots → size = n * n_slots
        - ``x[i,j,s]`` for ordered pairs (i != j) and all slots → size = n * (n-1) * n_slots

    Note:
        The `.sol` file provides variables in `isj` order, but they are converted
        to `ijs` order to align with OMMX format.

    Args:
        file_path (str): Path to the `.sol` file.
        n (int): Number of teams.
        n_slots (int): Number of slots.

    Returns:
        tuple[float, dict[int, float]]:
            - Objective value parsed from the file.
            - Dictionary mapping variable IDs (int) to their assigned values (float)
              in the required strict ordering.
    """

    sol = {}
    counter = 0
    obj_val = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    # regex patterns
    pat_var = re.compile(r"^([a-zA-Z0-9#]+)\s+([-+]?\d+\.?\d*)")
    pat_obj = re.compile(r"#\s*Objective value\s*=\s*([-+]?\d+\.?\d*)")

    raw = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Objective value
        mobj = pat_obj.match(line)
        if mobj:
            obj_val = float(mobj.group(1))
            continue

        # Variables
        m = pat_var.match(line)
        if m:
            var, val = m.groups()
            raw[var] = float(val)

    # --------------------------
    # bh[i,s]
    # --------------------------
    for i in range(n):
        for s in range(n_slots):
            name = f"bh#{i}#{s}"
            sol[counter] = raw.get(name, 0.0)
            counter += 1

    # --------------------------
    # ba[i,s]
    # --------------------------
    for i in range(n):
        for s in range(n_slots):
            name = f"ba#{i}#{s}"
            sol[counter] = raw.get(name, 0.0)
            counter += 1

    # --------------------------
    # x[i,j,s]   (.sol's oreder is isj，need to be ijs)
    # --------------------------
    for i in range(n):
        for s in range(n_slots):
            for j in range(n):
                name = f"x#{i}#{j}#{s}"
                sol[counter] = raw.get(name, 0.0)
                counter += 1

    return obj_val, sol
