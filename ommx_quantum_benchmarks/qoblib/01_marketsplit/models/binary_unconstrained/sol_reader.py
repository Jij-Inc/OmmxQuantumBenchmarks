import re


def parse_sol_to_ordered_dict(file_path: str, constraint_count: int) -> dict:
    """
    Parse a .sol file and return a dictionary ordered by s#1, s#2, ..., followed by x#1, x#2, ...
    If the solution file lacks s variables, missing ones are filled with 0 based on the constraint count.

    Args:
        file_path: path to the .sol file
        constraint_count: number of constraints (i.e., expected number of s variables)

    Returns:
        dict: mapping of ordered indices to variable values
    """
    # First, collect all variables
    s_vars = {}  # s variables: {index: value}
    x_vars = {}  # x variables: {index: value}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Header lines
            if line.startswith("#"):
                # match Energy
                m = re.match(
                    r"#\s*Objective value\s*=\s*([-+]?\d*\.?\d+)", line, re.IGNORECASE
                )
                if m:
                    energy = float(m.group(1))
                    continue

            # Parse variable line
            parts = line.split()
            if len(parts) == 2:
                var_name, value = parts
                try:
                    var_value = float(value)

                    # Extract variable type and index
                    match = re.search(r"([a-zA-Z]+)#(\d+)", var_name)
                    if match:
                        var_type = match.group(1)
                        var_index = int(match.group(2))

                        if var_type == "x":
                            x_vars[var_index] = var_value

                except ValueError:
                    continue

    # Build dictionary in the specified order
    ordered_dict = {}
    current_index = 0

    # First add s variables (in index order)
    for s_idx in sorted(s_vars.keys()):
        ordered_dict[current_index] = s_vars[s_idx]
        current_index += 1

    # Then add x variables (in index order)
    for x_idx in sorted(x_vars.keys()):
        ordered_dict[current_index] = x_vars[x_idx]
        current_index += 1
    energy_dict = {"Energy": energy}
    return energy_dict, ordered_dict
