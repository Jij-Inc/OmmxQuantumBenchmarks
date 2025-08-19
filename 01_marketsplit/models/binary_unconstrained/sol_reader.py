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

            # Skip comments and empty lines
            if line.startswith("#") or not line:
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

                        # if var_type == 's':
                        #     s_vars[var_index] = var_value
                        if var_type == "x":
                            x_vars[var_index] = var_value

                except ValueError:
                    continue

    # If there are fewer s variables than expected, fill missing ones with 0
    if len(s_vars) < constraint_count:
        print(
            f"Warning: insufficient number of s variables in solution file ({len(s_vars)}/{constraint_count}); filling missing s variables with 0"
        )
        for i in range(1, constraint_count + 1):
            if i not in s_vars:
                s_vars[i] = 0.0

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

    return ordered_dict


def show_variable_mapping(file_path: str, constraint_count: int):
    """
    Display the mapping from dictionary index to variable name and value.

    Args:
        file_path: path to the .sol file
        constraint_count: number of constraints (expected s variable count)
    """
    s_vars = {}
    x_vars = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue

            parts = line.split()
            if len(parts) == 2:
                var_name, value = parts
                try:
                    var_value = float(value)
                    match = re.search(r"([a-zA-Z]+)#(\d+)", var_name)
                    if match:
                        var_type = match.group(1)
                        var_index = int(match.group(2))

                        if var_type == "s":
                            s_vars[var_index] = var_value
                        elif var_type == "x":
                            x_vars[var_index] = var_value
                except ValueError:
                    continue

    # Fill missing s variables with 0
    for i in range(1, constraint_count + 1):
        if i not in s_vars:
            s_vars[i] = 0.0

    print("Variable mapping:")
    print("Dict Index -> Variable Name -> Value")
    print("-" * 35)

    current_index = 0
    # s variables
    for s_idx in sorted(s_vars.keys()):
        status = "(filled with 0)" if s_vars[s_idx] == 0.0 else ""
        print(f"{current_index:2d} -> s#{s_idx} -> {s_vars[s_idx]} {status}")
        current_index += 1

    # x variables
    for x_idx in sorted(x_vars.keys()):
        print(f"{current_index:2d} -> x#{x_idx} -> {x_vars[x_idx]}")
        current_index += 1
