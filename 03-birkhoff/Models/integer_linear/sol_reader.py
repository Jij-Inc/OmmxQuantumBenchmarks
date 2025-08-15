import re


def parse_sol_file(
    file_path: str, n: int
) -> tuple[dict[str, int], dict[str, int], dict[int, float]]:

    energy = None
    z_vars = {}
    x_vars = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Objective value
            m = re.match(
                r"#\s*Objective value\s*=\s*([-+]?\d*\.?\d+)", line, re.IGNORECASE
            )
            if m:
                energy = float(m.group(1))
                continue

            # z#k val
            m = re.match(r"z#(\d+)\s+([-+]?\d*\.?\d+)", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                z_vars[idx] = float(m.group(2))
                continue

            # x#k val
            m = re.match(r"x#(\d+)\s+([-+]?\d*\.?\d+)", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                x_vars[idx] = float(m.group(2))
                continue

    # Build solution_dict in required order: z1..zn, x1..xn
    solution_dict = {}
    for i in range(1, n + 1):
        solution_dict[i - 1] = z_vars.get(i, 0.0)
    for i in range(1, n + 1):
        solution_dict[n + i - 1] = x_vars.get(i, 0.0)

    energy_dict = {"Energy": energy}

    return energy_dict, solution_dict
