import re

# Number of units per asset and position (ub in parameter_u3_c10.zpl).
UB = 3
# Number of position signs: tau = 1 (long, l = 0) and tau = -1 (short, l = 1).
NUM_SIGNS = 2

_OBJECTIVE_RE = re.compile(r"#\s*Objective value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_sol_file(
    lines, symbols: list[str], num_periods: int
) -> tuple[float, dict[tuple[str, tuple[int, ...]], float]]:
    """Parse a QOBLIB portfolio solution file in Gurobi .sol format.

    The solution files under `solutions/bqp` of the original QOBLIB repository
    contain one variable per line (`<name> <value>`) plus an
    `# Objective value = ...` header.  Variable names come from ZIMPL's LP
    output:

    - `y#<k>#<t>` and `s2#<c>#<t>` for the slack variables,
    - `x$<symbol>#<m>#<tau>#<t>` for the asset variables with tau = 1,
    - mangled names like `x$AAPL#1#_1#0@a` for tau = -1.  ZIMPL replaces the
      invalid character '-' and truncates long names, appending '@' and the
      0-based variable index in declaration order
      (`var x[SX*TX]` with SX = S * {1..ub} * {1,-1}, day fastest), which we
      use to recover the subscripts.

    Args:
        lines: iterable of text lines of the solution file.
        symbols: stock symbols in price-file order (defines the asset index).
        num_periods: number of allocation days T.

    Returns:
        A tuple of:
        - the objective value declared in the file header,
        - dict mapping (variable name, subscripts) to the variable value,
          with subscripts (i, m, l, t) for "x" and (k, t) for "y" / "s".
    """
    sym_idx = {s: i for i, s in enumerate(symbols)}
    # Declaration order of x[SX*TX]: asset, unit, sign, day (day fastest).
    x_decl = [
        (i, m, l, t)
        for i in range(len(symbols))
        for m in range(UB)
        for l in range(NUM_SIGNS)
        for t in range(num_periods)
    ]

    objective = None
    values: dict[tuple[str, tuple[int, ...]], float] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = _OBJECTIVE_RE.match(line)
            if match:
                objective = float(match.group(1))
            continue

        name, value_str = line.rsplit(None, 1)
        name = name.strip()
        value = float(value_str)

        if name.startswith("x$"):
            if "@" in name:
                # Mangled (truncated) name: decode via the declaration index.
                decl_index = int(name.rsplit("@", 1)[1], 16)
                subscripts = x_decl[decl_index]
            else:
                _, symbol, m, tau, t = name.replace("$", "#").split("#")
                l = 0 if int(tau) == 1 else 1
                subscripts = (sym_idx[symbol], int(m) - 1, l, int(t))
            values[("x", subscripts)] = value
        elif name.startswith("y#"):
            _, k, t = name.split("#")
            values[("y", (int(k), int(t)))] = value
        elif name.startswith("s2#"):
            _, c, t = name.split("#")
            values[("s", (int(c), int(t)))] = value
        else:
            raise ValueError(f"Unrecognized variable name in solution file: {name}")

    if objective is None:
        raise ValueError("No '# Objective value = ...' header found.")
    expected = len(x_decl) + 4 * num_periods + 7 * num_periods
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} variables in solution file, but got {len(values)}."
        )
    return objective, values
