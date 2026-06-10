import re

# Number of units per asset and position (ub in parameter_u3_c10.zpl).
UB = 3
# Number of position signs: tau = 1 (long, l = 0) and tau = -1 (short, l = 1).
NUM_SIGNS = 2
# Binary expansion widths of the two slack variables (CS1, CS2 in
# parameter_u3_c10.zpl).
NUM_Y_SLACKS = 4
NUM_S_SLACKS = 7

_OBJECTIVE_RE = re.compile(
    r"#\s*Objective value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE
)


def parse_sol_file(
    lines, symbols: list[str], num_periods: int
) -> tuple[float, dict[tuple[str, tuple[int, ...]], float]]:
    """Parse a QOBLIB portfolio solution file in Gurobi .sol format.

    The solution files under `solutions/bqp` of the original QOBLIB repository
    contain one variable per line (`<name> <value>`) plus an
    `# Objective value = ...` header.  Variable names come from ZIMPL's LP
    output:

    - `y#<k>#<t>` and `s2#<c>#<t>` for the slack variables,
    - `x$<symbol>#<m>#<tau>#<t>` for the asset variables, where ZIMPL replaces
      the invalid character '-' with '_', so tau = -1 appears as `_1`,
    - mangled names like `x$AAPL#1#_1#0@a` for long (truncated) names.  ZIMPL
      truncates such names, appending '@' and the
      0-based variable index in declaration order
      (`var x[SX*TX]` with SX = S * {1..ub} * {1,-1}, day fastest), which we
      use to recover the subscripts.

    Note that the solution files under `solutions/uqo` are NOT parseable: they
    are written in the abs2 solver's internal variable numbering whose mapping
    (ZIMPL .tbl files) was not published.  The BQP solutions use the same
    variable space and satisfy both equality constraints, so they evaluate on
    the UQO model with vanishing penalty terms.

    Variables omitted by the solver are filled with 0, since Gurobi .sol
    writers commonly drop zero-valued variables; the returned dict therefore
    always covers the full model.

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
                if decl_index >= len(x_decl):
                    raise ValueError(
                        f"Declaration index {decl_index} from {name!r} is out of "
                        f"range for {len(x_decl)} x variables; the symbol list or "
                        f"num_periods ({num_periods}) is likely wrong."
                    )
                subscripts = x_decl[decl_index]
            else:
                _, symbol, m, tau, t = name.replace("$", "#").split("#")
                # ZIMPL replaces the invalid character '-' with '_', so a short
                # (non-truncated) tau = -1 name carries "_1" rather than "-1".
                l = 0 if int(tau.replace("_", "-")) == 1 else 1
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

    # Gurobi .sol writers may omit zero-valued variables, so fewer entries than
    # the full model is valid; only more than expected (e.g. a wrong
    # num_periods or an unexpected variable) is an error. Fill the omitted
    # variables with 0 so the returned assignment covers the whole model.
    expected = len(x_decl) + NUM_Y_SLACKS * num_periods + NUM_S_SLACKS * num_periods
    if len(values) > expected:
        raise ValueError(
            f"Expected at most {expected} variables in solution file, "
            f"but got {len(values)}."
        )
    for subscripts in x_decl:
        values.setdefault(("x", subscripts), 0.0)
    for k in range(NUM_Y_SLACKS):
        for t in range(num_periods):
            values.setdefault(("y", (k, t)), 0.0)
    for c in range(NUM_S_SLACKS):
        for t in range(num_periods):
            values.setdefault(("s", (c, t)), 0.0)
    return objective, values
