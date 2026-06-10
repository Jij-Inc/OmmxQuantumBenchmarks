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


def _slack_subscripts(name: str, width: int, num_periods: int) -> tuple[int, int]:
    """Parse a slack variable name `y#<k>#<t>` / `s2#<c>#<t>` into subscripts."""
    parts = name.split("#")
    try:
        if len(parts) != 3:
            raise ValueError
        k, t = int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(
            f"Unrecognized slack variable name in solution file: {name!r}"
        ) from None
    if not 0 <= k < width or not 0 <= t < num_periods:
        raise ValueError(f"Subscripts out of range in solution file variable {name!r}.")
    return (k, t)


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
    always covers the full model.  Malformed lines, non-binary values, unknown
    symbols, out-of-range subscripts, and duplicate variables raise ValueError
    so that a corrupt solution file can never be mapped silently.

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

        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed line in solution file: {line!r}")
        name, value_str = parts
        name = name.strip()
        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(
                f"Malformed value in solution file line: {line!r}"
            ) from None
        # All model variables are binary; tolerate solver output like
        # 0.9999999996 but reject anything that is not a 0/1 within tolerance.
        rounded = round(value)
        if abs(value - rounded) > 1e-6 or rounded not in (0, 1):
            raise ValueError(
                f"Non-binary value {value} for variable {name!r} in solution file."
            )
        value = float(rounded)

        if name.startswith("x$"):
            if "@" in name:
                # Mangled (truncated) name: decode via the declaration index.
                try:
                    decl_index = int(name.rsplit("@", 1)[1], 16)
                except ValueError:
                    raise ValueError(
                        f"Unrecognized mangled variable name in solution file: "
                        f"{name!r}"
                    ) from None
                if decl_index >= len(x_decl):
                    raise ValueError(
                        f"Declaration index {decl_index} from {name!r} is out of "
                        f"range for {len(x_decl)} x variables; the symbol list or "
                        f"num_periods ({num_periods}) is likely wrong."
                    )
                subscripts = x_decl[decl_index]
            else:
                name_parts = name.replace("$", "#").split("#")
                if len(name_parts) != 5:
                    raise ValueError(
                        f"Unrecognized x variable name in solution file: {name!r}"
                    )
                _, symbol, m_str, tau_str, t_str = name_parts
                if symbol not in sym_idx:
                    raise ValueError(
                        f"Unknown symbol {symbol!r} in solution file variable "
                        f"{name!r}."
                    )
                try:
                    m = int(m_str)
                    # ZIMPL replaces the invalid character '-' with '_', so a
                    # short (non-truncated) tau = -1 name carries "_1".
                    tau = int(tau_str.replace("_", "-"))
                    t = int(t_str)
                except ValueError:
                    raise ValueError(
                        f"Unrecognized x variable name in solution file: {name!r}"
                    ) from None
                if tau not in (1, -1) or not 1 <= m <= UB or not 0 <= t < num_periods:
                    raise ValueError(
                        f"Subscripts out of range in solution file variable {name!r}."
                    )
                subscripts = (sym_idx[symbol], m - 1, 0 if tau == 1 else 1, t)
            key = ("x", subscripts)
        elif name.startswith("y#"):
            key = ("y", _slack_subscripts(name, NUM_Y_SLACKS, num_periods))
        elif name.startswith("s2#"):
            key = ("s", _slack_subscripts(name, NUM_S_SLACKS, num_periods))
        else:
            raise ValueError(f"Unrecognized variable name in solution file: {name}")

        if key in values:
            raise ValueError(f"Duplicate variable {name!r} in solution file.")
        values[key] = value

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
