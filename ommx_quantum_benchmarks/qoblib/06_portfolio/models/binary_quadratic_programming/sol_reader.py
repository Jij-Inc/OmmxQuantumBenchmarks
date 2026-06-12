import re

from constants import NUM_S_SLACKS, NUM_SIGNS, NUM_UNITS, NUM_Y_SLACKS, TAU

_OBJECTIVE_RE = re.compile(
    r"#\s*Objective value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE
)
BINARY_FEASIBILITY_TOLERANCE = 1e-5


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


def _zimpl_x_name(symbols: list[str], subscripts: tuple[int, int, int, int]) -> str:
    """Reconstruct the unmangled ZIMPL LP name for an x variable."""
    i, m, l, t = subscripts
    tau_name = str(TAU[l]).replace("-", "_")
    return f"x${symbols[i]}#{m + 1}#{tau_name}#{t}"


def parse_sol_file(
    lines, symbols: list[str], num_periods: int
) -> tuple[float, dict[tuple[str, tuple[int, ...]], int]]:
    """Parse a QOBLIB portfolio solution file in Gurobi .sol format.

    The solution files under `solutions/bqp` of the original QOBLIB repository
    contain one variable per line (`<name> <value>`) plus an
    `# Objective value = ...` header.  Variable names come from ZIMPL's LP
    output:

    - `y#<k>#<t>` and `s2#<c>#<t>` for the slack variables,
    - `x$<symbol>#<m>#<tau>#<t>` for the asset variables, where ZIMPL replaces
      the invalid character '-' with '_', so tau = -1 appears as `_1`,
    - mangled names like `x$AAPL#1#_1#0@a` for long (truncated) names.  When
      LP variable names exceed ZIMPL's output-name limit, ZIMPL keeps a prefix
      and appends `@<hex declaration index>`.  The declaration order is
      `var x[SX*TX]` with SX = S * {1..ub} * {1,-1}, followed by day, so it is
      asset -> unit -> sign -> day (day fastest).  We reproduce that order in
      `x_decl`, recover the subscripts from the hex index, then verify that the
      full reconstructed ZIMPL name starts with the retained prefix.

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
        - dict mapping (variable name, subscripts) to the integer variable
          value, with subscripts (i, m, l, t) for "x" and (k, t) for
          "y" / "s2".
    """
    sym_idx = {s: i for i, s in enumerate(symbols)}
    # Declaration order of x[SX*TX]: asset, unit, sign, day (day fastest).
    x_decl = [
        (i, m, l, t)
        for i in range(len(symbols))
        for m in range(NUM_UNITS)
        for l in range(NUM_SIGNS)
        for t in range(num_periods)
    ]

    objective = None
    values: dict[tuple[str, tuple[int, ...]], int] = {}
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
        # All model variables are binary. Use Gurobi's default IntFeasTol so
        # valid .sol output is not rejected before ommx_create.py performs the
        # objective and feasibility checks that gate artifact writing.
        rounded = round(value)
        if (
            abs(value - rounded) > BINARY_FEASIBILITY_TOLERANCE
            or rounded not in (0, 1)
        ):
            raise ValueError(
                f"Non-binary value {value} for variable {name!r} in solution file."
            )
        value = int(rounded)

        if name.startswith("x$"):
            if "@" in name:
                # Mangled (truncated) name: decode via the declaration index.
                # The retained prefix is checked below against the reconstructed
                # full name, so a stale symbol list or wrong declaration-order
                # assumption is caught instead of silently remapping variables.
                try:
                    decl_index = int(name.rsplit("@", 1)[1], 16)
                except ValueError:
                    raise ValueError(
                        f"Unrecognized mangled variable name in solution file: "
                        f"{name!r}"
                    ) from None
                if not 0 <= decl_index < len(x_decl):
                    raise ValueError(
                        f"Declaration index {decl_index} from {name!r} is out of "
                        f"range for {len(x_decl)} x variables; the symbol list or "
                        f"num_periods ({num_periods}) is likely wrong."
                    )
                subscripts = x_decl[decl_index]
                prefix = name.rsplit("@", 1)[0]
                full_name = _zimpl_x_name(symbols, subscripts)
                if not full_name.startswith(prefix):
                    raise ValueError(
                        f"Mangled variable name {name!r} decodes to {full_name!r}, "
                        f"which does not match the retained prefix {prefix!r}."
                    )
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
                if (
                    tau not in TAU
                    or not 1 <= m <= NUM_UNITS
                    or not 0 <= t < num_periods
                ):
                    raise ValueError(
                        f"Subscripts out of range in solution file variable {name!r}."
                    )
                subscripts = (sym_idx[symbol], m - 1, TAU.index(tau), t)
            key = ("x", subscripts)
        elif name.startswith("y#"):
            key = ("y", _slack_subscripts(name, NUM_Y_SLACKS, num_periods))
        elif name.startswith("s2#"):
            key = ("s2", _slack_subscripts(name, NUM_S_SLACKS, num_periods))
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
        values.setdefault(("x", subscripts), 0)
    for k in range(NUM_Y_SLACKS):
        for t in range(num_periods):
            values.setdefault(("y", (k, t)), 0)
    for c in range(NUM_S_SLACKS):
        for t in range(num_periods):
            values.setdefault(("s2", (c, t)), 0)
    return objective, values
