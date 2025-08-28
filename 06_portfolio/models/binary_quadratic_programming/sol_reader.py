import re


def parse_solution_ordered_dict_debug_v7(
    sol_path: str,
    symbol_to_i: dict[str, int],
    nSc: int,
    nTx: int,
    nCs1: int,
    nCs2: int,
    *,
    eps: float = 1e-6,  # tolerance for binarizing floating-point values
) -> tuple[dict[int, int], dict[int, str], float | None]:
    """Parse a Jij-style `.sol` file into ordered variable assignments with debug mapping and objective value.

    This parser supports scientific-notation floats, binarizes values with a tolerance, and preserves the
    v5 time-selection logic.

    Time selection precedence for `x$...` lines:
        1) If the token contains `@HEX` and there is an active `current_t_block` (set by the latest `s1` line),
           use `current_t_block`.
        2) Else if the token contains `@HEX` and there is a `last_t_global` seen earlier, use `last_t_global`.
        3) Else if the line provides an explicit `#t`, use that and update `last_t_global`.
        4) Else auto-increment a per-(i,m,sl) counter.

    Sign-to-side mapping:
        - '#_1' → sl = 0  (long)
        - '#1'  → sl = 1  (short)
        - '#0'  → sl = 0  (long)            # some generators emit 0/1
        - '#_'  → sl = 0  (long)            # underscore-only form (no '1')
        - In pattern `#(?P<sgn>...)@HEX`, `sgn` may be `'_'`, `'_1'`, `'1'`, or `'0'`.

    Variable ID layout (strict, contiguous):
        - x[i,m,sl,t]    : ids 0 .. (nS*nSc*2*nTx - 1)
        - s1[k1,t]       : next  nCs1*nTx ids
        - s2[k2,t]       : next  nCs2*nTx ids

    Args:
        sol_path: Path to the `.sol` file.
        symbol_to_i: Mapping from symbol string (e.g., stock ticker) to contiguous index `i` (0-based).
        nSc: Number of subclass choices `m` per symbol (file uses 1-based; IDs are 0-based).
        nTx: Number of time steps `t`.
        nCs1: Number of `s1` control bits per time step.
        nCs2: Number of `s2` control bits per time step.
        eps: Tolerance to binarize floating-point values: values within `eps` of 0 or 1 are snapped to 0/1.

    Returns:
        A tuple (sol_dict, dbg_dict, energy):
            - sol_dict: `{var_id: 0/1}` — all parsed variable assignments in the required strict order.
            - dbg_dict: `{line_no: debug_str}` — per-line debug messages for unmatched or unmapped lines.
            - energy: Parsed objective value from a line like `# Objective value = <float>`, or `None` if absent.

    Notes:
        - Line numbers in `dbg_dict` start from 0 (Python enumerate default).
        - Scientific-notation values are supported; values between 0 and 1 after snapping are rounded by `round()`.

    Example:
        >>> sol_dict, dbg, energy = parse_solution_ordered_dict_debug_v6(
        ...     "bqp_eval_a050_t10_q0.0005_b020.sol",
        ...     symbol_to_i={"AAPL": 0, "NVDA": 1},
        ...     nSc=3, nTx=10, nCs1=4, nCs2=20
        ... )
        >>> energy is None or isinstance(energy, float)
        True
    """
    nS, nSl = len(symbol_to_i), 2

    def xid(i: int, m: int, sl: int, t: int) -> int:
        return (((i * nSc) + m) * nSl + sl) * nTx + t

    def s1id(k1: int, t: int) -> int:
        return nS * nSc * nSl * nTx + k1 * nTx + t

    def s2id(k2: int, t: int) -> int:
        return nS * nSc * nSl * nTx + nCs1 * nTx + k2 * nTx + t

    # floating-point number pattern (supports scientific notation, optional sign)
    VAL = r"(?P<val>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)"

    rx_obj = re.compile(r"#\s*Objective value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    # Accept sgn in { '_1', '1', '_', '0' }
    SGN = r"(?P<sgn>(?:_?1|_|0))"

    # x with explicit #t (and optional @HEX)
    rx_x_full = re.compile(
        rf"^x\$(?P<sym>[A-Za-z0-9._-]+)#(?P<m>\d+)#{SGN}#(?P<t>\d+)(?P<ats>@[0-9a-fA-F]+)?\s+{VAL}$"
    )

    # x with optional #t and optional @HEX
    rx_x_flex = re.compile(
        rf"^x\$(?P<sym>[A-Za-z0-9._-]+)#(?P<m>\d+)#{SGN}(?:#(?P<t>\d+))?(?P<ats>@[0-9a-fA-F]+)?\s+{VAL}$"
    )

    # x with HEX time token (covers '#_@HEX', '#_1@HEX', '#1@HEX', '#0@HEX', and with an optional extra '#')
    rx_x_hex = re.compile(
        rf"^x\$(?P<sym>[A-Za-z0-9._-]+)#(?P<m>\d+)#{SGN}#?@(?P<hex>[0-9a-fA-F]+)\s+{VAL}$"
    )

    rx_s1 = re.compile(rf"^s1#(?P<k>\d+)#(?P<t>\d+)\s+{VAL}$")
    rx_s2 = re.compile(rf"^s2#(?P<k>\d+)#(?P<t>\d+)\s+{VAL}$")

    sol_dict: dict[int, int] = {}
    dbg_dict: dict[int, str] = {}

    t_counters: dict[tuple, int] = {}
    current_t_block = None
    last_t_global = None
    energy: float | None = None

    def decode_sl(sgn: str) -> int:
        # Map: '1' -> 1 (short), everything else ('_1','_','0') -> 0 (long)
        return 1 if sgn == "1" else 0

    def binarize(v_str: str) -> int:
        v = float(v_str)
        if abs(v) <= eps:
            return 0
        if abs(1.0 - v) <= eps:
            return 1
        return int(round(v))

    with open(sol_path, "r") as f:
        for line_no, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue

            # Objective value
            m_obj = rx_obj.match(line)
            if m_obj:
                try:
                    energy = float(m_obj.group(1))
                except ValueError:
                    energy = None
                continue

            if line.startswith("#"):
                continue

            # s1 variables
            m1 = rx_s1.match(line)
            if m1:
                k, t = int(m1["k"]), int(m1["t"])
                v = binarize(m1["val"])
                current_t_block = t
                last_t_global = t
                idx = s1id(k, t)
                sol_dict[idx] = v
                # dbg_dict[line_no] = f"{line} => id {idx} (s1Var k={k}, t={t}, v={v})"
                continue

            # s2 variables
            m2 = rx_s2.match(line)
            if m2:
                k, t = int(m2["k"]), int(m2["t"])
                v = binarize(m2["val"])
                idx = s2id(k, t)
                sol_dict[idx] = v
                # dbg_dict[line_no] = f"{line} => id {idx} (s2Var k={k}, t={t}, v={v})"
                continue

            # x with explicit #t / optional @HEX
            mxf = rx_x_full.match(line)
            if mxf:
                sym = mxf["sym"]
                if sym in symbol_to_i:
                    i = symbol_to_i[sym]
                    m = int(mxf["m"]) - 1
                    sl = decode_sl(mxf["sgn"])
                    has_at = mxf["ats"] is not None
                    t_exp = int(mxf["t"])
                    v = binarize(mxf["val"])

                    if has_at and current_t_block is not None:
                        t, reason = current_t_block, "t := current_t_block (has @HEX)"
                    elif has_at and last_t_global is not None:
                        t, reason = last_t_global, "t := last_t_global (has @HEX, no s1)"
                    else:
                        t, reason = t_exp, "t := explicit #t"
                        last_t_global = t

                    idx = xid(i, m, sl, t)
                    sol_dict[idx] = v
                    # dbg_dict[line_no] = f"{line} => id {idx} (xVar {sym}, m={m}, sl={sl}, t={t}, v={v}; {reason})"
                else:
                    dbg_dict[line_no] = f"{line} => <unmapped symbol>"
                continue

            # x with optional #t and optional @HEX
            mxf2 = rx_x_flex.match(line)
            if mxf2:
                sym = mxf2["sym"]
                if sym in symbol_to_i:
                    i = symbol_to_i[sym]
                    m = int(mxf2["m"]) - 1
                    sl = decode_sl(mxf2["sgn"])
                    has_at = mxf2["ats"] is not None
                    t_inline = mxf2["t"]
                    v = binarize(mxf2["val"])

                    if has_at and current_t_block is not None:
                        t, reason = current_t_block, "t := current_t_block (has @HEX)"
                    elif has_at and last_t_global is not None:
                        t, reason = last_t_global, "t := last_t_global (has @HEX, no s1)"
                    elif t_inline is not None:
                        t, reason = int(t_inline), "t := explicit #t"
                        last_t_global = t
                    else:
                        key = (i, m, sl)
                        t = t_counters.get(key, 0)
                        t_counters[key] = t + 1
                        reason = "t := auto-increment"

                    idx = xid(i, m, sl, t)
                    sol_dict[idx] = v
                    # dbg_dict[line_no] = f"{line} => id {idx} (xVar {sym}, m={m}, sl={sl}, t={t}, v={v}; {reason})"
                else:
                    dbg_dict[line_no] = f"{line} => <unmapped symbol>"
                continue

            # x with HEX time token only (no #t)
            mh = rx_x_hex.match(line)
            if mh:
                sym = mh["sym"]
                if sym in symbol_to_i:
                    i = symbol_to_i[sym]
                    m = int(mh["m"]) - 1
                    sl = decode_sl(mh["sgn"])
                    v = binarize(mh["val"])

                    if current_t_block is not None:
                        t, reason = current_t_block, "t := current_t_block (has @HEX)"
                    elif last_t_global is not None:
                        t, reason = last_t_global, "t := last_t_global (has @HEX, no s1)"
                    else:
                        key = (i, m, sl)
                        t = t_counters.get(key, 0)
                        t_counters[key] = t + 1
                        reason = "t := auto-increment (no s1 yet)"

                    idx = xid(i, m, sl, t)
                    sol_dict[idx] = v
                    # dbg_dict[line_no] = f"{line} => id {idx} (xVar {sym}, m={m}, sl={sl}, t={t}, v={v}; {reason})"
                else:
                    dbg_dict[line_no] = f"{line} => <unmapped symbol>"
                continue

            # unmatched line
            dbg_dict[line_no] = f"{line} => <no regex matched>"

    return sol_dict, dbg_dict, energy
