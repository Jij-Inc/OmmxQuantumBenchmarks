import os
import re


def parse_sol_file(file_path: str, n: int) -> tuple[dict[str, float], dict[int, float]]:
    """Parse MIS solution file with automatic format detection.

    Supports two common formats found in `.sol` files:

    **A) Key–Value format (e.g., `opt.sol`, `sol`)**
        Lines may include:
            - `# Objective value = <float>`
            - `x#<idx> <float>` (typically 1-based indexing)

    **B) Index-list format (e.g., `bst.sol`, sometimes `opt.sol`)**
        Each line contains a single integer representing a selected vertex.
        Values are interpreted as 0-based or 1-based automatically.

    Args:
        file_path (str): Path to the solution file.
        n (int): Number of vertices in the MIS instance.

    Returns:
        tuple:
            - dict[str, float]: `{"Energy": <float or None>}`.
              The objective value if present; otherwise, the count of selected vertices.
            - dict[int, float]: Mapping of vertex indices (0..n−1) to 0.0/1.0 (or float in KV format).
    """
    # read full file
    with open(file_path, "r") as f:
        raw_lines = [ln.strip() for ln in f]

    lines = [ln for ln in raw_lines if ln]

    # detect KV format
    kv_pattern = re.compile(r"^x#(\d+)\s+([-+]?\d*\.?\d+)$", re.IGNORECASE)
    has_kv = any(kv_pattern.match(ln) for ln in lines)

    # check objective value
    obj_pattern = re.compile(
        r"#\s*Objective\s*value\s*=\s*([-+]?\d*\.?\d+)", re.IGNORECASE
    )
    obj_value = None
    for ln in lines:
        m = obj_pattern.match(ln)
        if m:
            obj_value = float(m.group(1))
            break

    if has_kv:
        # === KV format ===
        x_vars = {}
        for ln in lines:
            m = kv_pattern.match(ln)
            if m:
                idx = int(m.group(1))  # assume 1-based: x#1, x#2, ...
                val = float(m.group(2))
                x_vars[idx] = val

        solution_dict = {i - 1: x_vars.get(i, 0.0) for i in range(1, n + 1)}
        return {"Energy": obj_value}, solution_dict

    # === Index-list format ===
    selected = []
    only_ints = True
    for ln in lines:
        try:
            val = int(ln)
            selected.append(val)
        except ValueError:
            only_ints = False
            break

    if only_ints and selected:
        all_in_1_based = all(1 <= v <= n for v in selected)
        all_in_0_based = all(0 <= v <= n - 1 for v in selected)

        if not (all_in_0_based or all_in_1_based):
            raise ValueError(
                f"Index list not within [0, {n-1}] or [1, {n}] in {file_path}: {selected[:10]}..."
            )

        use_one_based = all_in_1_based
        solution_dict = {i: 0.0 for i in range(n)}
        if use_one_based:
            for v in selected:
                solution_dict[v - 1] = 1.0
        else:
            for v in selected:
                solution_dict[v] = 1.0

        if obj_value is None:
            obj_value = float(sum(solution_dict.values()))

        return {"Energy": obj_value}, solution_dict

    raise ValueError(
        f"Unrecognized solution format in {file_path}. "
        "Expect either 'x#i val' lines or pure index-per-line."
    )
