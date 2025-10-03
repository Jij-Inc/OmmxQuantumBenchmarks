import re


def parse_x_hash_kv(
    file_path: str,
    *,
    binarize_eps: float = 1e-9,
) -> dict[int, float]:
    """Parse lines 'x#<id> <value>' into {id: value}.

    - Ignore blank lines and comment lines (those starting with '#').
    - Support scientific notation.
    - If the same id appears multiple times, keep the last occurrence.
    - Set binarize_eps=None to disable binarization of values near 0/1.

    Args:
        file_path: For example, 'uqo_a200_t10_q0_b050.sol.mst'.
        binarize_eps: Tolerance for binarizing values close to 0 or 1.

    Returns:
        dict[int, float]
    """
    pat = re.compile(
        r"^\s*x#(\d+)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    result: dict[int, float] = {}
    parsed_any = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = pat.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            val = float(m.group(2))
            if binarize_eps is not None:
                if abs(val - 0.0) <= binarize_eps:
                    val = 0.0
                elif abs(val - 1.0) <= binarize_eps:
                    val = 1.0
            result[idx - 1] = val
            parsed_any = True

    if not parsed_any:
        raise ValueError(f"No 'x#<id> <value>' lines found in: {file_path}")
    return result
