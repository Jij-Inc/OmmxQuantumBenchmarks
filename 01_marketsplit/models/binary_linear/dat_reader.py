import numpy as np

def read_qoblib_dat_file(filepath: str, *, dtype: type = np.int64) -> dict[str, np.ndarray]:
    """Read and parse a QOBLIB-format .dat file into arrays for JijModeling.

    The file format is:
      - First non-comment line: "m n"  (m constraints, n variables)
      - Next m lines: each has n coefficients followed by 1 RHS constant (total n+1 numbers per line)
      - Lines starting with '#' (after optional leading spaces) are comments and ignored.

    Args:
        filepath: Path to the .dat file.
        dtype: NumPy dtype for numbers; default np.int64. Use np.float64 if your
               instance contains non-integers.

    Returns:
        dict with:
          - "I": np.arange(m)        (shape: (m,))
          - "J": np.arange(n)        (shape: (n,))
          - "a": coefficient matrix  (shape: (m, n))
          - "b": RHS vector          (shape: (m,))

    Raises:
        ValueError: If the file is empty, first line is invalid, or data length mismatches.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Keep non-empty, non-comment lines (allow leading spaces before '#')
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if raw.lstrip().startswith("#"):
            continue
        lines.append(line)

    if not lines:
        raise ValueError("No data lines found in the file.")

    # First line: m n
    first = lines[0].split()
    if len(first) < 2:
        raise ValueError(f"Invalid header line (need 'm n'): {lines[0]!r}")
    try:
        m = int(first[0])
        n = int(first[1])
    except Exception as e:
        raise ValueError(f"Failed to parse 'm n' from header: {lines[0]!r}") from e

    # Flatten all remaining numbers
    tokens: list[str] = []
    for ln in lines[1:]:
        tokens.extend(ln.split())

    expected = m * (n + 1)
    if len(tokens) != expected:
        raise ValueError(
            f"Data length mismatch: expected {expected} numbers for m={m}, n={n}, "
            f"but got {len(tokens)}."
        )

    # Vectorized reshape → (m, n+1)
    try:
        arr = np.asarray(tokens, dtype=dtype).reshape(m, n + 1)
    except Exception as e:
        raise ValueError("Failed to convert/reshape data numbers into matrix form.") from e

    a = arr[:, :n]
    b = arr[:, -1]

    return {
        "I": np.arange(m, dtype=np.int64),
        "J": np.arange(n, dtype=np.int64),
        "a": a,
        "b": b,
    }
