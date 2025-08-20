import re

def parse_solution_zfx(file_path: str, n: int) -> dict[int, float]:
    """Parse Jij solution (.sol) into {id: value} with strict z→f→x ordering.

    Order:
      - id=0 : z
      - id=1.. : f[k,i,j] (k=0..n-1, i=0..n-1, j=0..n-1)
      - then: x[i,j] (i=0..n-1, j=0..n-1)

    Args:
        file_path (str): Path to solution file
        n (int): Number of nodes

    Returns:
        dict[int,float]: {id: value} in z→f→x order
    """
    z_val = None
    f_vals = {}
    x_vals = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # z
            m = re.match(r"^z\s+([-+]?\d+\.?\d*)$", line)
            if m:
                z_val = float(m.group(1))
                continue

            # f#k#i#j
            m = re.match(r"^f#(\d+)#(\d+)#(\d+)\s+([-+]?\d+\.?\d*)$", line)
            if m:
                k, i, j, val = (
                    int(m.group(1)) - 1,
                    int(m.group(2)) - 1,
                    int(m.group(3)) - 1,
                    float(m.group(4)),
                )
                f_vals[(k, i, j)] = val
                continue

            # x#i#j
            m = re.match(r"^x#(\d+)#(\d+)\s+([-+]?\d+\.?\d*)$", line)
            if m:
                i, j, val = int(m.group(1)) - 1, int(m.group(2)) - 1, float(m.group(3))
                x_vals[(i, j)] = val
                continue

    if z_val is None:
        raise ValueError("No z found in solution file.")

    # ---- 重建 {id: value} ----
    sol_dict = {}
    idx = 0

    # z
    sol_dict[idx] = z_val
    idx += 1

    # f[k,i,j]，字典序
    for k in range(n):
        for i in range(n):
            for j in range(n):
                sol_dict[idx] = f_vals.get((k, i, j), 0.0)
                idx += 1

    # x[i,j]，字典序
    for i in range(n):
        for j in range(n):
            sol_dict[idx] = x_vals.get((i, j), 0.0)
            idx += 1

    return sol_dict