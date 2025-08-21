from __future__ import annotations
from math import sqrt


def read_vrp_tsplib(
    path: str,
    vehicle_limit: int,
    *,
    euc2d_round: bool = True,  # TSPLIB EUC_2D convention: round distances to nearest integer
    depot_policy: str = "first",  # Handling of multiple depots (only "first" supported)
) -> dict:
    """Read a TSPLIB-style CVRP instance and return JijModeling instance data.

    Supports the standard TSPLIB format for the Capacitated Vehicle Routing
    Problem (CVRP), including parsing coordinates, demands, and depot
    definitions. Distances are computed with the EUC_2D convention.

    Args:
        path (str): Path to the TSPLIB-format CVRP file.
        vehicle_limit (int): Maximum number of vehicles allowed.
        euc2d_round (bool, optional): Whether to round EUC_2D distances to
            the nearest integer (default True, per TSPLIB convention).
        depot_policy (str, optional): Policy for selecting the depot if
            multiple depots are provided. Currently only "first" is supported.

    Returns:
        Dict: A dictionary formatted for JijModeling instance_data with keys:
            - "n" (int): Number of nodes (dimension).
            - "VEHICLE_LIMIT" (int): Vehicle limit.
            - "CAPACITY" (int): Vehicle capacity.
            - "DEMAND" (List[int]): Demand at each node (0-based).
            - "D" (List[List[float|int]]): Distance matrix (symmetric).
            - "DEPOT" (int): Depot index (0-based).
    """
    # --- 1) Read file lines ---
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f]

    lines = [ln for ln in raw_lines if ln != ""]

    # --- 2) Parse header ---
    header: dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        ln = lines[idx]
        if ln.upper().endswith("SECTION") or ln.upper() == "EOF":
            break
        if ":" in ln:
            key, val = ln.split(":", 1)
            header[key.strip().upper()] = val.strip()
        idx += 1

    try:
        dim = int(header["DIMENSION"])
        cap = int(header["CAPACITY"])
        edge_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
        if edge_type not in ("EUC_2D",):
            raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}")
    except KeyError as e:
        raise ValueError(f"Missing required header field: {e}")

    # --- 3) Section indices ---
    def find_section(name: str) -> int:
        nameU = name.upper()
        for k in range(idx, len(lines)):
            if lines[k].upper().startswith(nameU):
                return k
        return -1

    sec_coord = find_section("NODE_COORD_SECTION")
    sec_demand = find_section("DEMAND_SECTION")
    sec_depot = find_section("DEPOT_SECTION")
    sec_eof = find_section("EOF")
    if sec_coord < 0 or sec_demand < 0 or sec_depot < 0:
        raise ValueError(
            "Missing one of required sections: NODE_COORD_SECTION / DEMAND_SECTION / DEPOT_SECTION"
        )
    if sec_eof < 0:
        sec_eof = len(lines)

    # --- 4) Node coordinates ---
    coords: list[tuple[float, float]] = []
    k = sec_coord + 1
    while k < len(lines) and k < sec_demand:
        ln = lines[k]
        if ln.upper().endswith("SECTION"):
            break
        parts = ln.split()
        if len(parts) >= 3:
            x = float(parts[1])
            y = float(parts[2])
            coords.append((x, y))
        k += 1
    if len(coords) != dim:
        raise ValueError(f"NODE_COORD_SECTION count {len(coords)} != DIMENSION {dim}")

    # --- 5) Demands ---
    demand: list[int] = [0] * dim
    k = sec_demand + 1
    while k < len(lines) and k < sec_depot:
        ln = lines[k]
        if ln.upper().endswith("SECTION"):
            break
        parts = ln.split()
        if len(parts) >= 2:
            idx1 = int(parts[0])
            dem = int(parts[1])
            demand[idx1 - 1] = dem
        k += 1

    # --- 6) Depot section ---
    depots_1b: list[int] = []
    k = sec_depot + 1
    while k < len(lines) and k < sec_eof:
        ln = lines[k]
        if ln.upper().endswith("SECTION"):
            break
        v = ln.split()[0]
        if v == "-1":
            break
        depots_1b.append(int(v))
        k += 1
    if not depots_1b:
        raise ValueError("DEPOT_SECTION is empty")
    if depot_policy != "first":
        raise ValueError(f"Unsupported depot_policy={depot_policy}")
    depot0 = depots_1b[0] - 1

    # --- 7) Distance matrix ---
    def euc2d(a: tuple[float, float], b: tuple[float, float]) -> float:
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    D: list[list[float]] = [[0.0] * dim for _ in range(dim)]
    for u in range(dim):
        for v in range(dim):
            if u == v:
                D[u][v] = 0.0
            else:
                d = euc2d(coords[u], coords[v])
                D[u][v] = int(round(d)) if euc2d_round else d

    # --- 8) Return instance_data ---
    instance_data = {
        "n": dim,
        "VEHICLE_LIMIT": int(vehicle_limit),
        "CAPACITY": cap,
        "DEMAND": demand,
        "D": D,
        "DEPOT": depot0,
    }
    return instance_data
