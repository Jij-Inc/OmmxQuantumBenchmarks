import jijmodeling as jm


def build_vrp_ilp() -> jm.Problem:
    """Create Vehicle Routing Problem (VRP) ILP model.

    Formulates the capacitated vehicle routing problem with a single depot
    using a standard integer linear programming approach. The objective is
    to minimize total travel distance subject to vehicle limit and capacity
    constraints.

    Parameters (placeholders):
        - n (int): Number of nodes (including depot).
        - VEHICLE_LIMIT (int): Maximum number of vehicles allowed.
        - CAPACITY (int): Capacity of each vehicle.
        - DEMAND (ndarray, shape (n,)): Demand at each node.
        - D (ndarray, shape (n, n)): Distance matrix.
        - DEPOT (int): Index of the depot node (0-based).

    Variables:
        - x[i, j] ∈ {0,1}: 1 if arc (i → j) is used, 0 otherwise.
        - y[i] ∈ ℤ, [0, CAPACITY]: Load upon arrival at node i.

    Objective:
        - Minimize total distance: Σ_{i, j} D[i, j] · x[i, j].

    Constraints:
        - Each customer visited exactly once (excl. depot).
        - Flow conservation for all non-depot nodes.
        - Departures from depot ≤ VEHICLE_LIMIT.
        - Capacity propagation (MTZ-style) to prevent subtours.
        - Capacity bounds at all customer nodes.

    Returns:
        jm.Problem: JijModeling problem instance with variables, objective,
        and constraints for the capacitated VRP.
    """
    # ------------ Placeholders ------------
    n = jm.Placeholder("n")
    VEHICLE_LIMIT = jm.Placeholder("VEHICLE_LIMIT")
    CAPACITY = jm.Placeholder("CAPACITY")
    DEMAND = jm.Placeholder("DEMAND", ndim=1)  # shape (n,)
    D = jm.Placeholder("D", ndim=2)  # shape (n,n)
    DEPOT = jm.Placeholder("DEPOT")  # single depot index (0-based)

    # ------------ Indices ------------
    i = jm.Element("i", belong_to=(0, n))
    j = jm.Element("j", belong_to=(0, n))
    h = jm.Element("h", belong_to=(0, n))

    # ------------ Variables ------------
    x = jm.BinaryVar("x", shape=(n, n), description="arc i->j used")
    y = jm.IntegerVar(
        "y",
        shape=(n,),
        lower_bound=0,
        upper_bound=CAPACITY,
        description="load upon arrival at node",
    )

    # ------------ Problem & Objective ------------
    problem = jm.Problem("vrp_ilp", sense=jm.ProblemSense.MINIMIZE)
    problem += jm.sum([i, j], D[i, j] * x[i, j])  # (10) minimize total distance

    # ------------ Constraints ------------
    # (11) Each customer visited exactly once (excluding depot)
    problem += jm.Constraint(
        "customer_visited_once",
        jm.sum([(j, j != i)], x[i, j]) == 1,
        forall=[(i, i != DEPOT)],
    )

    # (12) Flow conservation for non-depot nodes
    problem += jm.Constraint(
        "flow_conservation",
        jm.sum([(i, i != h)], x[i, h]) - jm.sum([(i, i != h)], x[h, i]) == 0,
        forall=[(h, h != DEPOT)],
    )

    # (13) Vehicle limit: departures from depot ≤ VEHICLE_LIMIT
    problem += jm.Constraint(
        "vehicle_limit", jm.sum([(j, j != DEPOT)], x[DEPOT, j]) <= VEHICLE_LIMIT
    )

    # (14) Capacity propagation (MTZ-style), exclude depot and i=j
    problem += jm.Constraint(
        "capacity_limit",
        y[j] >= y[i] + DEMAND[j] * x[i, j] - CAPACITY * (1 - x[i, j]),
        forall=[i, (j, (j != DEPOT) & (j != i))],
    )

    # (15) Capacity bounds at nodes
    problem += jm.Constraint("capacity_limit_node_ub", y[i] <= CAPACITY, forall=[i])
    problem += jm.Constraint("capacity_limit_node_lb", DEMAND[i] <= y[i], forall=[i])

    return problem
