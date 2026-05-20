import jijmodeling as jm


def build_vrp_ilp() -> jm.Problem:
    """Create Vehicle Routing Problem (VRP) ILP model.

    Capacitated VRP with a single depot. See the docstring of the original
    JijModeling 1 implementation for the full specification.
    """
    problem = jm.Problem("vrp_ilp", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        n = problem.Length("n")
        VEHICLE_LIMIT = problem.Natural("VEHICLE_LIMIT")
        CAPACITY = problem.Natural("CAPACITY")
        DEMAND = problem.Float("DEMAND", ndim=1)
        D = problem.Float("D", ndim=2)
        DEPOT = problem.Natural("DEPOT")

        x = problem.BinaryVar("x", shape=(n, n), description="arc i->j used")
        y = problem.IntegerVar(
            "y",
            shape=(n,),
            lower_bound=0,
            upper_bound=CAPACITY,
            description="load upon arrival at node",
        )

        # (10) objective: minimize total distance
        problem += jm.sum(D[i, j] * x[i, j] for i in n for j in n)

        # (11) Each customer visited exactly once (excluding depot)
        problem += problem.Constraint(
            "customer_visited_once",
            lambda i: jm.sum(x[i, j] for j in n if j != i) == 1,
            domain=jm.set(n).filter(lambda i: i != DEPOT),
        )

        # (12) Flow conservation for non-depot nodes
        problem += problem.Constraint(
            "flow_conservation",
            lambda h: jm.sum(x[i, h] for i in n if i != h) - jm.sum(x[h, i] for i in n if i != h) == 0,
            domain=jm.set(n).filter(lambda h: h != DEPOT),
        )

        # (13) Vehicle limit: departures from depot ≤ VEHICLE_LIMIT
        problem += problem.Constraint(
            "vehicle_limit",
            jm.sum(x[DEPOT, j] for j in n if j != DEPOT) <= VEHICLE_LIMIT,
        )

        # (14) Capacity propagation (MTZ-style), exclude depot and i=j
        problem += problem.Constraint(
            "capacity_limit",
            lambda i, j: y[j] >= y[i] + DEMAND[j] * x[i, j] - CAPACITY * (1 - x[i, j]),
            domain=jm.product(n, n).filter(lambda i, j: (j != DEPOT) & (j != i)),
        )

        # (15) Capacity bounds at nodes
        problem += problem.Constraint(
            "capacity_limit_node_ub",
            lambda i: y[i] <= CAPACITY,
            domain=n,
        )
        problem += problem.Constraint(
            "capacity_limit_node_lb",
            lambda i: DEMAND[i] <= y[i],
            domain=n,
        )

    return problem
