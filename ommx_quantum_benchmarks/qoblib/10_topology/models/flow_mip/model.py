import jijmodeling as jm


def create_topology_model() -> jm.Problem:
    """Create Topology optimization model (flow MIP variant)."""
    problem = jm.Problem("TopologyOptimization", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        n = problem.Length("nodes", description="Number of nodes in the graph")
        d = problem.Natural("degree", description="Maximum degree constraint")

        diameter = problem.IntegerVar(
            "diameter",
            lower_bound=0,
            upper_bound=n - 1,
            description="Diameter of the graph",
        )
        shortest_path = problem.IntegerVar(
            "SP",
            lower_bound=0,
            upper_bound=n - 1,
            shape=(n, n),
            description="Shortest path length between node pairs",
        )
        z = problem.BinaryVar("z", shape=(n, n), description="Edge existence between nodes")
        x = problem.BinaryVar(
            "x",
            shape=(n, n, n, n),
            description="Flow variables for shortest paths",
        )

        problem += diameter

        # C1: SP[s,t] <= diameter, s < t
        problem += problem.Constraint(
            "diameter",
            lambda s, t: shortest_path[s, t] <= diameter,
            domain=jm.product(n, n).filter(lambda s, t: s < t),
        )

        # C2: SP[s,t] == sum_{i != j} x[s,t,i,j]
        problem += problem.Constraint(
            "APSP",
            lambda s, t: shortest_path[s, t] == jm.sum(x[s, t, i, j] for i in n for j in n if i != j),
            domain=jm.product(n, n).filter(lambda s, t: s < t),
        )

        # C3: flow conservation at intermediate nodes
        problem += problem.Constraint(
            "SPtransit",
            lambda s, t, i: jm.sum(x[s, t, i, j] for j in n if i != j) - jm.sum(x[s, t, j, i] for j in n if j != i) == 0,
            domain=jm.product(n, n, n).filter(lambda s, t, i: (s < t) & (i != s) & (i != t)),
        )

        # C4: flow out from source node = 1
        problem += problem.Constraint(
            "SPsource",
            lambda s, t: jm.sum(x[s, t, s, j] for j in n if s != j) - jm.sum(x[s, t, j, s] for j in n if j != s) == 1,
            domain=jm.product(n, n).filter(lambda s, t: s < t),
        )

        # C5: flow into target node = -1
        problem += problem.Constraint(
            "SPtarget",
            lambda s, t: jm.sum(x[s, t, t, j] for j in n if t != j) - jm.sum(x[s, t, j, t] for j in n if j != t) == -1,
            domain=jm.product(n, n).filter(lambda s, t: s < t),
        )

        # C6: degree constraint
        problem += problem.Constraint(
            "degree_constraint",
            lambda i: jm.sum(z[i, j] for j in n if i < j) + jm.sum(z[j, i] for j in n if j < i) <= d,
            domain=n,
        )

        # C7: ZXlink forward / backward, i < j
        problem += problem.Constraint(
            "ZXlink_forward",
            lambda s, t, i, j: z[i, j] >= x[s, t, i, j],
            domain=jm.product(n, n, n, n).filter(lambda s, t, i, j: (s < t) & (i < j)),
        )
        problem += problem.Constraint(
            "ZXlink_backward",
            lambda s, t, i, j: z[i, j] >= x[s, t, j, i],
            domain=jm.product(n, n, n, n).filter(lambda s, t, i, j: (s < t) & (i < j)),
        )

    return problem
