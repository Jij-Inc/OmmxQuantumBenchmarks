import jijmodeling as jm


def create_topology_model() -> jm.Problem:
    """Create Topology optimization model using Seidel linear formulation."""
    problem = jm.Problem("TopologyOptimization", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        n = problem.Length("nodes", description="Number of nodes in the graph")
        d = problem.Natural("degree", description="Maximum degree constraint")
        min_d = problem.Natural("minDiameter", description="Minimum diameter bound")
        max_d = problem.Length("maxDiameter", description="Maximum diameter bound")

        N_arr = problem.Natural("N_arr", ndim=1, description="Node index array")

        diameter = problem.IntegerVar(
            "diameter",
            lower_bound=min_d,
            upper_bound=max_d,
            description="Diameter of the graph",
        )

        dist = problem.BinaryVar(
            "dist",
            shape=(n, n, max_d),
            description="Distance variables: 1 if shortest path of length k between nodes s,t",
        )

        y = problem.BinaryVar(
            "y",
            shape=(n, n, n, max_d),
            description="Linearization variables for distance products",
        )

        problem += diameter

        # diameter_constraint
        problem += problem.Constraint(
            "diameter_constraint",
            lambda s, t: 1 + jm.sum(1 - dist[s, t, k] for k in max_d) <= diameter,
            domain=jm.product(n, n).filter(lambda s, t: s < t),
        )

        # DistCalc
        problem += problem.Constraint(
            "DistCalc",
            lambda s, t, k: dist[s, t, k + 1] <= dist[s, t, k] + jm.sum(y[s, t, i, k] for i in n if (i != s) & (i != t)),
            domain=jm.product(n, n, max_d).filter(lambda s, t, k: (s < t) & (k != max_d - 1)),
        )

        # DistLinearize_si
        problem += problem.Constraint(
            "DistLinearize_si",
            lambda s, t, i, k: y[s, t, i, k] <= dist[jm.min(s, i), jm.max(s, i), k],
            domain=jm.product(n, n, n, max_d).filter(lambda s, t, i, k: (s < t) & (i != s) & (i != t) & (k != max_d - 1)),
        )

        # DistLinearize_it
        problem += problem.Constraint(
            "DistLinearize_it",
            lambda s, t, i, k: y[s, t, i, k] <= dist[jm.min(i, t), jm.max(i, t), 0],
            domain=jm.product(n, n, n, max_d).filter(lambda s, t, i, k: (s < t) & (i != s) & (i != t) & (k != max_d - 1)),
        )

        # degreeButLast
        problem += problem.Constraint(
            "degreeButLast",
            lambda j_idx: jm.sum(dist[jm.min(N_arr[j_idx], i), jm.max(N_arr[j_idx], i), 0] for i in n if i != N_arr[j_idx]) == d,
            domain=jm.set(N_arr.len_at(0)).filter(lambda j_idx: N_arr[j_idx] != n - 1),
        )

        # degreeLast
        problem += problem.Constraint(
            "degreeLast",
            jm.sum(dist[i, n - 1, 0] for i in n if i != n - 1) == d - ((n * d) % 2),
        )

    return problem
