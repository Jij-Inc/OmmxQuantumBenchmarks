import jijmodeling as jm


def create_topology_model() -> jm.Problem:
    """Create Topology optimization model using Seidel quadratic formulation."""
    problem = jm.Problem("TopologyOptimization", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        n = problem.Length("nodes", description="Number of nodes in the graph")
        d = problem.Natural("degree", description="Maximum degree constraint")
        min_d = problem.Natural("minDiameter", description="Minimum diameter bound")
        max_d = problem.Length("maxDiameter", description="Maximum diameter bound")

        # N_arr[j_idx] = j_idx, used to obtain a natural-typed node index
        # when iterating with a single-arg lambda over a filtered set. Without
        # this, JijModeling 2's type system rejects `jm.min(j, i)` because
        # the lambda parameter is typed as ElementOf rather than natural.
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

        problem += diameter

        # C1 diameter: 1 + sum_k (1 - dist[s,t,k]) <= diameter, s < t
        problem += problem.Constraint(
            "diameter",
            lambda s, t: 1 + jm.sum(1 - dist[s, t, k] for k in max_d) <= diameter,
            domain=jm.product(n, n).filter(lambda s, t: s < t),
        )

        # C2 DistCalc: dist[s,t,k+1] <= dist[s,t,k] + sum_i dist[min(s,i),max(s,i),k]
        #                                              * dist[min(i,t),max(i,t),0]
        problem += problem.Constraint(
            "DistCalc",
            lambda s, t, k: dist[s, t, k + 1]
            <= dist[s, t, k]
            + jm.sum(
                dist[jm.min(s, i), jm.max(s, i), k]
                * dist[jm.min(i, t), jm.max(i, t), 0]
                for i in n
                if (i != s) & (i != t)
            ),
            domain=jm.product(n, n, max_d).filter(
                lambda s, t, k: (s < t) & (k != max_d - 1)
            ),
        )

        # C3 degreeButLast: sum_{i != j} dist[min(j,i),max(j,i),0] == d, j != n - 1
        problem += problem.Constraint(
            "degreeButLast",
            lambda j_idx: jm.sum(
                dist[jm.min(N_arr[j_idx], i), jm.max(N_arr[j_idx], i), 0]
                for i in n
                if i != N_arr[j_idx]
            )
            == d,
            domain=jm.set(N_arr.len_at(0)).filter(
                lambda j_idx: N_arr[j_idx] != n - 1
            ),
        )

        # C4 degreeLast: sum_{i != n - 1} dist[i, n - 1, 0] == d - (n*d % 2)
        problem += problem.Constraint(
            "degreeLast",
            jm.sum(dist[i, n - 1, 0] for i in n if i != n - 1) == d - ((n * d) % 2),
        )

    return problem
