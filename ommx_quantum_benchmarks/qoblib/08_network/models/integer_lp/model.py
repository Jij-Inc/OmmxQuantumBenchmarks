import jijmodeling as jm


def build_ip_formulation() -> jm.Problem:
    """Create integer programming formulation for the arc-based flow problem.

    See the ZPL spec referenced in the original JijModeling 1 implementation
    for the full description.
    """
    problem = jm.Problem("ip_formulation", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        n = problem.Length("n")
        t = problem.Float("t", ndim=2)
        M = problem.Natural("M")
        intscale = problem.Natural("intscale")

        x = problem.BinaryVar("x", shape=(n, n), description="arc i->j selected")
        f = problem.IntegerVar(
            "f",
            shape=(n, n, n),
            lower_bound=0,
            upper_bound=intscale * M,
            description="flow of commodity k on arc i->j",
        )
        z = problem.IntegerVar("z", lower_bound=0, upper_bound=intscale * M)

        problem += z

        # c1: ∀ i ∈ N :  Σ_{j ≠ i} x[i,j] = 2
        problem += problem.Constraint(
            "c1_outdeg_eq_2",
            lambda i: jm.sum(x[i, j] for j in n if j != i) == 2,
            domain=n,
        )

        # c2: ∀ j ∈ N :  Σ_{i ≠ j} x[i,j] = 2
        problem += problem.Constraint(
            "c2_indeg_eq_2",
            lambda j: jm.sum(x[i, j] for i in n if i != j) == 2,
            domain=n,
        )

        # c11: flow balance
        problem += problem.Constraint(
            "c11_flow_balance",
            lambda k, i: jm.sum(f[k, j, i] for j in n if j != i) - jm.sum(f[k, i, j] for j in n if (j != i) & (j != k)) == t[k, i] * intscale,
            domain=jm.product(n, n).filter(lambda k, i: k != i),
        )

        # c14: capacity bound
        problem += problem.Constraint(
            "c14_capacity_by_x",
            lambda k, i, j: f[k, i, j] <= M * intscale * x[i, j],
            domain=jm.product(n, n, n).filter(lambda k, i, j: (i != j) & (k != j)),
        )

        # c100: z upper bound on flow
        problem += problem.Constraint(
            "c100_z_upper_bounds_flow",
            lambda i, j: jm.sum(f[k, i, j] for k in n if k != j) <= z,
            domain=jm.product(n, n).filter(lambda i, j: i != j),
        )

    return problem
