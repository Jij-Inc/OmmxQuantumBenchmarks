import jijmodeling as jm


def create_problem() -> jm.Problem:
    problem = jm.Problem(
        "Low Autocorrelation Binary Sequences (LABS)",
        sense=jm.ProblemSense.MINIMIZE,
    )

    @problem.update
    def _(problem: jm.DecoratedProblem):
        # K[k_idx] = k is provided as data so that arithmetic involving the
        # constraint-family lambda parameter `k_idx` (typed as ElementOf) can be
        # carried out via natural-typed values K[k_idx]. The same applies to I.
        I = problem.Natural("I", ndim=1)
        K = problem.Natural("K", ndim=1)
        N = problem.NamedExpr("N", I.len_at(0))

        x = problem.BinaryVar("x", shape=I.shape, description="Variable x")
        c = problem.IntegerVar(
            "c",
            shape=K.shape,
            lower_bound=-(N - 1),
            upper_bound=N - 1,
            description="Variable c",
        )

        problem += jm.sum(c[k_idx] * c[k_idx] for k_idx in K.len_at(0))

        problem += problem.Constraint(
            "c1",
            lambda k_idx: c[k_idx]
            == jm.sum(
                (2 * x[i] - 1) * (2 * x[i + K[k_idx] + 1] - 1)
                for i in N
                if i + K[k_idx] + 1 < N
            ),
            domain=K.len_at(0),
        )

    return problem
