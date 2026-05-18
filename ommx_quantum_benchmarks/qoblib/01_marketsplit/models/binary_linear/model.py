import jijmodeling as jm


def create_problem() -> jm.Problem:
    """
    Create the JijModeling problem definition.
    """
    problem = jm.Problem("SetCovering", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        I = problem.Natural("I", ndim=1, description="Set I")
        J = problem.Natural("J", ndim=1, description="Set J")
        a = problem.Integer("a", ndim=2, description="Parameter a")
        b = problem.Integer("b", ndim=1, description="Parameter b")

        x = problem.BinaryVar("x", shape=J.shape, description="Variable x")
        s = problem.IntegerVar(
            "s",
            shape=I.shape,
            lower_bound=0,
            upper_bound=100000,
            description="Variable s",
        )

        problem += jm.sum(s[i] for i in I.len_at(0))

        problem += problem.Constraint(
            "c1",
            lambda i: s[i] + jm.sum(a[i, j] * x[j] for j in J.len_at(0)) == b[i],
            domain=I.len_at(0),
        )

    return problem
