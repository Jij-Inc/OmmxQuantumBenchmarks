import jijmodeling as jm


def create_problem() -> jm.Problem:
    """
    Create the JijModeling problem definition.
    """
    problem = jm.Problem("Marketsplit_unconstrained", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        I = problem.Natural("I", ndim=1, description="Set I")
        J = problem.Natural("J", ndim=1, description="Set J")
        a = problem.Integer("a", ndim=2, description="Parameter a")
        b = problem.Integer("b", ndim=1, description="Parameter b")

        x = problem.BinaryVar("x", shape=J.shape, description="Variable x")

        problem += jm.sum((b[i] - jm.sum(a[i, j] * x[j] for j in J.len_at(0))) ** 2 for i in I.len_at(0))

    return problem
