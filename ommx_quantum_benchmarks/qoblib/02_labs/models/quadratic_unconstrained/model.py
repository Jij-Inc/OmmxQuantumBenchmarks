import jijmodeling as jm


def create_problem():
    problem = jm.Problem("Energy_Minimization", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        I = problem.Natural("I", ndim=1)
        K = problem.Natural("K", ndim=1)
        N = problem.NamedExpr("N", I.len_at(0))
        P = problem.Float("P")

        x = problem.BinaryVar("x", shape=I.shape, description="Binary variable x")
        z = problem.BinaryVar(
            "z",
            shape=(I.len_at(0), K.len_at(0)),
            description="Binary variable z",
        )

        # K[k_idx] gives a natural-typed lag value usable in arithmetic.
        first_term = jm.sum((jm.sum(4 * z[i, k_idx] - 2 * x[i] - 2 * x[i + K[k_idx] + 1] + 1 for i in N if i + K[k_idx] + 1 < N)) ** 2 for k_idx in K.len_at(0))

        second_term = P * jm.sum(jm.sum(3 * z[i, k_idx] - 2 * z[i, k_idx] * x[i] - 2 * z[i, k_idx] * x[i + K[k_idx] + 1] + x[i] * x[i + K[k_idx] + 1] for i in N if i + K[k_idx] + 1 < N) for k_idx in K.len_at(0))

        problem += first_term + second_term

    return problem
