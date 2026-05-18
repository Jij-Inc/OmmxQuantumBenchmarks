import jijmodeling as jm


def build_mis_problem() -> jm.Problem:
    """Create Maximum Independent Set (MIS) optimization model.

    Formulates the maximum independent set problem using binary decision
    variables, where the objective is to maximize the number of selected
    vertices subject to adjacency constraints.

    Returns:
        jm.Problem: JijModeling problem instance with all constraints and
        variables defined for the maximum independent set problem.
    """
    problem = jm.Problem("maximum_independent_set", sense=jm.ProblemSense.MAXIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        N = problem.Length("N", description="number of nodes")
        E = problem.Natural(
            "E", ndim=2, description="edge list as pairs (u,v), 0-based"
        )

        x = problem.BinaryVar("x", shape=(N,), description="1 if vertex i is selected")

        problem += jm.sum(x[v] for v in N)

        problem += problem.Constraint(
            "no_adjacent",
            lambda idx: x[E[idx, 0]] + x[E[idx, 1]] <= 1,
            domain=E.shape[0],
        )

    return problem
