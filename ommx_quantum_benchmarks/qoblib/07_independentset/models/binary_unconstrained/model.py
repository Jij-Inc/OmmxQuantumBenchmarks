import jijmodeling as jm


def build_mis_unconstrained() -> jm.Problem:
    """Create unconstrained Maximum Independent Set (MIS) model in QUBO form.

    Formulates the MIS problem as a quadratic unconstrained binary optimization
    (QUBO), where the objective maximizes the number of selected vertices while
    penalizing the selection of adjacent vertices.

    Objective:
        maximize  Σ_v x[v]  −  2 Σ_(u,v)∈E x[u]·x[v]

    Assumptions:
        - Edge list E must contain each undirected edge only once.
        - No self-loops are allowed.

    Returns:
        jm.Problem: JijModeling problem instance with objective defined in
        quadratic unconstrained form.
    """
    problem = jm.Problem("mis_unconstrained_qubo", sense=jm.ProblemSense.MAXIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        N = problem.Length("N", description="number of nodes")
        E = problem.Natural(
            "E", ndim=2, description="edge list as pairs (u,v), 0-based"
        )

        x = problem.BinaryVar("x", shape=(N,), description="1 if vertex i is chosen")

        problem += jm.sum(x[v] for v in N) - 2 * jm.sum(
            x[E[idx, 0]] * x[E[idx, 1]] for idx in E.len_at(0)
        )

    return problem
