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
    # -------- Placeholders --------
    N = jm.Placeholder("N", description="number of nodes")
    # E is a (num_edges, 2) integer array (0-based endpoints)
    E = jm.Placeholder("E", ndim=2, description="edge list as pairs (u,v), 0-based")

    # -------- Decision vars --------
    x = jm.BinaryVar("x", shape=(N,), description="1 if vertex i is chosen")

    # -------- Objective (quadratic) --------
    v = jm.Element("v", belong_to=(0, N))
    e = jm.Element("e", belong_to=E)

    obj = jm.sum(v, x[v]) - 2 * jm.sum(e, x[e[0]] * x[e[1]])

    # -------- Problem (no constraints) --------
    prob = jm.Problem("mis_unconstrained_qubo", sense=jm.ProblemSense.MAXIMIZE)
    prob += obj

    return prob
