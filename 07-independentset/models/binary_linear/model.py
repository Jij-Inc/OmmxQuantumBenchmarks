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
    # Placeholders
    N = jm.Placeholder("N", description="number of nodes")
    E = jm.Placeholder("E", ndim=2, description="edge list as pairs (u,v), 0-based")

    # Decision variable: x[i] ∈ {0,1}
    x = jm.BinaryVar("x", shape=(N,), description="1 if vertex i is selected")

    # Objective: maximize sum_v x[v]
    v = jm.Element("v", belong_to=(0, N))
    obj = jm.sum(v, x[v])

    # Problem
    probem = jm.Problem("maximum_independent_set", sense=jm.ProblemSense.MAXIMIZE)
    probem += obj

    # Constraints: for all (u,v) in E, x[u] + x[v] <= 1
    e = jm.Element("e", belong_to=E)
    probem += jm.Constraint("no_adjacent", x[e[0]] + x[e[1]] <= 1, forall=e)

    return probem
