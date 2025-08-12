import jijmodeling as jm

def create_problem():
    # Define sets
    I_set = jm.Placeholder("I", ndim=1)  # Placeholder for set I
    K_set = jm.Placeholder("K", ndim=1)  # Placeholder for set K
    n = I_set.len_at(0, latex="n")

    # Define decision variables
    x = jm.BinaryVar("x", shape=I_set.shape, description="Variable x")
    c = jm.IntegerVar(
        "c",
        shape=K_set.shape,
        lower_bound=-(n - 1),
        upper_bound=n - 1,
        description="Variable c",
    )

    # Define elements
    k = jm.Element("k", belong_to=K_set)
    t = jm.Element("i", belong_to=(0, n - k - 1))

    # Define the problem
    problem = jm.Problem(
        "Low Autocorrelation Binary Sequences (LABS)", sense=jm.ProblemSense.MINIMIZE
    )

    # Define the objective function
    problem += jm.sum(k, c[k] * c[k])

    # Define constraint c1
    problem += jm.Constraint(
        "c1",
        c[k] == jm.sum(t, (2 * x[t] - 1) * (2 * x[t + k + 1] - 1)),
        forall=k,
    )
    return problem
