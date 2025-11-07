import jijmodeling as jm


def create_problem():
    # Define sets
    i_set = jm.Placeholder("I", ndim=1)  # Placeholder for set I
    k_set = jm.Placeholder("K", ndim=1)  # Placeholder for set K
    n = i_set.len_at(0, latex="n")

    # Define decision variables
    x = jm.BinaryVar("x", shape=i_set.shape, description="Variable x")

    # Define elements
    k = jm.Element("k", belong_to=k_set)
    i = jm.Element("i", belong_to=(0, n - k - 1))

    # Define the problem
    problem = jm.Problem(
        "Low Autocorrelation Binary Sequences (LABS)", sense=jm.ProblemSense.MINIMIZE
    )

    # Define the objective function
    problem += jm.sum(k, jm.sum(i, (2 * x[i] - 1) * (2 * x[i + k + 1] - 1)) ** 2)

    return problem
