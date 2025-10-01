import jijmodeling as jm


def create_problem():
    # Define sets
    i_set = jm.Placeholder("I", ndim=1)
    k_set = jm.Placeholder("K", ndim=1)
    n = i_set.len_at(0, latex="n")

    # Define parameters
    P = jm.Placeholder("P", ndim=0)

    # Define decision variables
    x = jm.BinaryVar("x", shape=i_set.shape, description="Binary variable x")
    z = jm.BinaryVar(
        "z", shape=(i_set.shape[0], k_set.shape[0]), description="Binary variable z"
    )

    # Define elements
    k = jm.Element("k", belong_to=(0, n - 1))
    i = jm.Element("i", belong_to=(0, n - k - 1))

    # Define the problem
    problem = jm.Problem("Energy_Minimization", sense=jm.ProblemSense.MINIMIZE)

    # Define the objective function
    first_term = jm.sum(
        k, (jm.sum(i, 4 * z[i, k] - 2 * x[i] - 2 * x[i + k + 1] + 1)) ** 2
    )

    second_term = P * jm.sum(
        k,
        jm.sum(
            i,
            3 * z[i, k]
            - 2 * z[i, k] * x[i]
            - 2 * z[i, k] * x[i + k + 1]
            + x[i] * x[i + k + 1],
        ),
    )

    problem += first_term + second_term
    return problem
