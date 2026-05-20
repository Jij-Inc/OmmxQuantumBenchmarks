import jijmodeling as jm


def create_problem():
    """
    Create the JijModeling problem definition.
    """
    problem = jm.Problem(
        "Birkhoff Integer Decomposition", sense=jm.ProblemSense.MINIMIZE
    )

    @problem.update
    def _(problem: jm.DecoratedProblem):
        msize = problem.Natural(
            "msize", description="Size of the square matrix (n x n)"
        )
        scale = problem.Natural(
            "scale", description="Scaling factor for the decomposition"
        )

        J = problem.Natural(
            "J", ndim=1, description="Index set for matrix rows and columns"
        )
        A_mn = problem.Integer(
            "A",
            ndim=2,
            description="Target matrix to decompose (scaled doubly stochastic)",
        )
        P_i = problem.Integer(
            "P",
            ndim=3,
            description="Set of 3D permutation matrices (|I| x msize x msize)",
        )
        I = problem.Natural("I", ndim=1, description="Index set for permutations")

        x = problem.IntegerVar(
            "x",
            shape=(I.len_at(0),),
            lower_bound=0,
            upper_bound=scale,
            description="Integer weights for each permutation matrix",
        )
        z = problem.BinaryVar(
            "z",
            shape=(I.len_at(0),),
            description="Binary activation variable for each permutation matrix",
        )

        problem += jm.sum(z[i] for i in I.len_at(0))

        problem += problem.Constraint(
            "c1", jm.sum(x[i] for i in I.len_at(0)) == scale
        )

        problem += problem.Constraint(
            "c2",
            lambda m, n: jm.sum(x[i] * P_i[i, m, n] for i in I.len_at(0)) == A_mn[m, n],
            domain=jm.product(J.len_at(0), J.len_at(0)),
        )

        problem += problem.Constraint(
            "c3",
            lambda i: x[i] <= scale * z[i],
            domain=I.len_at(0),
        )

    return problem
