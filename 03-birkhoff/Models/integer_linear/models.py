import jijmodeling as jm
import ommx
import numpy as np

import jijmodeling as jm
import numpy as np

def create_problem():
    """
    Create the JijModeling problem definition.
    """
    # Define placeholders for model parameters
    msize = jm.Placeholder("msize", description="Size of the square matrix (n x n)")
    scale = jm.Placeholder("scale", description="Scaling factor for the decomposition")

    # Define sets and parameter placeholders
    J = jm.Placeholder("J", ndim=1, description="Index set for matrix rows and columns")
    A_mn = jm.Placeholder("A", ndim=2, description="Target matrix to decompose (scaled doubly stochastic)")
    P_i = jm.Placeholder("P", ndim=3, description="Set of 3D permutation matrices (|I| x msize x msize)")
    I = jm.Placeholder("I", ndim=1, description="Index set for permutations")

    # Define decision variables
    x = jm.IntegerVar("x", shape=(I.shape[0],), lower_bound=0, upper_bound=scale,
                    description="Integer weights for each permutation matrix")
    z = jm.BinaryVar("z", shape=(I.shape[0],),
                    description="Binary activation variable for each permutation matrix")

    # Create the optimization problem
    problem = jm.Problem("Birkhoff Integer Decomposition", sense=jm.ProblemSense.MINIMIZE)

    # Define index element i
    i = jm.Element("i", belong_to=I)

    # Objective: minimize the number of permutation matrices used (i.e., number of active z[i])
    problem += jm.sum(i, z[i])

    # Constraint c1: sum of all weights equals the given scale
    problem += jm.Constraint("c1", jm.sum(i, x[i]) == scale)

    # Constraint c2: for all matrix elements (m,n), the weighted sum of permutation matrices equals the target matrix a3
    # This enforces A = sum_i x[i] * P_i[i]
    m = jm.Element("m", belong_to=J)
    n = jm.Element("n", belong_to=J)
    problem += jm.Constraint("c2", jm.sum(i, x[i] * P_i[i, m, n]) == A_mn[m, n], forall=[m, n])

    # Constraint c3: x[i] > 0 implies z[i] = 1 (linking integer variable with binary)
    # Enforces: x[i] <= scale * z[i]
    problem += jm.Constraint("c3", x[i] <= scale * z[i], forall=[i])

    return problem