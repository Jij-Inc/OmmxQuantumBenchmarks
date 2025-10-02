import jijmodeling as jm

def build_ip_formulation() -> jm.Problem:
    """Create integer programming formulation for the arc-based flow problem.

    This model aligns with the ZPL (0-based) specification, formulating a
    multi-commodity flow problem with integer scaling and big-M constraints.

    Sets:
        - N = {0..n−1}
        - A = {(i, j) | i ≠ j}
        - T = {(k, i, j) | i ≠ j and k ≠ j}

    Parameters:
        - n (int): Number of nodes.
        - t (ndarray, shape (n, n)): Demand matrix, with zero diagonal.
        - M (int): Big-M constant for capacity constraints.
        - intscale (int): Integer scaling factor.

    Variables:
        - x[i, j] ∈ {0, 1}: Binary arc selection variable.
        - f[k, i, j] ∈ ℤ, 0..intscale·M: Flow of commodity k on arc (i, j).
        - z ∈ ℤ, 0..intscale·M: Global upper bound on flow.

    Objective:
        - Minimize z.

    Constraints:
        - c1: ∀ i ∈ N: Σ_{j ≠ i} x[i, j] = 2 (out-degree = 2).
        - c2: ∀ j ∈ N: Σ_{i ≠ j} x[i, j] = 2 (in-degree = 2).
        - c11: ∀ (k, i), k ≠ i:
                  Σ_{j ≠ i} f[k, j, i] − Σ_{j ≠ i, j ≠ k} f[k, i, j]
                  = t[k, i]·intscale
        - c14: ∀ (k, i, j), i ≠ j, k ≠ j:
                  f[k, i, j] ≤ M·intscale·x[i, j]
        - c100: ∀ (i, j), i ≠ j:
                  Σ_{k ≠ j} f[k, i, j] ≤ z

    Returns:
        jm.Problem: JijModeling problem instance with all variables,
        objective, and constraints defined.
    """
    # ---- Placeholders ----
    n = jm.Placeholder("n")
    t = jm.Placeholder("t", ndim=2)   # (n,n)
    M = jm.Placeholder("M")
    intscale = jm.Placeholder("intscale")

    # ---- Indices ----
    i = jm.Element("i", belong_to=(0, n))
    j = jm.Element("j", belong_to=(0, n))
    k = jm.Element("k", belong_to=(0, n))

    # ---- Vars ----
    x = jm.BinaryVar("x", shape=(n, n), description="arc i->j selected")
    f = jm.IntegerVar(
        "f", shape=(n, n, n),
        lower_bound=0, upper_bound=intscale * M,
        description="flow of commodity k on arc i->j"
    )
    z = jm.IntegerVar("z", lower_bound=0, upper_bound=intscale * M)

    # ---- Problem & Objective ----
    problem = jm.Problem("ip_formulation", sense=jm.ProblemSense.MINIMIZE)
    problem += z

    # c1: ∀ i ∈ N :  Σ_{j ≠ i} x[i,j] = 2
    problem += jm.Constraint(
        "c1_outdeg_eq_2",
        jm.sum([(j, j != i)], x[i, j]) == 2,
        forall=[i]
    )
    
    # c2: ∀ j ∈ N :  Σ_{i ≠ j} x[i,j] = 2
    problem += jm.Constraint(
        "c2_indeg_eq_2",
        jm.sum([(i, i != j)], x[i, j]) == 2,
        forall=[j]
    )

    # c11: flow balance
    problem += jm.Constraint(
        "c11_flow_balance",
        jm.sum([(j, j != i)], f[k, j, i])
        - jm.sum([(j, (j != i) & (j != k))], f[k, i, j])
        == t[k, i] * intscale,
        forall=[k, (i, k != i)]
    )

    # c14: capacity bound
    problem += jm.Constraint(
        "c14_capacity_by_x",
        f[k, i, j] <= M * intscale * x[i, j],
        forall=[k, i, (j, (i != j) & (k != j))]
    )

    # c100: z upper bound on flow
    problem += jm.Constraint(
        "c100_z_upper_bounds_flow",
        jm.sum([(k, k != j)], f[k, i, j]) <= z,
        forall=[i, (j, i != j)]
    )

    return problem
