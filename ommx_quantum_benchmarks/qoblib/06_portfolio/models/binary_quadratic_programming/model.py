import jijmodeling as jm

from constants import NUM_S_SLACKS, NUM_SIGNS, NUM_UNITS, NUM_Y_SLACKS


@jm.Problem.define(
    "Portfolio_binary_quadratic_programming", sense=jm.ProblemSense.MINIMIZE
)
def _portfolio_bqp(problem: jm.DecoratedProblem):
    """QOBLIB portfolio binary quadratic programming (BQP) model.

    This is the constrained formulation of the multi-period portfolio
    optimization problem (06-portfolio) defined in
    models/binary_quadratic_programming/bqp_u3_c10.zpl of the original QOBLIB
    repository.  The capital limit and the number-of-assets limit are kept as
    hard equality constraints (the unconstrained_quadratic_optimization model
    folds them into the objective as penalty terms instead).

    All cost coefficients are precomputed (including ZIMPL's `round()`) by
    `dat_reader.read_portfolio_instance`:

    - risk[i, li, j, lj, t]   = round(lambda * tau[li] * tau[lj]
                                      * cov[i, j, t] * p[i, t] * p[j, t])
    - ret[i, l, t]            = round(tau[l] * (p[i, t+1] - p[i, t]))
                                (zero-padded at t = T-1)
    - tcost_pair[i, t]        = round(delta * p[i, t]) for 0 < t < T-1
                                (transaction cost between consecutive days;
                                zero-padded at t = 0 and t = T-1)
    - tcost_single[i, t]      = round(delta * p[i, t]) at t = 0 and t = T-1
                                (first-day transaction and last-day
                                liquidation cost; zero elsewhere)
    - short_cost[i, t]        = round(rho * p[i, t])
    - cash_interest[k]        = round(nu * unit * 2^k)
    - tau                     = [1, -1] (long/short indicator)
    - pow2_y, pow2_s          = powers of two of the binary slack expansions
    - C, B                    = capital limit, asset limit

    Decision variables:
    - x[i, m, l, t]: hold the m-th unit of asset i with position tau[l]
                     (l = 0: long, l = 1: short) on day t
    - y[k, t]:       binary expansion slack of the capital limit constraint
    - s2[c, t]:      binary expansion slack of the asset limit constraint
                     (same name as the original ZIMPL model; OMMX subscripts
                     are 0-based)
    """
    # Placeholders for data from file
    A = problem.Length("A", description="Number of assets")
    T = problem.Length("T", description="Number of allocation days")
    risk = problem.Float("risk", ndim=5, description="Rounded risk coefficients")
    ret = problem.Float("ret", ndim=3, description="Rounded return coefficients")
    tcost_pair = problem.Float(
        "tcost_pair", ndim=2, description="Rounded transaction costs (rebalancing)"
    )
    tcost_single = problem.Float(
        "tcost_single", ndim=2, description="Rounded transaction costs (first/last day)"
    )
    short_cost = problem.Float(
        "short_cost", ndim=2, description="Rounded short selling costs"
    )
    cash_interest = problem.Float(
        "cash_interest", ndim=1, description="Rounded cash interests"
    )
    tau = problem.Float("tau", ndim=1, description="Position sign (1: long, -1: short)")
    pow2_y = problem.Float("pow2_y", ndim=1, description="Powers of two for y slack")
    pow2_s = problem.Float("pow2_s", ndim=1, description="Powers of two for s2 slack")
    C = problem.Float("C", description="Capital limit in units")
    B = problem.Float("B", description="Maximum number of assets")

    # Decision variables
    x = problem.BinaryVar(
        "x", shape=(A, NUM_UNITS, NUM_SIGNS, T), description="Variable x"
    )
    y = problem.BinaryVar("y", shape=(NUM_Y_SLACKS, T), description="Slack variable y")
    s2 = problem.BinaryVar(
        "s2", shape=(NUM_S_SLACKS, T), description="Slack variable s2"
    )

    # Risk term
    objective = jm.sum(
        risk[i, li, j, lj, t] * x[i, m, li, t] * x[j, n, lj, t]
        for t in T
        for i in A
        for m in jm.range(NUM_UNITS)
        for li in jm.range(NUM_SIGNS)
        for j in A
        for n in jm.range(NUM_UNITS)
        for lj in jm.range(NUM_SIGNS)
    )
    # Cash interest term
    objective -= jm.sum(
        cash_interest[k] * y[k, t] for t in T for k in jm.range(NUM_Y_SLACKS)
    )
    # Short selling cost term (l = 1 corresponds to tau = -1)
    objective += jm.sum(
        short_cost[i, t] * x[i, m, 1, t]
        for t in T
        for i in A
        for m in jm.range(NUM_UNITS)
    )
    # Return term (zero-padded at the last day)
    objective -= jm.sum(
        ret[i, l, t] * x[i, m, l, t]
        for t in T
        for i in A
        for m in jm.range(NUM_UNITS)
        for l in jm.range(NUM_SIGNS)
    )
    # Transaction costs for rebalancing between consecutive days
    # (tcost_pair is zero-padded at t = 0, so the t = 0 term vanishes)
    objective += jm.sum(
        tcost_pair[i, t]
        * (x[i, m, l, t - 1] + x[i, m, l, t] - 2 * x[i, m, l, t - 1] * x[i, m, l, t])
        for t in jm.range(1, T)
        for i in A
        for m in jm.range(NUM_UNITS)
        for l in jm.range(NUM_SIGNS)
    )
    # First-day transaction and last-day liquidation costs
    objective += jm.sum(
        tcost_single[i, t] * x[i, m, l, t]
        for t in T
        for i in A
        for m in jm.range(NUM_UNITS)
        for l in jm.range(NUM_SIGNS)
    )

    problem += objective

    # Capital limit: total position units plus slack must match C on every day
    problem += problem.Constraint(
        "capital_limit",
        (
            jm.sum(
                tau[l] * x[i, m, l, t]
                for i in A
                for m in jm.range(NUM_UNITS)
                for l in jm.range(NUM_SIGNS)
            )
            + jm.sum(pow2_y[k] * y[k, t] for k in jm.range(NUM_Y_SLACKS))
            == C
            for t in T
        ),
    )
    # Number-of-assets limit: total holdings plus slack must match B on every day
    problem += problem.Constraint(
        "asset_limit",
        (
            jm.sum(
                x[i, m, l, t]
                for i in A
                for m in jm.range(NUM_UNITS)
                for l in jm.range(NUM_SIGNS)
            )
            + jm.sum(pow2_s[c] * s2[c, t] for c in jm.range(NUM_S_SLACKS))
            == B
            for t in T
        ),
    )


def create_problem() -> jm.Problem:
    """
    Create the JijModeling problem definition.
    """
    return _portfolio_bqp
