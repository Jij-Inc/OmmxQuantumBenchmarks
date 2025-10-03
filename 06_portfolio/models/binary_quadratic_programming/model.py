import jijmodeling as jm


def createPortfolioBqpModel():
    m = jm.Problem("portfolioBqpMain")

    # -------- Scalars --------
    cash = jm.Placeholder("cash")
    unit = jm.Placeholder("unit")
    delta = jm.Placeholder("delta")
    rhoC = jm.Placeholder("rhoC")
    rhoS = jm.Placeholder("rhoS")
    q = jm.Placeholder("q")
    upscale = jm.Placeholder("upscale")
    bTot = jm.Placeholder("bTot")
    bCsh = jm.Placeholder("bCsh")

    # -------- Sets --------
    setS = jm.Placeholder("setS", ndim=1)  # assets
    setSc = jm.Placeholder("setSc", ndim=1)  # magnitude bits (not scaled by 2^m)
    setSl = jm.Placeholder("setSl", ndim=1)  # [0 = long, 1 = short]
    setTx = jm.Placeholder("setTx", ndim=1)  # time steps
    setCs1 = jm.Placeholder("setCs1", ndim=1)  # cash bits
    setCs2 = jm.Placeholder("setCs2", ndim=1)  # total position bits

    # -------- Parameters --------
    up = jm.Placeholder("up", ndim=2)  # scaled with unit (same as ZPL "up")
    cov = jm.Placeholder("cov", ndim=3)  # covariance

    slPnl = jm.Placeholder("slPnl", ndim=1)  # [+1, -1] (PnL/risk direction)
    slCash = jm.Placeholder("slCash", ndim=1)  # [-1, +1] (used in c2: cash balance)
    slNeg = jm.Placeholder("slNeg", ndim=1)  # [0, 1] (indicator for short positions)

    pow2Cs1 = jm.Placeholder("pow2Cs1", ndim=1)  # 2^k scaling for cash bits
    pow2Cs2 = jm.Placeholder("pow2Cs2", ndim=1)  # 2^k scaling for total position bits

    isFirst = jm.Placeholder("isFirst", ndim=1)  # indicator: first time step
    isLast = jm.Placeholder("isLast", ndim=1)  # indicator: last time step
    isMid = jm.Placeholder("isMid", ndim=1)  # indicator: middle time steps
    nextT = jm.Placeholder("nextT", ndim=1)  # index of next time step
    prevT = jm.Placeholder("prevT", ndim=1)  # index of previous time step

    # -------- Elements --------
    i = jm.Element("i", belong_to=setS)  # asset index
    j = jm.Element("j", belong_to=setS)  # asset index (for covariance pairs)
    m2 = jm.Element("m", belong_to=setSc)  # magnitude bit index
    n2 = jm.Element("n", belong_to=setSc)  # magnitude bit index (paired)
    sl = jm.Element("sl", belong_to=setSl)  # long/short flag
    sl2 = jm.Element("sl2", belong_to=setSl)  # long/short flag (paired)
    t = jm.Element("t", belong_to=setTx)  # time index
    k1 = jm.Element("k1", belong_to=setCs1)  # cash bit index
    k2 = jm.Element("k2", belong_to=setCs2)  # total position bit index

    # -------- Lengths --------
    nS = setS.len_at(0)  # number of assets
    nSc = setSc.len_at(0)  # number of magnitude bits
    nSl = setSl.len_at(0)  # number of long/short states
    nTx = setTx.len_at(0)  # number of time steps
    nCs1 = setCs1.len_at(0)  # number of cash bits
    nCs2 = setCs2.len_at(0)  # number of total position bits

    # -------- Decision variables --------
    xVar = jm.BinaryVar("xVar", shape=(nS, nSc, nSl, nTx))  # asset positions
    s1Var = jm.BinaryVar("s1Var", shape=(nCs1, nTx))  # cash bits
    s2Var = jm.BinaryVar("s2Var", shape=(nCs2, nTx))  # total position bits

    # -------- Helper functions --------
    def X(i_, m_, sl_, t_):
        return xVar[i_, m_, sl_, t_]

    def Xprev(i_, m_, sl_, t_):
        return xVar[i_, m_, sl_, prevT[t_]]

    def UP(i_, t_):
        return up[i_, t_]

    def UPnext(i_, t_):
        return up[i_, nextT[t_]]

    # ===== Objective terms: aligned with ZPL model =====

    # (1) Per-time-step base term:
    # q * Risk  - rhoC * unit * Σ(2^k * s1)  + rhoS * Σ(up * short)
    risk_interest = jm.sum(
        t,
        q
        * jm.sum(
            [i, m2, sl, j, n2, sl2],
            slPnl[sl]
            * slPnl[sl2]
            * cov[i, j, t]
            * X(i, m2, sl, t)
            * UP(i, t)
            * X(j, n2, sl2, t)
            * UP(j, t),
        )
        - rhoC * unit * jm.sum(k1, pow2Cs1[k1] * s1Var[k1, t])
        + rhoS * jm.sum([i, m2, sl], slNeg[sl] * UP(i, t) * X(i, m2, sl, t)),
    )

    # (2) Mid-period (time steps except first and last):
    # Profit - TransactionFee
    mid_profit_minus_fee = jm.sum(
        t,
        isMid[t]
        * (
            jm.sum([i, m2, sl], slPnl[sl] * (UPnext(i, t) - UP(i, t)) * X(i, m2, sl, t))
            - jm.sum(
                [i, m2, sl],
                delta
                * UP(i, t)
                * (
                    Xprev(i, m2, sl, t)
                    + X(i, m2, sl, t)
                    - 2 * Xprev(i, m2, sl, t) * X(i, m2, sl, t)
                ),
            )
        ),
    )

    # (3) First period:
    # Profit_first - TransactionFee_first
    first_day_term = jm.sum(
        t,
        isFirst[t]
        * (
            jm.sum([i, m2, sl], slPnl[sl] * (UPnext(i, t) - UP(i, t)) * X(i, m2, sl, t))
            - jm.sum([i, m2, sl], delta * UP(i, t) * X(i, m2, sl, t))
        ),
    )

    # (4) Last period:
    # TransactionFee_last
    last_day_fee = jm.sum(
        t, isLast[t] * jm.sum([i, m2, sl], delta * UP(i, t) * X(i, m2, sl, t))
    )

    # Objective function
    m += upscale * (
        risk_interest + mid_profit_minus_fee + first_day_term + last_day_fee
    )

    # -------- Constraints --------
    # c2: Cash balance per time step
    # Σ(slCash * x) + Σ(2^k * s1) = bCsh (not scaled by price or magnitude bits)
    m += jm.Constraint(
        "c2",
        jm.sum([i, m2, sl], slCash[sl] * xVar[i, m2, sl, t])
        + jm.sum(k1, pow2Cs1[k1] * s1Var[k1, t])
        == bCsh,
        forall=t,
    )

    # c3: Total position size per time step
    # Σ(x) + Σ(2^k * s2) = bTot (not scaled by magnitude bits)
    m += jm.Constraint(
        "c3",
        jm.sum([i, m2, sl], xVar[i, m2, sl, t]) + jm.sum(k2, pow2Cs2[k2] * s2Var[k2, t])
        == bTot,
        forall=t,
    )

    return m
