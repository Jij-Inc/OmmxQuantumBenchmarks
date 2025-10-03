import jijmodeling as jm


def createPortfolioUQOModel():
    """
    Single-variable (x only) UQO model.
    x has length nAll = nS*nSc*2*nTx + nCs1*nTx + nCs2*nTx.
      - assetIndex[i,m,sl,t]   ∈ [0, base_c2)   # 資產段
      - c2Index[k1,t]          ∈ [base_c2, base_c3)
      - c3Index[k2,t]          ∈ [base_c3, nAll)
    """

    m = jm.Problem("portfolioBqpMainUQO_X690")

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
    penalty7 = jm.Placeholder("penalty7")

    # -------- Sets --------
    setS = jm.Placeholder("setS", ndim=1)
    setSc = jm.Placeholder("setSc", ndim=1)
    setSl = jm.Placeholder("setSl", ndim=1)
    setTx = jm.Placeholder("setTx", ndim=1)
    setCs1 = jm.Placeholder("setCs1", ndim=1)
    setCs2 = jm.Placeholder("setCs2", ndim=1)

    # -------- Parameters --------
    up = jm.Placeholder("up", ndim=2)  # (nS, nTx)
    cov = jm.Placeholder("cov", ndim=3)  # (nS, nS, nTx)
    slPnl = jm.Placeholder("slPnl", ndim=1)  # [+1,-1] or你的順序
    slCash = jm.Placeholder("slCash", ndim=1)  # [-1,+1]
    slNeg = jm.Placeholder("slNeg", ndim=1)  # [0,1]

    isFirst = jm.Placeholder("isFirst", ndim=1)
    isLast = jm.Placeholder("isLast", ndim=1)
    isMid = jm.Placeholder("isMid", ndim=1)
    nextT = jm.Placeholder("nextT", ndim=1)
    prevT = jm.Placeholder("prevT", ndim=1)

    pow2Cs1 = jm.Placeholder("pow2Cs1", ndim=1)
    pow2Cs2 = jm.Placeholder("pow2Cs2", ndim=1)

    # -------- Index mappings --------
    assetIndex = jm.Placeholder("assetIndex", ndim=4)  # (nS, nSc, 2, nTx)
    c2Index = jm.Placeholder("c2Index", ndim=2)  # (nCs1, nTx)
    c3Index = jm.Placeholder("c3Index", ndim=2)  # (nCs2, nTx)

    nAll = jm.Placeholder("nAll")  # e.g., 690
    x = jm.BinaryVar("x", shape=(nAll,))

    # -------- Elements --------
    i = jm.Element("i", belong_to=setS)
    j = jm.Element("j", belong_to=setS)
    m2 = jm.Element("m", belong_to=setSc)
    n2 = jm.Element("n", belong_to=setSc)
    sl = jm.Element("sl", belong_to=setSl)
    sl2 = jm.Element("sl2", belong_to=setSl)
    t = jm.Element("t", belong_to=setTx)
    k1 = jm.Element("k1", belong_to=setCs1)
    k2 = jm.Element("k2", belong_to=setCs2)

    # ===== Helpers: pick from 1D x =====
    def XA(i_, m_, sl_, t_):
        return x[assetIndex[i_, m_, sl_, t_]]

    def XC2(k1_, t_):
        return x[c2Index[k1_, t_]]

    def XC3(k2_, t_):
        return x[c3Index[k2_, t_]]

    # ===== Objective =====

    # (1) per-time base: q*risk - rhoC*unit*Σ(2^k*s1) + rhoS*Σ(up*short)
    risk_interest = jm.sum(
        t,
        q
        * jm.sum(
            [i, m2, sl, j, n2, sl2],
            slPnl[sl]
            * slPnl[sl2]
            * cov[i, j, t]
            * up[i, t]
            * XA(i, m2, sl, t)
            * up[j, t]
            * XA(j, n2, sl2, t),
        )
        - rhoC * unit * jm.sum(k1, pow2Cs1[k1] * XC2(k1, t))
        + rhoS * jm.sum([i, m2, sl], slNeg[sl] * up[i, t] * XA(i, m2, sl, t)),
    )

    # (2) mid window (exclude first/last): profit - transaction fee
    mid_profit_minus_fee = jm.sum(
        t,
        isMid[t]
        * (
            jm.sum(
                [i, m2, sl], slPnl[sl] * (up[i, nextT[t]] - up[i, t]) * XA(i, m2, sl, t)
            )
            - jm.sum(
                [i, m2, sl],
                delta
                * up[i, t]
                * (
                    XA(i, m2, sl, prevT[t])
                    + XA(i, m2, sl, t)
                    - 2 * XA(i, m2, sl, prevT[t]) * XA(i, m2, sl, t)
                ),
            )
        ),
    )

    # (3) first day
    first_day_term = jm.sum(
        t,
        isFirst[t]
        * (
            jm.sum(
                [i, m2, sl], slPnl[sl] * (up[i, nextT[t]] - up[i, t]) * XA(i, m2, sl, t)
            )
            - jm.sum([i, m2, sl], delta * up[i, t] * XA(i, m2, sl, t))
        ),
    )

    # (4) last day
    last_day_fee = jm.sum(
        t,
        isLast[t] * jm.sum([i, m2, sl], delta * up[i, t] * XA(i, m2, sl, t)),
    )

    core_obj = upscale * (
        risk_interest + mid_profit_minus_fee + first_day_term + last_day_fee
    )

    # ===== QUBO penalties on x =====
    # c2: Σ(slCash * x_asset) + Σ(2^k * x_c2) == bCsh
    lhs_c2 = jm.sum([i, m2, sl], slCash[sl] * XA(i, m2, sl, t)) + jm.sum(
        k1, pow2Cs1[k1] * XC2(k1, t)
    )
    pen_c2 = jm.sum(t, penalty7 * (lhs_c2 - bCsh) * (lhs_c2 - bCsh))

    # c3: Σ(x_asset) + Σ(2^k * x_c3) == bTot
    lhs_c3 = jm.sum([i, m2, sl], XA(i, m2, sl, t)) + jm.sum(
        k2, pow2Cs2[k2] * XC3(k2, t)
    )
    pen_c3 = jm.sum(t, penalty7 * (lhs_c3 - bTot) * (lhs_c3 - bTot))

    m += core_obj + pen_c2 + pen_c3
    return m
