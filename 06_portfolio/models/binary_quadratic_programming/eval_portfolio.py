import numpy as np


def eval_portfolio_from_solution(
    *,
    # solution tensors (from the main model)
    xVal: np.ndarray,  # shape (nS, nSc, 2, nT), sl=0→long, sl=1→short
    s1Val: np.ndarray,  # shape (nCs1, nT)
    s2Val: np.ndarray,  # shape (nCs2, nT)  # only used for c3 checks (optional)
    # parameters / placeholders
    up: np.ndarray,  # shape (nS, nT); the ZPL uses "up" everywhere
    cov: np.ndarray,  # shape (nS, nS, nT)
    slPnl: np.ndarray,  # shape (2,), [+1, -1] for PnL/risk sign
    slCash: np.ndarray,  # shape (2,), [-1, +1] for c2 cash sign (optional check)
    slNeg: np.ndarray,  # shape (2,), [0, 1] short indicator
    pow2Cs1: np.ndarray,  # shape (nCs1,), values 2**k
    pow2Cs2: np.ndarray,  # shape (nCs2,), values 2**k  # only for c3 check
    isFirst: np.ndarray,  # shape (nT,), {0,1}
    isMid: np.ndarray,  # shape (nT,), {0,1}
    isLast: np.ndarray,  # shape (nT,), {0,1}
    nextT: np.ndarray,  # shape (nT,), int index (clamped at end)
    prevT: np.ndarray,  # shape (nT,), int index (clamped at start)
    # scalars
    unit: float,
    delta: float,
    rhoC: float,
    rhoS: float,
    q: float,
    upscale: float,
    bCsh: float,
    bTot: float,
    check_constraints: bool = True,
) -> dict:
    """Evaluate objective and components for the portfolio BQP in pure NumPy.

    Objective (your evaluation convention):
        obj = q * risk - profit

    Components:
        risk = upscale * Σ_t exposure(t)^T · cov[:,:,t] · exposure(t)
            where exposure[i,t] = up[i,t] * Σ_{m,sl} x[i,m,sl,t] * slPnl[sl]

        profit = upscale * (
                    rhoC * unit * Σ_t Σ_k 2^k s1[k,t]
                  - rhoS * Σ_t Σ_{i,m} up[i,t] * x[i,m,SHORT,t]
                  + pnl_mid + pnl_first
                  - trans_fee
                 )

        trans_fee = fee_mid + fee_first + fee_last
            fee_mid   = Σ_{t in mid} delta * Σ_i up[i,t] * Σ_m XOR(x[i,m,*,t], x[i,m,*,t-1])
            fee_first = Σ_{t is first} delta * Σ_i up[i,t] * Σ_{m,sl} x[i,m,sl,t]
            fee_last  = Σ_{t is last}  delta * Σ_i up[i,t] * Σ_{m,sl} x[i,m,sl,t]

        pnl_mid   = Σ_{t in mid} Σ_{i,m,sl} slPnl[sl] * (up[i,t+1]-up[i,t]) * x[i,m,sl,t]
        pnl_first = Σ_{t is first} same Δup formula applied at t=t0

    Args:
        xVal, s1Val, s2Val: Decision tensors extracted from a solution.
        up, cov, slPnl, slCash, slNeg, pow2Cs1, pow2Cs2, isFirst/isMid/isLast, nextT/prevT:
            Parameters/placeholders used by the model.
        unit, delta, rhoC, rhoS, q, upscale, bCsh, bTot: Scalar parameters.
        check_constraints: If True, verify c2 and c3 at each time step.

    Returns:
        dict with:
            - "obj": float
            - "risk": float
            - "profit": float
            - "trans_fee": float
            - Per-t breakdowns: "risk_t", "fee_mid_t", "fee_first_t", "fee_last_t",
                                "pnl_mid_t", "pnl_first_t",
                                "cash_interest_t", "short_cost_t"
            - Constraint checks: "c2_ok", "c3_ok" (bool or None)
    """
    # Basic shape checks (asserts keep code compact)
    nS, nSc, nSl, nT = xVal.shape
    assert nSl == 2, "sl dimension must be 2: [long=0, short=1]"
    LONG, SHORT = 0, 1

    # ---------- Signed exposure for risk / PnL ----------
    # x_sign[i,m,sl,t] = slPnl[sl] * x[i,m,sl,t]
    x_sign = xVal * slPnl[np.newaxis, np.newaxis, :, np.newaxis]  # (nS, nSc, 2, nT)

    # exposure[i,t] = up[i,t] * sum_{m,sl} x[i,m,sl,t] * slPnl[sl]
    exposure = x_sign.sum(axis=1).sum(axis=1) * up  # (nS, nT)

    # ---------- Risk ----------
    # risk_t = exposure(t)^T * cov[:,:,t] * exposure(t)
    risk_t = np.einsum("it,ijt,jt->t", exposure, cov, exposure)  # (nT,)
    risk_val = upscale * risk_t.sum()

    # ---------- Transaction fees ----------
    # Turnover via XOR per (i,m,sl,t):
    # XOR(a,b) = a + b - 2ab for binary a,b
    xor_long = (
        xVal[:, :, LONG, :]
        + xVal[:, :, LONG, prevT]
        - 2 * xVal[:, :, LONG, :] * xVal[:, :, LONG, prevT]
    )
    xor_short = (
        xVal[:, :, SHORT, :]
        + xVal[:, :, SHORT, prevT]
        - 2 * xVal[:, :, SHORT, :] * xVal[:, :, SHORT, prevT]
    )
    xor_sum = xor_long + xor_short  # (nS, nSc, nT)

    # Mid fee at each t: delta * sum_i up[i,t] * sum_m XOR(i,m,t)
    fee_mid_t = delta * (up * xor_sum.sum(axis=1)).sum(axis=0) * isMid  # (nT,)
    fee_mid = fee_mid_t.sum()

    # First/last fee at each t: delta * sum_i up[i,t] * sum_{m,sl} x[i,m,sl,t]
    pos_count_it = xVal.sum(axis=1).sum(axis=1)  # (nS, nT)
    fee_first_t = delta * (up * pos_count_it).sum(axis=0) * isFirst  # (nT,)
    fee_last_t = delta * (up * pos_count_it).sum(axis=0) * isLast  # (nT,)
    fee_first = fee_first_t.sum()
    fee_last = fee_last_t.sum()
    trans_fee = fee_mid + fee_first + fee_last

    # ---------- PnL ----------
    # Δup for each t (clamped with nextT)
    d_up = up[:, nextT] - up  # (nS, nT)
    signed_pos_it = x_sign.sum(axis=1).sum(axis=1)  # (nS, nT)

    # Mid PnL and first PnL per t
    pnl_mid_t = (d_up * signed_pos_it).sum(axis=0) * isMid  # (nT,)
    pnl_first_t = (d_up * signed_pos_it).sum(axis=0) * isFirst  # (nT,)
    pnl_mid = pnl_mid_t.sum()
    pnl_first = pnl_first_t.sum()

    # ---------- Cash interest / short borrow ----------
    # Cash interest: rhoC * unit * Σ_k 2^k s1[k,t]
    cash_interest_t = rhoC * unit * (pow2Cs1[:, None] * s1Val).sum(axis=0)  # (nT,)
    cash_interest = cash_interest_t.sum()

    # Short cost: rhoS * Σ_i up[i,t] * Σ_m x[i,m,SHORT,t]
    short_units_it = xVal[:, :, SHORT, :].sum(axis=1)  # (nS, nT)
    short_cost_t = rhoS * (up * short_units_it).sum(axis=0)  # (nT,)
    short_cost = short_cost_t.sum()

    # ---------- Profit and objective ----------
    profit = upscale * (cash_interest - short_cost + pnl_mid + pnl_first - trans_fee)
    obj = q * risk_val - profit

    # ---------- Optional constraint checks ----------
    c2_ok = c3_ok = None
    if check_constraints:
        # c2: Σ slCash* x + Σ 2^k s1 == bCsh  (per t)
        c2_lhs_t = (
            slCash[LONG] * xVal[:, :, LONG, :].sum(axis=1)
            + slCash[SHORT] * xVal[:, :, SHORT, :].sum(axis=1)
        ).sum(axis=0)
        c2_lhs_t += (pow2Cs1[:, None] * s1Val).sum(axis=0)
        c2_ok = np.allclose(c2_lhs_t, bCsh, atol=1e-6)

        # c3: Σ x + Σ 2^k s2 == bTot  (per t)
        c3_lhs_t = xVal.sum(axis=(0, 1, 2))
        c3_lhs_t += (pow2Cs2[:, None] * s2Val).sum(axis=0)
        c3_ok = np.allclose(c3_lhs_t, bTot, atol=1e-6)

    return {
        "obj": float(obj),
        "risk": float(risk_val),
        "profit": float(profit),
        "trans_fee": float(trans_fee),
        # per-t breakdowns
        "risk_t": risk_t,
        "fee_mid_t": fee_mid_t,
        "fee_first_t": fee_first_t,
        "fee_last_t": fee_last_t,
        "pnl_mid_t": pnl_mid_t,
        "pnl_first_t": pnl_first_t,
        "cash_interest_t": cash_interest_t,
        "short_cost_t": short_cost_t,
        # constraint checks
        "c2_ok": c2_ok,
        "c3_ok": c3_ok,
    }


def extract_solution_arrays_from_df(
    df, nS: int, nSc: int, nSl: int, nT: int, nCs1: int, nCs2: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert `decision_variables_df` (from OMMX solution) into x/s1/s2 NumPy arrays.

    Expected rows in `df`:
      - name == "xVar"  with subscripts = (i, m, sl, t), value ∈ {0,1}
      - name == "s1Var" with subscripts = (k, t),        value ∈ {0,1}
      - name == "s2Var" with subscripts = (k, t),        value ∈ {0,1}

    Args:
        df:   A DataFrame with columns ["name", "subscripts", "value"].
        nS:   Number of assets.
        nSc:  Number of magnitude bits.
        nSl:  Number of sign states (must be 2: long/short).
        nT:   Number of time steps.
        nCs1: Number of cash bits (s1).
        nCs2: Number of total bits (s2).

    Returns:
        (x_array, s1_array, s2_array):
            x_array  shape (nS, nSc, nSl, nT)
            s1_array shape (nCs1, nT)
            s2_array shape (nCs2, nT)
    """
    # xVar: shape (nS, nSc, nSl, nT)
    x_df = df[df["name"] == "xVar"]
    x_array = np.zeros((nS, nSc, nSl, nT))
    for _, row in x_df.iterrows():
        i, m, sl, t = row["subscripts"]
        x_array[int(i), int(m), int(sl), int(t)] = row["value"]

    # s1Var: shape (nCs1, nT)
    s1_df = df[df["name"] == "s1Var"]
    s1_array = np.zeros((nCs1, nT))
    for _, row in s1_df.iterrows():
        k, t = row["subscripts"]
        s1_array[int(k), int(t)] = row["value"]

    # s2Var: shape (nCs2, nT)
    s2_df = df[df["name"] == "s2Var"]
    s2_array = np.zeros((nCs2, nT))
    for _, row in s2_df.iterrows():
        k, t = row["subscripts"]
        s2_array[int(k), int(t)] = row["value"]

    return x_array, s1_array, s2_array
