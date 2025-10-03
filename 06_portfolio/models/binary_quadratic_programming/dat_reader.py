import numpy as np


def build_portfolio_instance_for_createPortfolioBqpModel(
    cov_file: str,
    price_file: str,
    *,
    # ZPL-aligned scalar defaults
    cash: float = 1_000_000,
    unit: float = 100_000,
    delta: float = 0.001,
    rho_c: float = 0.0001,
    rho_s: float = 0.000025,
    q: float = 1.0,
    upscale: float = 1.0,
    b_tot: int = 4,
    # Set size parameters (all 0-based continuous indices)
    ub: int = 3,  # setSc = {0..ub-1}
    cs1_bits: int = 4,  # setCs1 = {0..cs1_bits-1}
    cs2_bits: int = 5,  # setCs2 = {0..cs2_bits-1}
) -> dict:
    """
    Build an instance dictionary that matches createPortfolioBqpModel() placeholders.

    File formats:
      - cov_file:   lines of "t  stock_i  stock_j  value"
      - price_file: lines of "t  stock  price"
    """
    # ---------- Read covariance ----------
    cov_entries = []
    assets_set, times_set = set(), set()
    with open(cov_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            t_str, si, sj, v_str = line.split()
            t = int(t_str)
            v = float(v_str)
            cov_entries.append((t, si, sj, v))
            assets_set.update((si, sj))
            times_set.add(t)

    # ---------- Read stock prices ----------
    price_entries = []
    with open(price_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            t_str, s, v_str = line.split()
            t = int(t_str)
            v = float(v_str)
            price_entries.append((t, s, v))
            assets_set.add(s)
            times_set.add(t)

    # ---------- Sort and create mappings ----------
    asset_symbols = sorted(assets_set)
    time_values = sorted(times_set)
    nS, nT = len(asset_symbols), len(time_values)

    idxS = {a: i for i, a in enumerate(asset_symbols)}
    idxT = {t: i for i, t in enumerate(time_values)}

    # ---------- Build cov[i,j,t] (nS, nS, nT) ----------
    cov = np.zeros((nS, nS, nT), dtype=float)
    for t, si, sj, v in cov_entries:
        cov[idxS[si], idxS[sj], idxT[t]] = v
        # If the input only provides the upper triangle, you can also mirror here:
        # cov[idxS[sj], idxS[si], idxT[t]] = v

    # ---------- Build p[s,t] (nS, nT) ----------
    p = np.zeros((nS, nT), dtype=float)
    for t, s, v in price_entries:
        p[idxS[s], idxT[t]] = v

    # ---------- up = p * ucnt; ucnt[s] = unit / p[s, t_beg] ----------
    t_beg_col = 0
    denom = np.where(p[:, t_beg_col] == 0.0, 1e-12, p[:, t_beg_col])
    ucnt = unit / denom
    up = p * ucnt[:, None]  # (nS, nT)

    # ---------- Sets (all 0-based) ----------
    setS = list(range(nS))  # assets
    setSc = list(range(ub))  # magnitude bits (not multiplied by 2^m)
    setSl = [0, 1]
    setTx = list(range(nT))  # time
    setCs1 = list(range(cs1_bits))  # c2 expansion bits
    setCs2 = list(range(cs2_bits))  # c3 expansion bits

    # ---------- Time masks and neighbor indices ----------
    isFirst = np.zeros(nT, dtype=int)
    isLast = np.zeros(nT, dtype=int)
    isMid = np.zeros(nT, dtype=int)
    if nT > 0:
        isFirst[0] = 1
        isLast[-1] = 1
        if nT > 2:
            isMid[1:-1] = 1

    nextT = np.arange(nT, dtype=int)
    prevT = np.arange(nT, dtype=int)
    if nT > 1:
        nextT[:-1] = np.arange(1, nT, dtype=int)
        nextT[-1] = nT - 1
        prevT[1:] = np.arange(0, nT - 1, dtype=int)
        prevT[0] = 0

    # ---------- Direction vectors----------
    slCash = np.array([-1, +1], dtype=float)  # c2 (cash)
    ## PnL / Risk: short = -1, long = +1
    slPnl = np.array([-1, +1], dtype=float)
    ## Short indicator: short=1, long=0
    slNeg = np.array([1, 0], dtype=float)

    # ---------- 2^k vectors (for s1/s2 only) ----------
    pow2Cs1 = np.array([2**k for k in setCs1], dtype=float)
    pow2Cs2 = np.array([2**k for k in setCs2], dtype=float)

    # ---------- ZPL-aligned scalar ----------
    b_csh = float(cash) / float(unit)

    # ---------- Instance dictionary (aligned with model placeholders) ----------
    instance = {
        # Scalars
        "cash": float(cash),
        "unit": float(unit),
        "delta": float(delta),
        "rhoC": float(rho_c),
        "rhoS": float(rho_s),
        "q": float(q),
        "upscale": float(upscale),
        "bTot": int(b_tot),
        "bCsh": float(b_csh),
        # Sets
        "setS": setS,
        "setSc": setSc,
        "setSl": setSl,
        "setTx": setTx,
        "setCs1": setCs1,
        "setCs2": setCs2,
        # Parameters
        "up": up,  # (nS, nT). Note: ZPL formulas all use "up"
        "cov": cov,  # (nS, nS, nT)
        "slPnl": slPnl,  # (2,)
        "slCash": slCash,  # (2,)
        "slNeg": slNeg,  # (2,)
        "pow2Cs1": pow2Cs1,
        "pow2Cs2": pow2Cs2,
        "isFirst": isFirst,
        "isLast": isLast,
        "isMid": isMid,
        "nextT": nextT,
        "prevT": prevT,
    }
    return instance
