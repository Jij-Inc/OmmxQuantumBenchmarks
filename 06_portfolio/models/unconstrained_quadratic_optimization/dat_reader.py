import numpy as np


def build_portfolio_instance_for_createPortfolioUqoModel(
    cov_file: str,
    price_file: str,
    *,
    # ZPL-aligned scalars
    cash: float = 1_000_000,
    unit: float = 100_000,
    delta: float = 0.001,
    rho_c: float = 0.0001,
    rho_s: float = 0.000025,
    q: float = 1.0,
    upscale: float = 1.0,
    b_tot: int = 4,
    # set sizes (0-based)
    ub: int = 3,  # setSc = {0..ub-1}
    cs1_bits: int = 4,  # setCs1 = {0..cs1_bits-1}
    cs2_bits: int = 5,  # setCs2 = {0..cs2_bits-1}
    # QUBO penalties
    penalty7: float = 1e6,
    penaltyGhost: float = 1e7,  # ← 補上（ghost s1/s2 用；X=690 版不會用到）
) -> dict:
    """Build instance dict for the X=690 (1D x) UQO portfolio model.

    Files:
        cov_file:   lines of "t  stock_i  stock_j  value"
        price_file: lines of "t  stock  price"
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

    # ---------- Read prices ----------
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

    # ---------- Sorted indices ----------
    asset_symbols = sorted(assets_set)
    time_values = sorted(times_set)
    nS, nT = len(asset_symbols), len(time_values)

    idxS = {a: i for i, a in enumerate(asset_symbols)}
    idxT = {t: i for i, t in enumerate(time_values)}

    # ---------- Tensors ----------
    cov = np.zeros((nS, nS, nT), dtype=float)
    for t, si, sj, v in cov_entries:
        cov[idxS[si], idxS[sj], idxT[t]] = v

    p = np.zeros((nS, nT), dtype=float)
    for t, s, v in price_entries:
        p[idxS[s], idxT[t]] = v

    # up = p * (unit / p[:, t_beg]); use first column as baseline
    t_beg_col = 0
    denom = np.where(p[:, t_beg_col] == 0.0, 1e-12, p[:, t_beg_col])
    ucnt = unit / denom
    up = p * ucnt[:, None]  # (nS, nT)

    # ---------- Sets (0-based) ----------
    setS = list(range(nS))
    setSc = list(range(ub))
    setSl = [0, 1]  # 0=short, 1=long
    setTx = list(range(nT))
    setCs1 = list(range(cs1_bits))
    setCs2 = list(range(cs2_bits))

    # ---------- Time masks & neighbors ----------
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

    # ---------- Direction vectors ----------
    slCash = np.array([-1, +1], dtype=float)  # c2
    slPnl = np.array([-1, +1], dtype=float)  # PnL/Risk
    slNeg = np.array([1, 0], dtype=float)  # short indicator

    # ---------- 2^k weights ----------
    pow2Cs1 = np.array([2**k for k in setCs1], dtype=float)
    pow2Cs2 = np.array([2**k for k in setCs2], dtype=float)

    # ---------- ZPL-aligned scalar ----------
    b_csh = float(cash) / float(unit)

    # ---------- 1D x index mappings ----------
    nSc, nSl = len(setSc), len(setSl)
    nCs1, nCs2 = len(setCs1), len(setCs2)

    # assets (i,m,sl,t)
    assetIndex = np.zeros((nS, nSc, nSl, nT), dtype=int)
    for i in range(nS):
        for m in range(nSc):
            for sl in range(nSl):
                base = ((i * nSc) + m) * nSl + sl
                start = base * nT
                assetIndex[i, m, sl, :] = start + np.arange(nT, dtype=int)

    base_c2 = nS * nSc * nSl * nT

    # c2 (k1,t)
    c2Index = np.zeros((nCs1, nT), dtype=int)
    for k in range(nCs1):
        c2Index[k, :] = base_c2 + k * nT + np.arange(nT, dtype=int)

    base_c3 = base_c2 + nCs1 * nT

    # c3 (k2,t)
    c3Index = np.zeros((nCs2, nT), dtype=int)
    for k in range(nCs2):
        c3Index[k, :] = base_c3 + k * nT + np.arange(nT, dtype=int)

    nAll = int(base_c3 + nCs2 * nT)

    # ---------- Assemble ----------
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
        "penalty7": float(penalty7),
        "penaltyGhost": float(penaltyGhost),  # ← 補上
        # Sets
        "setS": setS,
        "setSc": setSc,
        "setSl": setSl,
        "setTx": setTx,
        "setCs1": setCs1,
        "setCs2": setCs2,
        # Parameters
        "up": up,
        "cov": cov,
        "slPnl": slPnl,
        "slCash": slCash,
        "slNeg": slNeg,
        "pow2Cs1": pow2Cs1,
        "pow2Cs2": pow2Cs2,
        "isFirst": isFirst,
        "isLast": isLast,
        "isMid": isMid,
        "nextT": nextT,
        "prevT": prevT,
        # Index maps for 1D x
        "assetIndex": assetIndex,
        "c2Index": c2Index,
        "c3Index": c3Index,
        "nAll": nAll,
    }
    return instance
