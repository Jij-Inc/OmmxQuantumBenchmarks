import gzip
import math
import os
from fractions import Fraction

import numpy as np

from constants import NUM_S_SLACKS, NUM_Y_SLACKS

# Model parameters fixed in parameter_u3_c10.zpl of the original QOBLIB repository.
CASH = 1_000_000
UNIT = 100_000
DELTA = Fraction("0.001")
NU = Fraction("0.0001")
RHO = Fraction("0.000025")
C = CASH // UNIT

# b_tot for each number of assets, defined in gen_archive.sh of the original
# QOBLIB repository.
B_BY_ASSETS = {10: 4, 50: 20, 200: 50, 400: 100}


def zimpl_round(value: Fraction) -> int:
    """Round half away from zero, as ZIMPL's round() does.

    ZIMPL computes with exact rationals (GMP), so the coefficients must be
    derived with exact rational arithmetic before rounding; float64 errors of
    a single ulp around .5 boundaries would otherwise flip the result.
    """
    if value >= 0:
        return math.floor(value + Fraction(1, 2))
    return math.ceil(value - Fraction(1, 2))


def _read_data_lines(filepath: str) -> list[list[str]]:
    """Read non-empty, non-comment lines of a (possibly gzipped) text file."""
    opener = gzip.open if filepath.endswith(".gz") else open
    rows = []
    with opener(filepath, "rt", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def read_portfolio_instance(
    instance_dir: str, lam: Fraction, num_assets: int
) -> tuple[dict, list[str]]:
    """Read a QOBLIB portfolio instance directory and build the instance data.

    The directory must contain `stock_prices.txt.gz` and
    `covariance_matrices.txt.gz` (price and covariance data per day), as in the
    `instances/po_aXXX_tXX_*` directories of the original QOBLIB repository.

    Args:
        instance_dir: Path to the instance directory.
        lam: Risk weight lambda of the objective function (exact rational).
        num_assets: Number of assets (used to look up the asset limit B).

    Returns:
        A tuple of:
        - instance data dict for the placeholders of `model.create_problem`,
        - list of stock symbols in file order (defines the asset index i).
    """
    price_rows = _read_data_lines(os.path.join(instance_dir, "stock_prices.txt.gz"))
    cov_rows = _read_data_lines(
        os.path.join(instance_dir, "covariance_matrices.txt.gz")
    )

    # Symbols in order of first appearance; day indices must be 0..T-1.
    symbols: list[str] = []
    days: set[int] = set()
    raw_p: dict[tuple[str, int], Fraction] = {}
    for row in price_rows:
        if len(row) != 3:
            raise ValueError(
                f"Malformed stock price line in {instance_dir}: {' '.join(row)!r}"
            )
        day_str, symbol, price = row
        day = int(day_str)
        days.add(day)
        if symbol not in symbols:
            symbols.append(symbol)
        raw_p[(symbol, day)] = Fraction(price)

    if len(symbols) != num_assets:
        raise ValueError(
            f"Number of symbols in {instance_dir} is {len(symbols)}, "
            f"but expected {num_assets}."
        )
    if num_assets not in B_BY_ASSETS:
        raise ValueError(
            f"Unhandled number of assets ({num_assets}). "
            f"Choose from {sorted(B_BY_ASSETS)}."
        )

    A = len(symbols)
    T = len(days)
    if sorted(days) != list(range(T)):
        raise ValueError(f"Day indices in {instance_dir} are not contiguous from 0.")
    missing = [(s, t) for s in symbols for t in range(T) if (s, t) not in raw_p]
    if missing:
        raise ValueError(
            f"Missing stock price entries in {instance_dir}: {missing[:3]}"
            + (" ..." if len(missing) > 3 else "")
        )
    sym_idx = {s: i for i, s in enumerate(symbols)}

    # One unit is UNIT / raw_p[s, 0] shares of stock s; p is the exact rational
    # price of one unit of stock s on day t.
    p = [[raw_p[(s, t)] * UNIT / raw_p[(s, 0)] for t in range(T)] for s in symbols]

    cov: dict[tuple[int, int, int], Fraction] = {}
    for row in cov_rows:
        if len(row) != 4:
            raise ValueError(
                f"Malformed covariance line in {instance_dir}: {' '.join(row)!r}"
            )
        day_str, s1, s2, value = row
        if s1 not in sym_idx or s2 not in sym_idx:
            unknown = s1 if s1 not in sym_idx else s2
            raise ValueError(
                f"Unknown symbol {unknown!r} in covariance data of {instance_dir}."
            )
        day = int(day_str)
        if not 0 <= day < T:
            raise ValueError(
                f"Covariance day index {day} in {instance_dir} is outside 0..{T - 1}."
            )
        cov[(sym_idx[s1], sym_idx[s2], day)] = Fraction(value)
    # The upstream files contain the full A x A matrix for every day; a partial
    # file would otherwise silently leave risk coefficients at zero.
    if len(cov) != A * A * T:
        raise ValueError(
            f"Covariance data in {instance_dir} has {len(cov)} unique entries, "
            f"but expected {A * A * T} (full matrix for all days)."
        )

    tau = (1, -1)

    # risk[i, li, j, lj, t] = round(lam * tau_li * tau_lj * cov[i,j,t] * p[i,t] * p[j,t])
    risk = np.zeros((A, 2, A, 2, T))
    if lam != 0:
        for (i, j, t), cov_value in cov.items():
            value = zimpl_round(lam * cov_value * p[i][t] * p[j][t])
            risk[i, 0, j, 0, t] = value
            risk[i, 1, j, 1, t] = value
            # Opposite signs: round(-q) == -round(q) for half away from zero.
            risk[i, 0, j, 1, t] = -value
            risk[i, 1, j, 0, t] = -value

    # ret[i, l, t] = round(tau_l * (p[i,t+1] - p[i,t])); zero-padded at t = T-1.
    ret = np.zeros((A, 2, T))
    for i in range(A):
        for t in range(T - 1):
            for l in range(2):
                ret[i, l, t] = zimpl_round(tau[l] * (p[i][t + 1] - p[i][t]))

    # Transaction costs round(DELTA * p), split into the rebalancing coupling
    # between consecutive days (intermediate days only) and the single-day
    # costs of the first day (initial buy) and the last day (liquidation).
    tcost = np.array([[zimpl_round(DELTA * p[i][t]) for t in range(T)] for i in range(A)])
    tcost_pair = np.zeros((A, T))
    tcost_pair[:, 1:-1] = tcost[:, 1:-1]
    tcost_single = np.zeros((A, T))
    tcost_single[:, 0] = tcost[:, 0]
    tcost_single[:, -1] = tcost[:, -1]

    short_cost = np.array(
        [[zimpl_round(RHO * p[i][t]) for t in range(T)] for i in range(A)]
    )
    cash_interest = np.array(
        [zimpl_round(NU * UNIT * 2**k) for k in range(NUM_Y_SLACKS)]
    )

    instance_data = {
        "A": A,
        "T": T,
        "risk": risk,
        "ret": ret,
        "tcost_pair": tcost_pair,
        "tcost_single": tcost_single,
        "short_cost": short_cost.astype(np.float64),
        "cash_interest": cash_interest.astype(np.float64),
        "tau": np.array(tau, dtype=np.float64),
        "pow2_y": 2.0 ** np.arange(NUM_Y_SLACKS),
        "pow2_s": 2.0 ** np.arange(NUM_S_SLACKS),
        "C": float(C),
        "B": float(B_BY_ASSETS[num_assets]),
    }
    return instance_data, symbols
