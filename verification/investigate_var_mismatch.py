"""For each VAR_MISMATCH steiner instance, compare upstream's nA/nL/nT/nR
against what the deployed jm1 OMMX encodes. If they differ, the migration
is correct and the registry was built from a different upstream snapshot
(data drift); if they match, there is a real migration bug to fix.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yuichironakano/マイドライブ/40_Qamomile/OmmxQuantumBenchmarks")
UPSTREAM = Path("/tmp/qoblib_upstream/04-steiner/instances")
sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Steiner  # noqa: E402

ROOT = REPO / "ommx_quantum_benchmarks/qoblib/04_steiner/models/integer_linear"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("st_reader", ROOT / "dat_reader.py").load_steiner_instance


def upstream_dims(name):
    p = UPSTREAM / name
    if not p.is_dir():
        return None
    raw = READER(str(p))
    return {
        "nA": len(raw["A"]),
        "nL": len(raw["L"]),
        "nR": len(raw["R"]),
        "nT": len(raw["T"]),
    }


def jm1_dims(name):
    """Decode jm1's variable encoding to infer nA / nL / nR / nT."""
    inst, _ = Steiner()("integer_linear", name)
    counts = Counter(v.name for v in inst.decision_variables)
    # x.shape = (nA, nT) ; y.shape = (nA, nL) ; z.shape = (nR, nT)
    nx = counts.get("x", 0)
    ny = counts.get("y", 0)
    nz = counts.get("z", 0)
    # Infer nA from subscripts: max(a)+1 in x
    sub_a_x = max((v.subscripts[0] for v in inst.decision_variables if v.name == "x"), default=-1) + 1
    sub_t_x = max((v.subscripts[1] for v in inst.decision_variables if v.name == "x"), default=-1) + 1
    sub_l_y = max((v.subscripts[1] for v in inst.decision_variables if v.name == "y"), default=-1) + 1
    sub_r_z = max((v.subscripts[0] for v in inst.decision_variables if v.name == "z"), default=-1) + 1
    sub_t_z = max((v.subscripts[1] for v in inst.decision_variables if v.name == "z"), default=-1) + 1
    return {
        "nA": sub_a_x,
        "nL": sub_l_y,
        "nR": sub_r_z,
        "nT": max(sub_t_x, sub_t_z),
        "_var_counts": dict(counts),
    }


CASES = [
    ("stp_s020_l2_t3_h3_rs97531", "MISMATCH"),
    ("stp_s020_l2_t4_h0_rs24098", "MISMATCH"),
    ("stp_s020_l3_t4_h0_rs24098", "MISMATCH"),
    ("stp_s020_l3_t4_h2_rs97531", "MISMATCH"),
    ("stp_s020_l4_t3_h3_rs97531", "MISSING_DAT"),
    ("stp_s020_l4_t4_h0_rs24098", "MISMATCH"),
    ("stp_s020_l5_t3_h3_rs24098", "MISMATCH"),
    ("stp_s020_l5_t4_h0_rs24098", "MISMATCH"),
    ("stp_s020_l5_t4_h3_rs24098", "MISSING_DAT"),
    ("stp_s030_l3_t4_h0_rs97531", "MISSING_DAT"),
]

for name, kind in CASES:
    print(f"--- {name} ({kind}) ---", flush=True)
    ud = upstream_dims(name)
    if ud is None:
        print(f"  upstream: MISSING", flush=True)
    else:
        print(
            f"  upstream: nA={ud['nA']} nL={ud['nL']} nR={ud['nR']} nT={ud['nT']}",
            flush=True,
        )
    j1 = jm1_dims(name)
    print(
        f"  jm1 reg : nA={j1['nA']} nL={j1['nL']} nR={j1['nR']} nT={j1['nT']}  {j1['_var_counts']}",
        flush=True,
    )
    if ud is not None:
        diff = [k for k in ("nA", "nL", "nR", "nT") if ud[k] != j1[k]]
        if diff:
            print(f"  ⇒ DATA DRIFT in: {diff}", flush=True)
        else:
            print(f"  ⇒ SAME DIMENSIONS — potential real bug!", flush=True)
    else:
        print(f"  ⇒ upstream missing, but registry has it → registry was built from different snapshot", flush=True)
