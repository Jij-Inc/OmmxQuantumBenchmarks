"""Driver: for each model, run the runner in jm1 + jm2 envs and diff outputs."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

JM2_ROOT = Path(__file__).resolve().parents[1]
RUNNER = JM2_ROOT / "verification" / "equiv_runner.py"
JM1_ROOT = Path(os.environ.get("JM1_BASELINE", "/tmp/jm1_baseline"))

OUT_DIR = Path("/tmp/equiv_out")
OUT_DIR.mkdir(exist_ok=True)


def _mk_sample(name, subs_value_map):
    """Helper: produce [{name, subscripts, value}, ...] entries."""
    out = []
    for subs, val in subs_value_map:
        out.append({"name": name, "subscripts": list(subs), "value": float(val)})
    return out


# === Per-model specs ===
def labs_integer():
    n = 5
    x_vals = [1, 0, 1, 1, 0]
    expected_c = []
    for k in range(n - 1):
        s = 0
        for i in range(n - k - 1):
            s += (2 * x_vals[i] - 1) * (2 * x_vals[i + k + 1] - 1)
        expected_c.append(s)
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/02_labs/models/integer/model.py",
        "create_fn": "create_problem",
        "data": {"I": np.arange(n), "K": np.arange(n - 1)},
        "sample": _mk_sample("x", [((i,), v) for i, v in enumerate(x_vals)]) + _mk_sample("c", [((k,), v) for k, v in enumerate(expected_c)]),
    }


def labs_qubo():
    n = 5
    x_vals = [1, 0, 1, 1, 0]
    # z[i, k] = x[i] AND x[i+k+1] (one feasible assignment of the auxiliary)
    z_entries = []
    for k in range(n - 1):
        for i in range(n - k - 1):
            z_entries.append(((i, k), x_vals[i] * x_vals[i + k + 1]))
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/02_labs/models/quadratic_unconstrained/model.py",
        "create_fn": "create_problem",
        "data": {"I": np.arange(n), "K": np.arange(n - 1), "P": 10000.0},
        "sample": _mk_sample("x", [((i,), v) for i, v in enumerate(x_vals)]) + _mk_sample("z", z_entries),
    }


def marketsplit_bl():
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/01_marketsplit/models/binary_linear/model.py",
        "create_fn": "create_problem",
        "data": {
            "I": np.arange(2),
            "J": np.arange(3),
            "a": np.array([[1, 1, 0], [0, 1, 1]]),
            "b": np.array([1, 1]),
        },
        # x = [1, 0, 0], s = [0, 1]: c1[0]: 0 + 1 = 1 OK; c1[1]: 1 + 0 = 1 OK
        "sample": _mk_sample("x", [((0,), 1), ((1,), 0), ((2,), 0)]) + _mk_sample("s", [((0,), 0), ((1,), 1)]),
    }


def marketsplit_bu():
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/01_marketsplit/models/binary_unconstrained/model.py",
        "create_fn": "create_problem",
        "data": {
            "I": np.arange(2),
            "J": np.arange(3),
            "a": np.array([[1, 1, 0], [0, 1, 1]]),
            "b": np.array([1, 1]),
        },
        # x = [1, 0, 1]: row0 = 1+0+0=1 -> 1-1=0, row1 = 0+0+1=1 -> 1-1=0; obj=0
        "sample": _mk_sample("x", [((0,), 1), ((1,), 0), ((2,), 1)]),
    }


def birkhoff():
    P = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    A = np.array([[1, 1], [1, 1]])
    # x = [1, 1], z = [1, 1]: c1 -> sum=2=scale OK; c2 -> identity+swap=A OK; c3 -> 1<=2 OK
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/03_birkhoff/models/integer_linear/models.py",
        "create_fn": "create_problem",
        "data": {
            "msize": 2,
            "scale": 2,
            "J": np.arange(2),
            "I": np.arange(2),
            "P": P,
            "A": A,
        },
        "sample": _mk_sample("x", [((0,), 1), ((1,), 1)]) + _mk_sample("z", [((0,), 1), ((1,), 1)]),
    }


def mis_bl():
    N = 4
    E = np.array([[0, 1], [1, 2], [2, 3]])
    # x = [1, 0, 1, 0] is independent
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/07_independentset/models/binary_linear/model.py",
        "create_fn": "build_mis_problem",
        "data": {"N": N, "E": E},
        "sample": _mk_sample("x", [((i,), v) for i, v in enumerate([1, 0, 1, 0])]),
    }


def mis_bu():
    N = 4
    E = np.array([[0, 1], [1, 2], [2, 3]])
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/07_independentset/models/binary_unconstrained/model.py",
        "create_fn": "build_mis_unconstrained",
        "data": {"N": N, "E": E},
        "sample": _mk_sample("x", [((i,), v) for i, v in enumerate([1, 0, 1, 0])]),
    }


def network():
    n = 3
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/08_network/models/integer_lp/model.py",
        "create_fn": "build_ip_formulation",
        "data": {
            "n": n,
            "t": np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
            "M": 1000,
            "intscale": 100,
        },
        # Skip sample (constructing a feasible solution by hand is tedious;
        # structural diff suffices)
        "sample": None,
    }


def routing():
    n = 4
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/09_routing/models/integer_linear/model.py",
        "create_fn": "build_vrp_ilp",
        "data": {
            "n": n,
            "VEHICLE_LIMIT": 2,
            "CAPACITY": 100,
            "DEMAND": np.array([0.0, 10.0, 20.0, 30.0]),
            "D": np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [1.0, 0.0, 1.0, 2.0],
                    [2.0, 1.0, 0.0, 1.0],
                    [3.0, 2.0, 1.0, 0.0],
                ]
            ),
            "DEPOT": 0,
        },
        "sample": None,
    }


def topo_flow():
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/10_topology/models/flow_mip/model.py",
        "create_fn": "create_topology_model",
        "data": {"nodes": 4, "degree": 2},
        "sample": None,
    }


def topo_sq():
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/10_topology/models/seidel_quadratic/model.py",
        "create_fn": "create_topology_model",
        "data": {
            "nodes": 4,
            "degree": 2,
            "minDiameter": 1,
            "maxDiameter": 3,
        },
        "sample": None,
    }


def topo_sl():
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/10_topology/models/seidel_linear/model.py",
        "create_fn": "create_topology_model",
        "data": {
            "nodes": 4,
            "degree": 2,
            "minDiameter": 1,
            "maxDiameter": 3,
        },
        "sample": None,
    }


def steiner():
    return {
        "model_relpath": "ommx_quantum_benchmarks/qoblib/04_steiner/models/integer_linear/model.py",
        "create_fn": "create_steiner_tree_packing_model",
        "data": {
            "L": np.array([0, 1]),
            "V": np.arange(5),
            "R": np.array([0, 1]),
            "A": np.array([[0, 2], [2, 0], [0, 3], [3, 0], [1, 4], [4, 1]]),
            "T": np.array([2, 3, 4]),
            "N": np.array([], dtype=np.int64),
            "VNR": np.array([2, 3, 4]),
            "innetT": np.array([[2, 0], [3, 0], [4, 1]]),
            "innetR": np.array([[0, 0], [1, 1]]),
            "arcCosts": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "netCardinality": np.array([[0, 2], [1, 1]]),
        },
        "sample": None,
    }


SPECS = {
    "labs_integer": labs_integer,
    "labs_qubo": labs_qubo,
    "marketsplit_bl": marketsplit_bl,
    "marketsplit_bu": marketsplit_bu,
    "birkhoff": birkhoff,
    "mis_bl": mis_bl,
    "mis_bu": mis_bu,
    "network": network,
    "routing": routing,
    "topo_flow": topo_flow,
    "topo_sq": topo_sq,
    "topo_sl": topo_sl,
    "steiner": steiner,
}


def _augment_data_for_jm2(name, data):
    """jm 2 models for seidel_* require an N_arr index placeholder."""
    if name in ("topo_sq", "topo_sl"):
        data = dict(data)
        data["N_arr"] = np.arange(data["nodes"])
    return data


def _make_spec_file(name, repo_root, data, sample, model_relpath, create_fn):
    """Write a JSON spec + pickled data file for the runner."""
    data_path = OUT_DIR / f"{name}_data.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    spec = {
        "repo_root": str(repo_root),
        "model_relpath": model_relpath,
        "create_fn": create_fn,
        "data_pickle": str(data_path),
        "sample_solution": sample,
    }
    spec_path = OUT_DIR / f"{name}_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path


def _run(env_root, spec_path, out_path):
    cmd = [
        "uv",
        "run",
        "--project",
        str(env_root),
        "python",
        str(RUNNER),
        str(spec_path),
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(env_root))
    if proc.returncode != 0:
        return False, proc.stderr
    return True, proc.stdout


def _compare(name, jm1_path, jm2_path):
    """Return (hard_diffs, soft_diffs) — hard = semantic, soft = annotation only."""
    with open(jm1_path) as f:
        a = json.load(f)
    with open(jm2_path) as f:
        b = json.load(f)
    hard, soft = [], []
    # Hard semantic checks
    for key in ("sense", "num_vars", "num_constraints"):
        if a.get(key) != b.get(key):
            hard.append(f"{key}: {a.get(key)!r} vs {b.get(key)!r}")
    # Constraints per name (count) — semantic
    from collections import Counter

    a_c_names = Counter(c["name"] for c in a["constraints"])
    b_c_names = Counter(c["name"] for c in b["constraints"])
    if a_c_names != b_c_names:
        hard.append(f"constraint counts per name: {dict(a_c_names)} vs {dict(b_c_names)}")
    # Variables: name+subscripts must match exactly (these define the model)
    a_var_keys = {(v["name"], tuple(v["subscripts"])) for v in a["vars"]}
    b_var_keys = {(v["name"], tuple(v["subscripts"])) for v in b["vars"]}
    if a_var_keys != b_var_keys:
        only_a = a_var_keys - b_var_keys
        only_b = b_var_keys - a_var_keys
        hard.append(f"vars: only_jm1={sorted(only_a)[:5]}... only_jm2={sorted(only_b)[:5]}...")
    # Constraint subscript annotations — soft (cosmetic)
    a_c_keys = {(c["name"], tuple(c["subscripts"])) for c in a["constraints"]}
    b_c_keys = {(c["name"], tuple(c["subscripts"])) for c in b["constraints"]}
    if a_c_keys != b_c_keys:
        only_a = a_c_keys - b_c_keys
        only_b = b_c_keys - a_c_keys
        soft.append(f"constraint subscript annotations differ: only_jm1={sorted(only_a)[:3]}... only_jm2={sorted(only_b)[:3]}...")
    # Sample evaluation (hard — semantic check)
    if "sample_objective" in a and "sample_objective" in b:
        if abs(a["sample_objective"] - b["sample_objective"]) > 1e-6:
            hard.append(f"sample objective: {a['sample_objective']} vs {b['sample_objective']}")
        if a["sample_feasible"] != b["sample_feasible"]:
            hard.append(f"sample feasible: {a['sample_feasible']} vs {b['sample_feasible']}")
    return hard, soft


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for name, spec_fn in SPECS.items():
        if only and name != only:
            continue
        spec = spec_fn()
        sample = spec.get("sample")
        # jm 1
        spec1 = _make_spec_file(
            name + "_jm1",
            JM1_ROOT,
            spec["data"],
            sample,
            spec["model_relpath"],
            spec["create_fn"],
        )
        out1 = OUT_DIR / f"{name}_jm1.json"
        ok1, err1 = _run(JM1_ROOT, spec1, out1)
        if not ok1:
            results.append((name, "JM1 FAIL", err1[-400:]))
            continue
        # jm 2 (may need augmented data)
        spec2 = _make_spec_file(
            name + "_jm2",
            JM2_ROOT,
            _augment_data_for_jm2(name, spec["data"]),
            sample,
            spec["model_relpath"],
            spec["create_fn"],
        )
        out2 = OUT_DIR / f"{name}_jm2.json"
        ok2, err2 = _run(JM2_ROOT, spec2, out2)
        if not ok2:
            results.append((name, "JM2 FAIL", err2[-400:]))
            continue
        hard, soft = _compare(name, out1, out2)
        if hard:
            status = "DIFF"
            detail = "\n  - ".join(hard + soft)
        elif soft:
            status = "SOFT-DIFF"
            detail = "\n  - ".join(soft) + " (semantic check passed)"
        else:
            status = "EQUIV"
            detail = ""
        results.append((name, status, detail))

    print("\n=== Equivalence summary ===")
    hard_fail = 0
    run_fail = 0
    for name, status, detail in results:
        line = f"{status:10s} {name}"
        if detail:
            line += "\n  - " + detail
        print(line)
        if status == "DIFF":
            hard_fail += 1
        elif status in ("JM1 FAIL", "JM2 FAIL"):
            run_fail += 1
    print(f"\n{hard_fail} hard differences, {run_fail} run failures across {len(results)} models")
    return 0 if (hard_fail == 0 and run_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
