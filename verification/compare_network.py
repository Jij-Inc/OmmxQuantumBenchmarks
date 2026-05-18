"""Compare migrated jm2 network model against qoblib_v2 artifacts.

The demand matrix for network is hardcoded in ommx_create.py, not in upstream
qoblib. We reuse the same matrix here.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yuichironakano/マイドライブ/40_Qamomile/OmmxQuantumBenchmarks")
sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Network  # noqa: E402

NET_ROOT = REPO / "ommx_quantum_benchmarks/qoblib/08_network/models/integer_lp"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODEL = _load_module("net_model", NET_ROOT / "model.py").build_ip_formulation

# Hardcoded 24x24 demand matrix from the original ommx_create.py.
# Extract by importing the function (it's defined inside batch_process), so we
# replicate it here. To stay accurate, copy the values from the source.
import ast
_src = (NET_ROOT / "ommx_create.py").read_text()
# Parse for the `your_t_0based = [...]` assignment.
_tree = ast.parse(_src)
T_BASED = None
for node in ast.walk(_tree):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "your_t_0based":
            T_BASED = ast.literal_eval(node.value)
            break
assert T_BASED is not None and len(T_BASED) == 24


def cut_matrix(t, n):
    return [row[:n] for row in t[:n]]


def compare_one(instance_name):
    n = int(instance_name.replace("network", ""))
    instance_data = {
        "n": n,
        "t": cut_matrix(T_BASED, n),
        "M": 1000,
        "intscale": 1000,
    }
    problem = MODEL()
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc)

    try:
        jm1_inst, jm1_sol = Network()("integer_lp", instance_name)
    except Exception as exc:
        return "REGISTRY_FAIL", repr(exc)

    jm1_keys = {(v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    jm2_keys = {(v.name, tuple(v.subscripts)) for v in jm2_inst.decision_variables}
    if jm1_keys != jm2_keys:
        return "VAR_MISMATCH", f"diff size {len(jm1_keys - jm2_keys)}/{len(jm2_keys - jm1_keys)}"

    jm1_cnt = Counter(c.name for c in jm1_inst.constraints)
    jm2_cnt = Counter(c.name for c in jm2_inst.constraints)
    if jm1_cnt != jm2_cnt:
        return "CONSTRAINT_COUNT_MISMATCH", f"jm1={dict(jm1_cnt)} jm2={dict(jm2_cnt)}"

    if jm1_sol is None:
        return "OK_NO_SOL", "structural ok"

    jm1_id_to_key = {v.id: (v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    jm2_key_to_id = {(v.name, tuple(v.subscripts)): v.id for v in jm2_inst.decision_variables}
    state = {}
    for k, val in jm1_sol.state.entries.items():
        key = jm1_id_to_key[k]
        state[jm2_key_to_id[key]] = val

    re_eval = jm2_inst.evaluate(state)
    if abs(re_eval.objective - jm1_sol.objective) > 1e-6:
        return "OBJ_MISMATCH", f"jm1={jm1_sol.objective} jm2={re_eval.objective}"
    if re_eval.feasible != jm1_sol.feasible:
        return "FEASIBLE_MISMATCH", f"jm1={jm1_sol.feasible} jm2={re_eval.feasible}"
    return "EQUIV", f"obj={jm1_sol.objective} feasible={jm1_sol.feasible}"


def main():
    summary = Counter()
    failures = []
    instances = Network().available_instances["integer_lp"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for i, name in enumerate(instances):
        if limit and i >= limit:
            break
        status, detail = compare_one(name)
        summary[status] += 1
        print(f"[network] {name}: {status}  {detail[:120]}")
        if status != "EQUIV":
            failures.append((name, status, detail))
    print(f"\n=== Summary ({sum(summary.values())} comparisons) ===")
    for status, n in summary.most_common():
        print(f"  {status:30s} {n}")
    if failures:
        print(f"\nNon-EQUIV (up to 5):")
        for f in failures[:5]:
            print(" ", f)


if __name__ == "__main__":
    main()
