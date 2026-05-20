"""Compare jm2 routing model against qoblib_v2 artifacts."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/tmp/qoblib_upstream/09-routing/instances")
sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Routing  # noqa: E402

ROOT = REPO / "ommx_quantum_benchmarks/qoblib/09_routing/models/integer_linear"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("vrp_reader", ROOT / "dat_reader.py").read_vrp_tsplib
MODEL = _load("vrp_model", ROOT / "model.py").build_vrp_ilp


def compare_one(instance_name):
    vrp_path = UPSTREAM / f"{instance_name}.vrp"
    if not vrp_path.exists():
        return "MISSING_DAT", str(vrp_path)
    instance_data = READER(str(vrp_path), vehicle_limit=4, euc2d_round=True)

    problem = MODEL()
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc)

    try:
        jm1_inst, jm1_sol = Routing()("integer_linear", instance_name)
    except Exception as exc:
        return "REGISTRY_FAIL", repr(exc)

    jm1_keys = {(v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    jm2_keys = {(v.name, tuple(v.subscripts)) for v in jm2_inst.decision_variables}
    if jm1_keys != jm2_keys:
        return "VAR_MISMATCH", f"{len(jm1_keys - jm2_keys)}/{len(jm2_keys - jm1_keys)}"

    jm1_cnt = Counter(c.name for c in jm1_inst.constraints)
    jm2_cnt = Counter(c.name for c in jm2_inst.constraints)
    if jm1_cnt != jm2_cnt:
        return "CONSTRAINT_COUNT_MISMATCH", f"jm1={dict(jm1_cnt)} jm2={dict(jm2_cnt)}"

    if jm1_sol is None:
        return "OK_NO_SOL", ""

    j1_id2key = {v.id: (v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    j2_key2id = {(v.name, tuple(v.subscripts)): v.id for v in jm2_inst.decision_variables}
    state = {j2_key2id[j1_id2key[k]]: v for k, v in jm1_sol.state.entries.items()}
    re_eval = jm2_inst.evaluate(state)
    if abs(re_eval.objective - jm1_sol.objective) > 1e-6:
        return "OBJ_MISMATCH", f"jm1={jm1_sol.objective} jm2={re_eval.objective}"
    if re_eval.feasible != jm1_sol.feasible:
        return "FEASIBLE_MISMATCH", f"jm1={jm1_sol.feasible} jm2={re_eval.feasible}"
    return "EQUIV", f"obj={jm1_sol.objective} feasible={jm1_sol.feasible}"


def main():
    instances = Routing().available_instances["integer_linear"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = Counter()
    failures = []
    total = 0
    for i, name in enumerate(instances):
        if limit and i >= limit:
            break
        total += 1
        status, detail = compare_one(name)
        summary[status] += 1
        print(f"[routing] {name}: {status}  {detail[:120]}")
        if status != "EQUIV":
            failures.append((name, status, detail))
    print(f"\n=== Summary ({total} comparisons) ===")
    for status, n in summary.most_common():
        print(f"  {status:30s} {n}")
    if failures:
        print(f"\nNon-EQUIV (up to 10):")
        for f in failures[:10]:
            print(" ", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
