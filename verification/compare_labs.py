"""Compare jm2 labs models against qoblib_v2 artifacts.

Labs instance data is purely synthetic: `instance_data = {"I": arange(n),
"K": arange(n - 1)}` (plus "P": 10000 for the QUBO variant). The instance
name `labsNNN` encodes n directly, so no upstream data files are required.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Labs  # noqa: E402

ROOT = REPO / "ommx_quantum_benchmarks/qoblib/02_labs/models"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODEL_INT = _load("labs_int", ROOT / "integer/model.py").create_problem
MODEL_QU = _load("labs_qu", ROOT / "quadratic_unconstrained/model.py").create_problem


def build_instance_data(model_name, instance_name):
    n = int(instance_name.replace("labs", ""))
    data = {"I": np.arange(n), "K": np.arange(n - 1)}
    if model_name == "quadratic_unconstrained":
        data["P"] = 10000.0
    return data


def compare_one(model_name, instance_name, create_problem):
    instance_data = build_instance_data(model_name, instance_name)
    problem = create_problem()
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc)

    try:
        jm1_inst, jm1_sol = Labs()(model_name, instance_name)
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
    dataset = Labs()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    model_fns = {"integer": MODEL_INT, "quadratic_unconstrained": MODEL_QU}
    summary = Counter()
    failures = []
    total = 0
    for model_name, instances in dataset.available_instances.items():
        if only and model_name != only:
            continue
        for i, name in enumerate(instances):
            if limit and i >= limit:
                break
            total += 1
            status, detail = compare_one(model_name, name, model_fns[model_name])
            summary[status] += 1
            print(f"[{model_name}] {name}: {status}  {detail[:120]}")
            if status != "EQUIV":
                failures.append((model_name, name, status, detail))
    print(f"\n=== Summary ({total} comparisons) ===")
    for status, n in summary.most_common():
        print(f"  {status:30s} {n}")
    if failures:
        print(f"\nNon-EQUIV (up to 10):")
        for f in failures[:10]:
            print(" ", f)


if __name__ == "__main__":
    main()
