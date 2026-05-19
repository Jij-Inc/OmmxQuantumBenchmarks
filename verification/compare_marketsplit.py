"""Compare migrated jm2 marketsplit models against deployed qoblib_v2 artifacts.

For each (model_name, instance_name) in Marketsplit.available_instances:
1. Read the original .dat file from upstream qoblib.
2. Build the jm2 Problem and evaluate it -> jm2_inst.
3. Pull the existing qoblib_v2 artifact -> (jm1_inst, jm1_sol).
4. Compare:
   - Structural: same variable set keyed by (name, subscripts), same constraint
     count per name.
   - Functional: re-map the reference solution from jm1's variable IDs to jm2's
     variable IDs (via (name, subscripts)), evaluate against jm2_inst, and
     compare objective + feasibility to the stored jm1 solution.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/tmp/qoblib_upstream/01-marketsplit")

sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Marketsplit  # noqa: E402

MS_ROOT = REPO / "ommx_quantum_benchmarks/qoblib/01_marketsplit/models"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Both binary_linear and binary_unconstrained share read_qoblib_dat_file.
DAT_READER = _load_module(
    "dat_reader_bl", MS_ROOT / "binary_linear/dat_reader.py"
).read_qoblib_dat_file

MODEL_BL = _load_module("model_bl", MS_ROOT / "binary_linear/model.py").create_problem
MODEL_BU = _load_module(
    "model_bu", MS_ROOT / "binary_unconstrained/model.py"
).create_problem


def _build_var_key_to_id(inst):
    """Return dict (name, tuple(subscripts)) -> variable id."""
    out = {}
    for v in inst.decision_variables:
        out[(v.name, tuple(v.subscripts))] = v.id
    return out


def compare_one(model_name: str, instance_name: str, create_problem):
    """Return (status, detail)."""
    dat_path = UPSTREAM / "instances" / f"{instance_name}.dat"
    if not dat_path.exists():
        return "MISSING_DAT", str(dat_path)

    instance_data = DAT_READER(str(dat_path))

    problem = create_problem()
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc)

    dataset = Marketsplit()
    try:
        jm1_inst, jm1_sol = dataset(model_name, instance_name)
    except Exception as exc:
        return "REGISTRY_FAIL", repr(exc)

    # Structural diff
    jm1_var_keys = {(v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    jm2_var_keys = {(v.name, tuple(v.subscripts)) for v in jm2_inst.decision_variables}
    if jm1_var_keys != jm2_var_keys:
        return (
            "VAR_MISMATCH",
            f"jm1\\jm2={sorted(jm1_var_keys - jm2_var_keys)[:3]}..."
            f" jm2\\jm1={sorted(jm2_var_keys - jm1_var_keys)[:3]}...",
        )

    jm1_c = Counter(c.name for c in jm1_inst.constraints)
    jm2_c = Counter(c.name for c in jm2_inst.constraints)
    if jm1_c != jm2_c:
        return "CONSTRAINT_COUNT_MISMATCH", f"jm1={dict(jm1_c)} jm2={dict(jm2_c)}"

    if jm1_sol is None:
        return "OK_NO_SOL", "structural ok; no solution to functional-check"

    # Functional check: re-map solution by (name, subscripts)
    jm1_id_to_key = {v.id: (v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    jm2_key_to_id = _build_var_key_to_id(jm2_inst)

    state_dict = {}
    for jm1_id, val in jm1_sol.state.entries.items():
        key = jm1_id_to_key.get(jm1_id)
        if key is None:
            return "UNKNOWN_VAR_IN_SOL", f"jm1 var id {jm1_id} not found"
        jm2_id = jm2_key_to_id.get(key)
        if jm2_id is None:
            return "MISSING_VAR_IN_JM2", f"key={key} not in jm2 instance"
        state_dict[jm2_id] = val

    try:
        re_eval = jm2_inst.evaluate(state_dict)
    except Exception as exc:
        return "EVAL_FAIL", repr(exc)

    if abs(re_eval.objective - jm1_sol.objective) > 1e-6:
        return (
            "OBJ_MISMATCH",
            f"jm1={jm1_sol.objective} jm2={re_eval.objective}",
        )
    if re_eval.feasible != jm1_sol.feasible:
        return (
            "FEASIBLE_MISMATCH",
            f"jm1={jm1_sol.feasible} jm2={re_eval.feasible}",
        )

    return "EQUIV", f"obj={jm1_sol.objective} feasible={jm1_sol.feasible}"


def main():
    dataset = Marketsplit()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    model_to_fn = {
        "binary_linear": MODEL_BL,
        "binary_unconstrained": MODEL_BU,
    }

    summary: Counter[str] = Counter()
    failures: list[tuple[str, str, str, str]] = []
    total = 0

    for model_name, instances in dataset.available_instances.items():
        if only and model_name != only:
            continue
        for i, instance_name in enumerate(instances):
            if limit is not None and i >= limit:
                break
            total += 1
            status, detail = compare_one(model_name, instance_name, model_to_fn[model_name])
            summary[status] += 1
            print(f"[{model_name}] {instance_name}: {status}  {detail[:100]}")
            if status != "EQUIV":
                failures.append((model_name, instance_name, status, detail))

    print()
    print(f"=== Summary ({total} comparisons) ===")
    for status, n in summary.most_common():
        print(f"  {status:30s} {n}")
    if failures:
        print(f"\nNon-EQUIV examples (up to 10):")
        for entry in failures[:10]:
            print(" ", entry)
    return 0 if summary["EQUIV"] == total else 1


if __name__ == "__main__":
    sys.exit(main())
