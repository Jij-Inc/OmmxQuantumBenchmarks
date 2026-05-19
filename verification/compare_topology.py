"""Compare jm2 topology models (flow_mip, seidel_linear, seidel_quadratic)
against qoblib_v2 artifacts."""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/tmp/qoblib_upstream/10-topology/instances")
sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Topology  # noqa: E402

ROOT = REPO / "ommx_quantum_benchmarks/qoblib/10_topology/models"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Each sub-model ships its own dat_reader returning exactly the fields its
# model declares (flow_mip returns {nodes, degree}; seidel_* additionally
# return minDiameter/maxDiameter/N_arr).
READERS = {
    "flow_mip": _load("topo_flow_rd", ROOT / "flow_mip/dat_reader.py").load_topology_instance,
    "seidel_linear": _load("topo_sl_rd", ROOT / "seidel_linear/dat_reader.py").load_topology_instance,
    "seidel_quadratic": _load("topo_sq_rd", ROOT / "seidel_quadratic/dat_reader.py").load_topology_instance,
}
MODELS = {
    "flow_mip": _load("topo_flow", ROOT / "flow_mip/model.py").create_topology_model,
    "seidel_linear": _load("topo_sl", ROOT / "seidel_linear/model.py").create_topology_model,
    "seidel_quadratic": _load("topo_sq", ROOT / "seidel_quadratic/model.py").create_topology_model,
}


def build_instance_data(instance_name, model_name):
    dat_path = UPSTREAM / f"{instance_name}.dat"
    if not dat_path.exists():
        return None
    return READERS[model_name](str(dat_path))


def compare_one(model_name, instance_name):
    create_problem = MODELS[model_name]
    t0 = time.time()
    problem = create_problem()
    instance_data = build_instance_data(instance_name, model_name)
    if instance_data is None:
        return "MISSING_DAT", "", 0.0
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc), time.time() - t0
    eval_time = time.time() - t0

    try:
        jm1_inst, jm1_sol = Topology()(model_name, instance_name)
    except Exception as exc:
        return "REGISTRY_FAIL", repr(exc), eval_time

    jm1_keys = {(v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    jm2_keys = {(v.name, tuple(v.subscripts)) for v in jm2_inst.decision_variables}
    if jm1_keys != jm2_keys:
        return "VAR_MISMATCH", f"{len(jm1_keys - jm2_keys)}/{len(jm2_keys - jm1_keys)}", eval_time

    jm1_cnt = Counter(c.name for c in jm1_inst.constraints)
    jm2_cnt = Counter(c.name for c in jm2_inst.constraints)
    if jm1_cnt != jm2_cnt:
        return "CONSTRAINT_COUNT_MISMATCH", f"jm1={dict(jm1_cnt)} jm2={dict(jm2_cnt)}", eval_time

    if jm1_sol is None:
        return "OK_NO_SOL", "", eval_time

    j1_id2key = {v.id: (v.name, tuple(v.subscripts)) for v in jm1_inst.decision_variables}
    j2_key2id = {(v.name, tuple(v.subscripts)): v.id for v in jm2_inst.decision_variables}
    state = {j2_key2id[j1_id2key[k]]: v for k, v in jm1_sol.state.entries.items()}
    re_eval = jm2_inst.evaluate(state)
    if abs(re_eval.objective - jm1_sol.objective) > 1e-6:
        return "OBJ_MISMATCH", f"jm1={jm1_sol.objective} jm2={re_eval.objective}", eval_time
    if re_eval.feasible != jm1_sol.feasible:
        return "FEASIBLE_MISMATCH", f"jm1={jm1_sol.feasible} jm2={re_eval.feasible}", eval_time
    return "EQUIV", f"obj={jm1_sol.objective} feasible={jm1_sol.feasible}", eval_time


def main():
    dataset = Topology()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
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
            status, detail, eval_t = compare_one(model_name, name)
            summary[status] += 1
            print(f"[{model_name}] {name}: {status} ({eval_t:.1f}s)  {detail[:120]}", flush=True)
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
