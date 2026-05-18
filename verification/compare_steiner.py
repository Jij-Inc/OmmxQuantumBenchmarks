"""Compare jm2 steiner model against qoblib_v2 artifacts."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
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
MODEL = _load("st_model", ROOT / "model.py").create_steiner_tree_packing_model


def compare_one(instance_name):
    inst_dir = UPSTREAM / instance_name
    if not inst_dir.is_dir():
        return "MISSING_DAT", str(inst_dir), 0.0

    t0 = time.time()
    raw = READER(str(inst_dir))
    problem = MODEL()
    used = {ph.name for ph in problem.used_placeholders}
    instance_data = {k: v for k, v in raw.items() if k in used}
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc), time.time() - t0
    eval_time = time.time() - t0

    try:
        jm1_inst, jm1_sol = Steiner()("integer_linear", instance_name)
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
    instances = Steiner().available_instances["integer_linear"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = Counter()
    failures = []
    total = 0
    for i, name in enumerate(instances):
        if limit and i >= limit:
            break
        total += 1
        status, detail, eval_t = compare_one(name)
        summary[status] += 1
        print(f"[steiner] {name}: {status} ({eval_t:.1f}s)  {detail[:120]}", flush=True)
        if status not in ("EQUIV", "MISSING_DAT", "OK_NO_SOL"):
            failures.append((name, status, detail))
    print(f"\n=== Summary ({total} comparisons) ===")
    for status, n in summary.most_common():
        print(f"  {status:30s} {n}")
    if failures:
        print(f"\nNon-EQUIV (up to 10):")
        for f in failures[:10]:
            print(" ", f)


if __name__ == "__main__":
    main()
