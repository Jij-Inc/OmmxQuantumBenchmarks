"""Compare jm2 birkhoff model against qoblib_v2 artifacts.

For each declared instance `bh{D|S}-{n}-{idx}`:
- Map to upstream JSON file `qbench_<NN>_<dense|sparse>.json` (with the n
  expressed as 2 digits, e.g. n=3 -> "03").
- Look up the entry by the zero-stripped index string (JSON object keys
  are strings); skip if not present in upstream — upstream only ships 10
  entries per (n, kind), but `available_instances` lists 100.
- Build instance_data via the existing dat_reader.process_entry (requires
  p{n}.dat files; we cd into the model directory).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/tmp/qoblib_upstream/03-birkhoff/instances")
sys.path.insert(0, str(REPO))

from ommx_quantum_benchmarks.qoblib.qoblib import Birkhoff  # noqa: E402

MODEL_DIR = REPO / "ommx_quantum_benchmarks/qoblib/03_birkhoff/models/integer_linear"


@contextlib.contextmanager
def _in_model_dir():
    """dat_reader.process_entry resolves `p{n}.dat` via a relative path, so
    we briefly chdir into the model directory and restore the previous CWD."""
    prev = os.getcwd()
    os.chdir(MODEL_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("birk_reader", MODEL_DIR / "dat_reader.py")
MODEL = _load("birk_model", MODEL_DIR / "models.py").create_problem


def parse_instance_name(instance_name):
    """bhD-3-001 -> ('D', 3, '1')"""
    parts = instance_name.split("-")
    if len(parts) != 3:
        return None
    kind = parts[0].replace("bh", "")  # 'D' or 'S'
    n = int(parts[1])
    idx = parts[2].lstrip("0") or "0"
    return kind, n, idx


def build_instance_data(instance_name):
    parsed = parse_instance_name(instance_name)
    if parsed is None:
        return None
    kind, n, key = parsed
    kind_str = "dense" if kind == "D" else "sparse"
    json_path = UPSTREAM / f"qbench_{n:02d}_{kind_str}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    if key not in data:
        return None
    with _in_model_dir():
        return READER.process_entry(data[key])


def compare_one(instance_name):
    instance_data = build_instance_data(instance_name)
    if instance_data is None:
        return "MISSING_DAT", ""

    problem = MODEL()
    try:
        jm2_inst = problem.eval(instance_data)
    except Exception as exc:
        return "JM2_EVAL_FAIL", repr(exc)

    try:
        jm1_inst, jm1_sol = Birkhoff()("integer_linear", instance_name)
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
    dataset = Birkhoff()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    only_kind = sys.argv[2] if len(sys.argv) > 2 else None
    summary = Counter()
    failures = []
    total = 0
    for name in dataset.available_instances["integer_linear"]:
        if only_kind and not name.startswith(only_kind):
            continue
        if limit and total >= limit:
            break
        total += 1
        status, detail = compare_one(name)
        summary[status] += 1
        print(f"[birkhoff] {name}: {status}  {detail[:100]}")
        if status not in ("EQUIV", "MISSING_DAT", "OK_NO_SOL"):
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
