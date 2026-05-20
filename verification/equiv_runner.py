"""Generate an OMMX instance for a given model + instance_data.

Designed to run unchanged under both JijModeling 1.x and 2.x: it auto-detects
which evaluation API is available.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import jijmodeling as jm


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("model_mod", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval(problem, data):
    """Run the appropriate evaluation API depending on jm version."""
    if hasattr(problem, "eval"):
        return problem.eval(data)
    interpreter = jm.Interpreter(data)
    return interpreter.eval_problem(problem)


def _serialize_var(var):
    return {
        "name": var.name,
        "subscripts": list(var.subscripts),
        "kind": str(var.kind),
        "lower": float(var.bound.lower) if hasattr(var, "bound") else None,
        "upper": float(var.bound.upper) if hasattr(var, "bound") else None,
    }


def _serialize_constr(c):
    return {
        "name": c.name,
        "subscripts": list(c.subscripts) if hasattr(c, "subscripts") else [],
        "equality": str(c.equality),
    }


def main():
    spec_path = sys.argv[1]
    out_path = Path(sys.argv[2])

    with open(spec_path) as f:
        spec = json.load(f)

    repo_root = Path(spec["repo_root"])
    model_path = repo_root / spec["model_relpath"]
    model_mod = _load_module(model_path)
    create = getattr(model_mod, spec["create_fn"])
    problem = create()

    # Reconstitute instance_data (numpy arrays were pickled)
    with open(spec["data_pickle"], "rb") as f:
        data = pickle.load(f)

    inst = _eval(problem, data)

    # Save OMMX bytes
    bytes_path = out_path.with_suffix(".ommx.bin")
    bytes_path.write_bytes(inst.to_bytes())

    # Save summary
    summary = {
        "name": getattr(problem, "name", None) or "?",
        "sense": str(inst.sense),
        "num_vars": len(inst.decision_variables),
        "num_constraints": len(inst.constraints),
        "vars": sorted(
            (_serialize_var(v) for v in inst.decision_variables),
            key=lambda d: (d["name"], d["subscripts"]),
        ),
        "constraints": sorted(
            (_serialize_constr(c) for c in inst.constraints),
            key=lambda d: (d["name"], d["subscripts"]),
        ),
    }

    # If a sample solution by (name, subscripts) was provided, evaluate
    sample = spec.get("sample_solution")
    if sample:
        # Map (name, subscripts) -> id from this side's instance
        var_id_by_key = {}
        for v in inst.decision_variables:
            key = (v.name, tuple(v.subscripts))
            var_id_by_key[key] = v.id
        solution_dict = {}
        for entry in sample:
            key = (entry["name"], tuple(entry["subscripts"]))
            if key not in var_id_by_key:
                raise RuntimeError(f"Variable not found: {key}")
            solution_dict[var_id_by_key[key]] = entry["value"]
        sol = inst.evaluate(solution_dict)
        summary["sample_objective"] = float(sol.objective)
        summary["sample_feasible"] = bool(sol.feasible)

    out_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
