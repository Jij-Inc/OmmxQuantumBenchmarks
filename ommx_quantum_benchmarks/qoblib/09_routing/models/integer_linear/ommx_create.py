import os
import glob
import jijmodeling as jm
from ommx.artifact import ArtifactBuilder
from model import build_vrp_ilp
from dat_reader import read_vrp_tsplib
from sol_reader import parse_vrp_solution_file
from ommx_quantum_benchmarks.qoblib.definitions import (
    QOBLIB_AUTHORS,
    QOBLIB_AUTHORS_STR,
    LICENSE,
)


def _pick_solution_file(sol_dir: str, base: str) -> str | None:
    """Pick first existing solution file for a basename."""
    candidates = [
        os.path.join(sol_dir, f"{base}.opt.sol"),
        os.path.join(sol_dir, f"{base}.best.sol"),
        os.path.join(sol_dir, f"{base}.bst.sol"),
        os.path.join(sol_dir, f"{base}.sol"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def batch_process(
    inst_dir: str = "../../instances",
    sol_root: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    """
    Process instances from a QBench JSON file and corresponding solution files,
    convert them into OMMX artifacts, and save them to the output directory.

    Parameters:
        inst_dir (str): Path to the gph..
        sol_root (str): Path to the root solutions directory.
        output_directory (str): Path to save the generated .ommx files.
    """
    os.makedirs(output_directory, exist_ok=True)

    problem = build_vrp_ilp()

    vrp_paths = sorted(glob.glob(os.path.join(inst_dir, "*.vrp")))
    if not vrp_paths:
        print(f"[MIS] No .vrp files found under: {inst_dir}")
        return

    processed_count = 0
    error_count = 0

    for vrp_path in vrp_paths:
        base = os.path.splitext(os.path.basename(vrp_path))[0]
        try:
            print(f"[{base}] Reading: {vrp_path}")
            instance_data = read_vrp_tsplib(vrp_path, vehicle_limit=4, euc2d_round=True)

            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)

            sol_path = _pick_solution_file(sol_root, base)
            solution = None
            if sol_path:
                try:
                    print(f"  → Evaluating solution: {sol_path}")
                    objective_value, solution_dict = parse_vrp_solution_file(
                        sol_path,
                        depot=0,
                        demand=instance_data["DEMAND"],
                        n=instance_data["n"],
                    )
                    solution = ommx_instance.evaluate(solution_dict)
                    if (
                        solution.feasible
                        and abs(solution.objective - objective_value) < 1e-6
                    ):
                        print(
                            f"    objective={solution.objective}, feasible={solution.feasible}"
                        )
                    else:
                        print(
                            "    ! Objective mismatch or infeasible; will save instance only."
                        )
                        solution = None
                except Exception as sol_err:
                    print(f"    ! Solution evaluation failed: {sol_err}")
                    solution = None

            out_path = os.path.join(output_directory, f"{base}.ommx")
            if os.path.exists(out_path):
                os.remove(out_path)

            # Add annotations to the instance.
            ommx_instance.title = base
            ommx_instance.license = LICENSE
            ommx_instance.dataset = "Vehicle Routing"
            ommx_instance.authors = QOBLIB_AUTHORS
            ommx_instance.num_variables = len(ommx_instance.decision_variables)
            ommx_instance.num_constraints = len(ommx_instance.constraints)
            ommx_instance.annotations["org.ommx.qoblib.url"] = (
                "https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/09-routing?ref_type=heads"
            )

            builder = ArtifactBuilder.new_archive_unnamed(out_path)
            instance_desc = builder.add_instance(ommx_instance)
            if solution is not None:
                solution.instance = instance_desc.digest
                solution.annotations["org.ommx.qoblib.authors"] = QOBLIB_AUTHORS_STR
                builder.add_solution(solution)
            builder.build()

            print(f"  ✓ Created: {out_path}")
            print("-" * 50)
            processed_count += 1

        except Exception as e:
            print(f"[{base}] Error: {e}")
            print("-" * 50)
            error_count += 1

    print(f"Batch complete — processed: {processed_count}, errors: {error_count}")


if __name__ == "__main__":
    batch_process(
        inst_dir="../../instances",
        sol_root="../../solutions",
        output_directory="./ommx_output",
    )
