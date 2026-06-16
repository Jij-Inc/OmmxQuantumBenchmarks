import os
import glob
import jijmodeling as jm
from ommx.artifact import ArtifactBuilder
from model import build_drr_model_exact
from dat_reader import parse_instance_xml
from sol_reader import parse_solution_as_dict


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

    problem = build_drr_model_exact()

    xml_paths = sorted(glob.glob(os.path.join(inst_dir, "*.xml")))
    if not xml_paths:
        print(f"[MIS] No .xml files found under: {inst_dir}")
        return

    processed_count = 0
    error_count = 0

    for xml_path in xml_paths:
        base = os.path.splitext(os.path.basename(xml_path))[0]
        try:
            print(f"[{base}] Reading: {xml_path}")
            instance_data = parse_instance_xml(xml_path)

            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)

            sol_path = _pick_solution_file(sol_root, base)
            solution = None
            if sol_path:
                try:
                    print(f"  → Evaluating solution: {sol_path}")
                    objective_value, solution_dict = parse_solution_as_dict(
                        sol_path, n=instance_data["T"], n_slots=instance_data["S"]
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

            builder = ArtifactBuilder.new_archive_unnamed(out_path)
            builder.add_instance(ommx_instance)
            if solution is not None:
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
        inst_dir="../../instances/Small",
        sol_root="../../solutions/Small",
        output_directory="./ommx_output",
    )

    batch_process(
        inst_dir="../../instances/Medium",
        sol_root="../../solutions/Medium",
        output_directory="./ommx_output",
    )
