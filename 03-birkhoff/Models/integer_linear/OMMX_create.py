import os
import json
import re
import jijmodeling as jm
import math
from ommx.artifact import ArtifactBuilder
from models import create_problem
from dat_reader import load_and_process_all
from sol_reader import parse_sol_file


def infer_name_func_from_subdir(sol_subdir: str):
    """
    Infer the base name generation function from the solution subdirectory name.

    The subdir should follow a pattern like:
        '03_dense'  -> bhD-3-001
        '04_dense'  -> bhD-4-001
        '03_sparse' -> bhS-3-001
        '07_sparse' -> bhS-7-001

    Returns:
        A function name_func(key: str) -> formatted base name string.
    """
    # Extract number and type from subdir name
    m = re.match(r"^\s*(\d{2}|\d+)\s*_(dense|sparse)\s*$", sol_subdir, re.IGNORECASE)
    if not m:
        # Fallback: try to find a number and a type (dense/sparse)
        num_match = re.search(r"(\d{1,2})", sol_subdir)
        kind_match = re.search(r"(dense|sparse)", sol_subdir, re.IGNORECASE)
        if not (num_match and kind_match):
            raise ValueError(f"Cannot infer naming pattern from subdir: {sol_subdir}")
        k = int(num_match.group(1))
        kind = kind_match.group(1).lower()
    else:
        k = int(m.group(1))
        kind = m.group(2).lower()

    letter = "D" if kind == "dense" else "S"

    def name_func(key: str) -> str:
        # Convert key to integer and format with leading zeros
        return f"bh{letter}-{k}-{int(key):03d}"

    return name_func


def batch_process_from_qbench_json(
    json_path: str,
    sol_subdir: str,
    sol_root: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    """
    Process instances from a QBench JSON file and corresponding solution files,
    convert them into OMMX artifacts, and save them to the output directory.

    Parameters:
        json_path (str): Path to the QBench JSON file.
        sol_subdir (str): Name of the solution subdirectory, e.g., "03_dense".
        sol_root (str): Path to the root solutions directory.
        output_directory (str): Path to save the generated .ommx files.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Load QBench JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    processed_data = load_and_process_all(raw_json)
    problem = create_problem()

    # Infer base name pattern from subdir
    name_func = infer_name_func_from_subdir(sol_subdir)

    processed_count = 0
    error_count = 0

    for key, instance_data in processed_data.items():
        # Example key: "001"
        n = instance_data["msize"]

        # Automatically match the correct file naming pattern
        base_name = name_func(key)

        # Locate the corresponding solution file
        sol_dir = os.path.join(sol_root, sol_subdir)
        possible_sol_files = [
            os.path.join(sol_dir, f"{base_name}.opt.sol"),
            os.path.join(sol_dir, f"{base_name}.bst.sol"),
            os.path.join(sol_dir, f"{base_name}.sol"),
        ]
        sol_file = next((p for p in possible_sol_files if os.path.exists(p)), None)
        if sol_file is None:
            print(f"Warning: {base_name} solution not found in {sol_subdir}")
            continue

        try:
            print(f"[{base_name}] Processing solution: {sol_file}")

            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)

            solution = None
            try:
                # Using the updated parse_sol_file (z1..zn, x1..xn)
                energy_dict, entries_dict, solution_dict = parse_sol_file(
                    sol_file, math.factorial(n)
                )
                solution = ommx_instance.evaluate(solution_dict)
                if energy_dict.get("Energy") == solution.objective and solution.feasible:
                    print(f"  → objective={solution.objective}, feasible={solution.feasible}")
                else:
                    print("  ! Objective or feasible mismatch")
            except Exception as sol_error:
                print(f"  ! Error evaluating solution: {sol_error}")
                print("    Skipping solution evaluation and only saving the instance...")

            output_filename = os.path.join(output_directory, f"{base_name}.ommx")
            if os.path.exists(output_filename):
                os.remove(output_filename)

            builder = ArtifactBuilder.new_archive_unnamed(output_filename)
            builder.add_instance(ommx_instance)
            if solution is not None:
                builder.add_solution(solution)
            builder.build()

            print(f"  ✓ Created: {output_filename}")
            print("-" * 50)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            print("-" * 50)
            error_count += 1

    print(f"Batch complete — processed: {processed_count}, errors: {error_count}")


if __name__ == "__main__":
    # Example: process 03_dense
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_03_dense.json",
        sol_subdir="03_dense",  # → bhD-3-***
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )

    # To process 03_sparse (→ bhS-3-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_03_sparse.json",
        sol_subdir="03_sparse",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )

    # To process 04_dense (→ bhD-4-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_04_dense.json",
        sol_subdir="04_dense",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )

    # To process 04_sparse (→ bhS-4-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_04_sparse.json",
        sol_subdir="04_sparse",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )

    # To process 05_dense (→ bhD-4-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_05_dense.json",
        sol_subdir="05_dense",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )

    # To process 05_sparse (→ bhD-4-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_05_sparse.json",
        sol_subdir="05_sparse",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )
    # To process 06_dense (→ bhD-4-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_06_dense.json",
        sol_subdir="06_dense",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )

    # To process 06_sparse (→ bhS-6-***), uncomment:
    batch_process_from_qbench_json(
        json_path="../../solutions/qbench_06_sparse.json",
        sol_subdir="06_sparse",
        sol_root="../../solutions",
        output_directory="./ommx_output"
    )
