import os
import glob
import math
from pathlib import Path
import jijmodeling as jm
from ommx.artifact import ArtifactBuilder
from sol_reader import parse_sol_to_ordered_dict
from dat_reader import read_qoblib_dat_file
from model import create_problem


def batch_process_files(
    dat_directory: str = "../../instances",
    sol_directory: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    """
    Batch process .dat and .sol files from different directories and convert them into .ommx files.

    Parameters:
    - dat_directory: Path to the directory containing .dat files
    - sol_directory: Path to the directory containing .sol files
    - output_directory: Path to the directory where .ommx files will be saved
    """

    # Create output directory (if it does not exist)
    os.makedirs(output_directory, exist_ok=True)

    # Create the problem definition (shared by all instances)
    problem = create_problem()

    # Find all .dat files
    dat_files = glob.glob(os.path.join(dat_directory, "*.dat"))

    if not dat_files:
        print(f"No .dat files found in directory {dat_directory}")
        return

    print(f"Found {len(dat_files)} .dat files in {dat_directory}")
    print(f"Solution files directory: {sol_directory}")
    print(f"Output directory: {output_directory}")
    print("-" * 50)

    processed_count = 0
    error_count = 0

    for dat_file in dat_files:
        try:
            # Get the base filename (without extension)
            base_name = Path(dat_file).stem

            # Construct possible .sol file paths in the solutions directory
            possible_sol_files = [
                os.path.join(sol_directory, f"{base_name}.opt.sol"),
                os.path.join(sol_directory, f"{base_name}.sol"),
            ]

            sol_file = None
            for possible_sol in possible_sol_files:
                if os.path.exists(possible_sol):
                    sol_file = possible_sol
                    break

            if sol_file is None:
                print(
                    "Warning: Corresponding solution file not found. Tried the following:"
                )
                for possible_sol in possible_sol_files:
                    print(f"  - {possible_sol}")
                continue

            print(f"Processing file pair: {dat_file} and {sol_file}")

            # Read the .dat file
            instance_data = read_qoblib_dat_file(dat_file) 

            # Create an OMMX instance
            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)

            # Read and evaluate the solution
            solution = None
            try:
                energy_dict, solution_dict = parse_sol_to_ordered_dict(
                    sol_file, len(instance_data["I"])
                )
                solution = ommx_instance.evaluate(solution_dict)

                if (
                    math.isclose(energy_dict["Energy"], solution.objective, rel_tol=1e-6)
                    and solution.feasible
                ):
                    print(f"  → Calculate objective={solution.objective:.6f}, "f"Solution objective={energy_dict['Energy']:.6f}")
                    print(f"  → objective={solution.objective:.6f}, feasible={solution.feasible}")
                else:
                    diff = solution.objective - energy_dict["Energy"]
                    print("Objective or feasible Error")
                    print(f"  ✗ mismatch: calc={solution.objective:.6f}, sol={energy_dict['Energy']:.6f}, "
                        f"Δ={diff:.6g}, feasible={solution.feasible}")

            except Exception as sol_error:
                print(f"  ! Error evaluating solution: {sol_error}")
                print("    Skipping solution evaluation and only saving the instance...")


            # Construct the output filename
            output_filename = os.path.join(output_directory, f"{base_name}.ommx")

            # If the file already exists, remove it
            if os.path.exists(output_filename):
                os.remove(output_filename)

            # Create OMMX Artifact
            builder = ArtifactBuilder.new_archive_unnamed(output_filename)
            builder.add_instance(ommx_instance)
            if solution is not None:
                builder.add_solution(solution)
            builder.build()

            print(f"Successfully created: {output_filename}")
            print("-" * 50)
            processed_count += 1

        except Exception as e:
            print(f"Error processing file {dat_file}: {str(e)}")
            error_count += 1
            continue

    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Number of errors: {error_count} files")
    print(f"OMMX files saved in: {output_directory}")
    print("-" * 50)


if __name__ == "__main__":
    batch_process_files()
