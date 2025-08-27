import os
import jijmodeling as jm
import numpy as np
import math
import glob
from ommx.artifact import ArtifactBuilder
from sol_reader import parse_sol_file
from model import create_problem
from solve_c_from_x import solve_c


def create_instance(n):
    instance_data = {
        "I": np.arange(n),  # [0, 1, 2, ..., n-1]
        "K": np.arange(n - 1),  # [0, 1, 2, ..., n-2]
    }
    return instance_data


def batch_process_files(
    sol_directory: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    """
    Batch process solution files, convert them to OMMX instances,
    and optionally attach evaluated solutions.

    This function iterates over problem instances `labs002` to `labs100`,
    searches for corresponding `.sol` files in the provided solutions directory,
    evaluates each solution against a shared problem definition, and saves
    the result as an `.ommx` file in the output directory.

    Args:
        sol_directory (str, optional):
            Path to the directory containing `.sol` solution files.
            Defaults to `"../../solutions"`.
        output_directory (str, optional):
            Path to the directory where `.ommx` files will be saved.
            Defaults to `"./ommx_output"`.

    Raises:
        Exception:
            If there is an unexpected error during problem creation,
            solution parsing, or file writing.
    """

    # Create the output directory if it does not already exist
    os.makedirs(output_directory, exist_ok=True)

    # Build the problem definition (shared by all instances)
    problem = create_problem()

    processed_count = 0
    error_count = 0

    # Generate instance_data directly for n from 2 to 100
    for n in range(2, 101):
        try:
            # Format base_name as labs002, labs003, … labs100
            base_name = f"labs{n:03d}"

            # Look for labsXXX.opt.sol or labsXXX.sol in the solutions directory
            sol_files = glob.glob(os.path.join(sol_directory, f"{base_name}*.sol"))
            if not sol_files:
                print(
                    f"Warning: Corresponding solution file for {base_name} not found."
                )
                continue
            sol_file = sol_files[0]
            print(f"Processing solution file: {sol_file}")

            # Generate instance_data using create_instance(n)
            instance_data = create_instance(n)

            # Create an OMMX instance
            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)

            # Read and evaluate the solution
            solution = None
            try:
                energy_dict, entries_dict, solution_dict = parse_sol_file(sol_file, n)

                for i in range(0, n - 1):
                    solution_dict[i] = solve_c(
                        ommx_instance.constraints[i], solution_dict, target_var_id=i
                    )
                solution = ommx_instance.evaluate(solution_dict)
                if (
                    math.isclose(
                        energy_dict["Energy"], solution.objective, rel_tol=1e-6
                    )
                    and solution.feasible
                ):
                    print(
                        f"  → objective={solution.objective}, feasible={solution.feasible}"
                    )
                else:
                    print("Objective or feasible Error")
            except Exception as sol_error:
                print(f"  ! Error evaluating solution: {sol_error}")
                print(
                    "    Skipping solution evaluation and only saving the instance..."
                )

            # Write out the .ommx artifact
            output_filename = os.path.join(output_directory, f"{base_name}.ommx")
            if os.path.exists(output_filename):
                os.remove(output_filename)

            # Create QOBLIB authors.
            qoblib_authors = ", ".join(
                [
                    "Thorsten Koch",
                    "David E. Bernal Neira",
                    "Ying Chen",
                    "Giorgio Cortiana",
                    "Daniel J. Egger",
                    "Raoul Heese",
                    "Narendra N. Hegade",
                    "Alejandro Gomez Cadavid",
                    "Rhea Huang",
                    "Toshinari Itoko",
                    "Thomas Kleinert",
                    "Pedro Maciel Xavier",
                    "Naeimeh Mohseni",
                    "Jhon A. Montanez-Barrera",
                    "Koji Nakano",
                    "Giacomo Nannicini",
                    "Corey O’Meara",
                    "Justin Pauckert",
                    "Manuel Proissl",
                    "Anurag Ramesh",
                    "Maximilian Schicker",
                    "Noriaki Shimada",
                    "Mitsuharu Takeori",
                    "Victor Valls",
                    "David Van Bulck",
                    "Stefan Woerner",
                    "Christa Zoufal",
                ]
            )
            # Add annotations to the instance.
            ommx_instance.title = base_name
            ommx_instance.license = "CC BY 4.0"
            ommx_instance.dataset = "Low Autocorrelation Binary Sequences (LABS)"
            ommx_instance.annotations["org.ommx.qoblib.authors"] = qoblib_authors
            ommx_instance.authors = ["Jij-Inc."]
            ommx_instance.num_variables = len(ommx_instance.decision_variables)
            ommx_instance.num_constraints = len(ommx_instance.constraints)
            # Build and save the artifact.
            builder = ArtifactBuilder.new_archive_unnamed(output_filename)
            instance_desc = builder.add_instance(ommx_instance)
            # Add solution if available.
            if solution is not None:
                solution.instance = instance_desc.digest
                solution.annotations["org.ommx.qoblib.authors"] = qoblib_authors
                solution.annotations["org.ommx.v1.solution.authors"] = "Jij-Inc."
                builder.add_solution(solution)
            builder.build()

            print(f"  ✓ Successfully created: {output_filename}")
            print("-" * 50)
            processed_count += 1

        except Exception as e:
            print(f"Error processing labs n={n}: {e}")
            error_count += 1

    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Number of errors: {error_count} files")
    print(f"OMMX files saved in: {output_directory}")
    print("-" * 50)


if __name__ == "__main__":
    batch_process_files()
