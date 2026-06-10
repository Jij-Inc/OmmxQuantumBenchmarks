import glob
import io
import math
import os
import re
import tarfile
from fractions import Fraction

from ommx.artifact import ArtifactBuilder
from sol_reader import parse_sol_file
from dat_reader import B_BY_ASSETS, read_portfolio_instance
from model import create_problem
from ommx_quantum_benchmarks.qoblib.definitions import (
    QOBLIB_AUTHORS,
    QOBLIB_AUTHORS_STR,
    LICENSE,
)

# Lambda (risk weight) values used by gen_archive.sh of the original QOBLIB
# repository, as (instance-name suffix, solution-file suffix).  The instance
# name suffix doubles as the exact decimal value of lambda.
LAMBDA_VALUES = [
    ("0", "0.0"),
    ("0.000001", "1e-06"),
    ("0.00001", "1e-05"),
    ("0.00005", "5e-05"),
    ("0.0001", "0.0001"),
    ("0.0005", "0.0005"),
    ("0.001", "0.001"),
    ("0.01", "0.01"),
]

# Same directory pattern as gen_archive.sh ("po_a0*"): the original QOBLIB
# repository provides BQP models and solutions only for the 10- and 50-asset
# instances.
_INSTANCE_DIR_RE = re.compile(r"^po_a(0\d{2})_t(\d{2})_(s\d{2}|orig)$")


def read_solution_lines(sol_directory: str, subdir: str, sol_name: str):
    """Read the lines of a solution file from solutions/bqp.

    Supports both an extracted directory (`bqp/<subdir>/<sol_name>`) and the
    distributed archive (`bqp/<subdir>.tar.gz`).  Returns None if not found.
    """
    sol_path = os.path.join(sol_directory, "bqp", subdir, sol_name)
    if os.path.exists(sol_path):
        with open(sol_path, "r", encoding="utf-8") as f:
            return f.readlines()

    tar_path = os.path.join(sol_directory, "bqp", f"{subdir}.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            try:
                member = tar.extractfile(f"{subdir}/{sol_name}")
            except KeyError:
                return None
            if member is None:
                return None
            return io.TextIOWrapper(member, encoding="utf-8").readlines()

    return None


def batch_process_files(
    dat_directory: str = "../../instances",
    sol_directory: str = "../../solutions",
    output_directory: str = "./ommx_output",
):
    """
    Batch process QOBLIB portfolio instance directories and the corresponding
    solution files, and convert them into .ommx files.

    Each instance directory `po_aXXX_tXX_{sXX|orig}` is combined with every
    lambda value of LAMBDA_VALUES, producing instances named like the original
    LP files, e.g. `bqp_a010_t10_orig_b004_l0.001`.

    Parameters:
    - dat_directory: Path to the directory containing the instance directories
    - sol_directory: Path to the directory containing the solutions (bqp/...)
    - output_directory: Path to the directory where .ommx files will be saved
    """

    # Create output directory (if it does not exist)
    os.makedirs(output_directory, exist_ok=True)

    # Create the problem definition (shared by all instances)
    problem = create_problem()

    instance_dirs = sorted(glob.glob(os.path.join(dat_directory, "po_*")))
    if not instance_dirs:
        print(f"No po_* instance directories found in {dat_directory}")
        return

    print(f"Found {len(instance_dirs)} instance directories in {dat_directory}")
    print(f"Solution files directory: {sol_directory}")
    print(f"Output directory: {output_directory}")
    print("-" * 50)

    processed_count = 0
    error_count = 0

    for instance_dir in instance_dirs:
        dir_name = os.path.basename(instance_dir)
        match = _INSTANCE_DIR_RE.match(dir_name)
        if match is None:
            print(f"Skipping {dir_name}: no BQP model for this instance upstream.")
            continue
        num_assets = int(match.group(1))
        num_periods = int(match.group(2))
        seed = match.group(3)
        b_total = B_BY_ASSETS[num_assets]
        subdir = f"a{num_assets:03d}_t{num_periods:02d}_{seed}_b{b_total:03d}"

        for lam_suffix, sol_suffix in LAMBDA_VALUES:
            lam = Fraction(lam_suffix)
            base_name = f"bqp_{subdir}_l{lam_suffix}"
            try:
                sol_name = f"{subdir}_l{sol_suffix}.sol"
                sol_lines = read_solution_lines(sol_directory, subdir, sol_name)
                if sol_lines is None:
                    print(
                        f"Warning: Solution file {sol_name} not found in "
                        f"{os.path.join(sol_directory, 'bqp')}. Skipping {base_name}."
                    )
                    continue

                print(f"Processing {base_name}")

                # Read the instance data
                instance_data, symbols = read_portfolio_instance(
                    instance_dir, lam, num_assets
                )

                # Create an OMMX instance
                ommx_instance = problem.eval(instance_data)

                # Read and evaluate the solution
                solution = None
                try:
                    sol_objective, sol_values = parse_sol_file(
                        sol_lines, symbols, num_periods
                    )
                    var_ids = {
                        (var.name, tuple(var.subscripts)): var.id
                        for var in ommx_instance.decision_variables
                    }
                    state = {
                        var_ids[(name, subscripts)]: value
                        for (name, subscripts), value in sol_values.items()
                    }
                    solution = ommx_instance.evaluate(state)

                    if (
                        math.isclose(
                            sol_objective, solution.objective, rel_tol=1e-6
                        )
                        and solution.feasible
                    ):
                        print(
                            f"  → Calculated objective={solution.objective:.6f}, "
                            f"Solution objective={sol_objective:.6f}"
                        )
                        print(
                            f"  → objective={solution.objective:.6f}, feasible={solution.feasible}"
                        )
                    else:
                        diff = solution.objective - sol_objective
                        print("Objective or feasible Error")
                        print(
                            f"  ✗ mismatch: calc={solution.objective:.6f}, sol={sol_objective:.6f}, "
                            f"Δ={diff:.6g}, feasible={solution.feasible}"
                        )

                except Exception as sol_error:
                    print(f"  ! Error evaluating solution: {sol_error}")
                    print(
                        "    Skipping solution evaluation and only saving the instance..."
                    )

                # Construct the output filename
                output_filename = os.path.join(output_directory, f"{base_name}.ommx")

                # If the file already exists, remove it
                if os.path.exists(output_filename):
                    os.remove(output_filename)

                # Add annotations to the instance.
                ommx_instance.title = base_name
                ommx_instance.license = LICENSE
                ommx_instance.dataset = "Portfolio Optimization"
                ommx_instance.authors = QOBLIB_AUTHORS
                ommx_instance.num_variables = len(ommx_instance.decision_variables)
                ommx_instance.num_constraints = len(ommx_instance.constraints)
                ommx_instance.annotations["org.ommx.qoblib.url"] = (
                    "https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/06-portfolio?ref_type=heads"
                )

                # Create OMMX Artifact
                builder = ArtifactBuilder.new_archive_unnamed(output_filename)
                instance_desc = builder.add_instance(ommx_instance)
                if solution is not None:
                    solution.instance = instance_desc.digest
                    solution.annotations["org.ommx.qoblib.authors"] = (
                        QOBLIB_AUTHORS_STR
                    )
                    builder.add_solution(solution)
                builder.build()

                print(f"Successfully created: {output_filename}")
                print("-" * 50)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
                error_count += 1
                continue

    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Number of errors: {error_count} files")
    print(f"OMMX files saved in: {output_directory}")
    print("-" * 50)


if __name__ == "__main__":
    batch_process_files()
