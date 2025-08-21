from datetime import datetime
import gc
import glob
import os
from pathlib import Path
import time
import traceback

from dateutil.tz import tzlocal
import jijmodeling as jm
import ommx.v1
from ommx.artifact import ArtifactBuilder

from dat_reader import load_topology_instance
from model import create_topology_model
from sol_reader import (
    parse_topology_sol_file,
    read_topology_solution_file_as_jijmodeling_format,
)


# Global optimization: reuse timestamp for better performance
_CREATION_TIME = datetime.now(tzlocal())


def verify_solution_quality(
    ommx_instance: ommx.v1.Instance,
    solution_path: str,
    instance_data: dict[str, object],
    epsilon: float = 1e-6,
) -> dict[str, object]:
    """Verify solution quality using OMMX Instance.evaluate() method.

    Args:
        ommx_instance (ommx.v1.Instance): OMMX instance for evaluation
        solution_path (str): Path to the solution file
        instance_data (dict[str, object]): Loaded instance data
        epsilon (float): Tolerance for objective value comparison

    Raises:
        ValueError: If solution is not feasible or computed objective does not match file objective

    Returns:
        dict[str, object]: Verification results containing:
            - feasible: Whether solution is feasible (if found)
            - objective_match: Whether computed objective matches file objective
            - file_objective: Objective value from solution file
            - computed_objective: Computed objective value
    """
    # Initialise the result dictionary
    result = {
        "feasible": None,
        "objective_match": None,
        "file_objective": None,
        "computed_objective": None,
        "ommx_solution": None,
    }

    # Parse solution file to get original objective
    print(f"Parsing solution file...")
    solution_data = parse_topology_sol_file(solution_path)

    # Store the original objective from the solution file (diameter for topology problems)
    result["file_objective"] = solution_data.get("diameter")

    # Load solution in JijModeling format.
    jm_solution = read_topology_solution_file_as_jijmodeling_format(
        solution_path, instance_data
    )

    # Get decision variables from OMMX instance using the correct API
    print("Getting OMMX decision variables...")

    # Use decision_variables_df to get pandas DataFrame with variable IDs as index
    decision_vars_df = ommx_instance.decision_variables_df

    var_ids = decision_vars_df.index.tolist()  # Variable IDs are the index
    print(f"Found {len(var_ids)} OMMX decision variables")

    # Create solution dictionary by matching variable names and subscripts
    solution_dict = {}

    print("Mapping JijModeling solution to OMMX variables by name/subscripts...")

    # Pre-extract variable info for vectorized processing
    var_names = decision_vars_df["name"].values
    var_subscripts = decision_vars_df["subscripts"].values
    var_ids = decision_vars_df.index.values

    # Process variables by mapping from JijModeling solution to OMMX variable IDs
    for var_name, subscripts, var_id in zip(var_names, var_subscripts, var_ids):
        if var_name == "diameter":
            # Scalar variable - no subscripts
            solution_dict[var_id] = jm_solution["diameter"]
        elif var_name == "SP":
            # 2D variable with subscripts [s, t]
            s, t = subscripts
            solution_dict[var_id] = jm_solution["SP"][s][t]
        elif var_name == "z":
            # 2D variable with subscripts [i, j]
            i, j = subscripts
            solution_dict[var_id] = jm_solution["z"][i][j]
        elif var_name == "x":
            # 4D variable with subscripts [s, t, i, j]
            s, t, i, j = subscripts
            solution_dict[var_id] = jm_solution["x"][s][t][i][j]
        else:
            raise ValueError(f"Invalid variable name: {var_name}")

    # Ensure all OMMX variables have values (this should not be needed with new implementation)
    missing_vars = set(var_ids) - set(solution_dict.keys())
    if missing_vars:
        print(f"Warning: {len(missing_vars)} variables missing, setting to 0")
        for vid in missing_vars:
            solution_dict[vid] = 0

    print(f"Created solution dictionary with {len(solution_dict)} variable assignments")

    # Evaluate solution using OMMX Instance.evaluate() method
    print("Evaluating solution with OMMX...")
    ommx_solution = ommx_instance.evaluate(solution_dict)
    result["ommx_solution"] = ommx_solution

    # Check if solution is feasible and get computed objective value
    result["feasible"] = ommx_solution.feasible
    result["computed_objective"] = ommx_solution.objective

    # Compute the difference between file objective and computed objective
    # and check if they match.
    if (
        result["file_objective"] is not None
        and result["computed_objective"] is not None
    ):
        diff = abs(result["computed_objective"] - result["file_objective"])
        result["objective_match"] = diff < epsilon

    return result


def verify_solution_qualities(
    instance_name: str,
    ommx_instance: ommx.v1.Instance,
    instance_data: dict[str, object],
    solution_directory: str,
    epsilon: float = 1e-6,
) -> tuple[bool, dict[str, object]]:
    """Verify solution quality using OMMX Instance.evaluate() method.

    Args:
        instance_name (str): Name of the instance (e.g., "stp_s020_l2_t3_h2_rs24098")
        ommx_instance (ommx.v1.Instance): OMMX instance for evaluation
        instance_data (dict[str, object]): Loaded instance data
        solution_directory (str): Path to directory containing solution files
        epsilon (float): Tolerance for objective value comparison

    Returns:
        tuple[bool, bool, dict[str, object]]: Tuple containing:
            - bool: Whether all solutions are feasible
            - bool: Whether all computed objectives match file objectives
            - dict[str, object]: Dictionary with detailed results for each solution file
    """
    # Initialise the results directory.
    results = dict()

    # Look for solution file with different extensions
    solution_paths = glob.glob(
        os.path.join(solution_directory, f"{instance_name}*.gph")
    ) + glob.glob(os.path.join(solution_directory, f"{instance_name}*.gph.gz"))

    for solution_path in solution_paths:
        results[solution_path] = verify_solution_quality(
            ommx_instance=ommx_instance,
            solution_path=solution_path,
            instance_data=instance_data,
            epsilon=epsilon,
        )

    is_feasible = all(result["feasible"] for result in results.values())
    is_objective_match = all(result["objective_match"] for result in results.values())

    return (is_feasible, is_objective_match, results)


def process_single_instance(
    instance_path: str, output_directory: str, solution_directory: str | None = None
) -> None:
    """Process a single Topology instance and create OMMX file.

    This function handles the complete pipeline for processing a single instance:
    1. Load instance data from the specified directory
    2. Convert to OMMX format using JijModeling
    3. Add metadata (title, creation timestamp)
    4. Save as .ommx file with proper cleanup

    Args:
        instance_path (str): Path to the directory containing instance files (.dat files)
        output_directory (str): Directory where the .ommx file will be saved
        solution_directory (str | None): Directory containing solution files for verification

    Raises:
        ValueError: if the instance is not feasible
        ValueError: if the computed objective does not match the file objective

    Note:
        Performs aggressive memory cleanup including garbage collection
        to optimize memory usage in parallel processing scenarios.
    """
    # Load instance data
    print(f"Loading instance data from {instance_path}...", flush=True)
    data = load_topology_instance(instance_path)

    # Convert to OMMX instance
    print("Creating OMMX instance...", flush=True)
    problem = create_topology_model()

    # Create instance data mapping for JijModeling Interpreter
    # This approach is more robust and explicit than passing the full dict
    used_placeholders = problem.used_placeholders()
    instance_data = {
        ph.name: data[ph.name] for ph in used_placeholders if ph.name in data
    }
    interpreter = jm.Interpreter(instance_data)
    print("Evaluating problem...", flush=True)
    ommx_instance = interpreter.eval_problem(problem)

    # Verify solution quality if solution directory is provided
    print(f"Verifying solutions qualities...", flush=True)
    instance_name = Path(instance_path).stem
    solution = None
    if solution_directory:
        is_feasible, is_objective_match, results = verify_solution_qualities(
            instance_name=instance_name,
            ommx_instance=ommx_instance,
            instance_data=data,
            solution_directory=solution_directory,
        )
        if not is_feasible:
            raise ValueError(
                f"There is a not feasible solution for instance {instance_name}."
                f" Results: {results}"
            )
        if not is_objective_match:
            raise ValueError(
                f"Computed objective does not match file objective for instance {instance_name}."
                f" Results: {results}"
            )

        # Try to store the optimal solution first.
        opt_solution_paths = glob.glob(
            os.path.join(solution_directory, f"{instance_name}*.opt.gph*")
        )
        if len(opt_solution_paths) > 0:
            solution = results[opt_solution_paths[0]]["ommx_solution"]
        # If no optimal solution, try the regular solution file.
        else:
            solution_paths = glob.glob(
                os.path.join(solution_directory, f"{instance_name}.gph*")
            )
            if solution_paths:
                solution = results[solution_paths[0]]["ommx_solution"]
            else:
                print(
                    f"No solution files found for {instance_name}, skipping solution attachment."
                )

    # Add metadata (reuse global timestamp)
    ommx_instance.title = f"Topology (flow_mip): {instance_name}"
    ommx_instance.created = _CREATION_TIME

    # Create output filename
    output_filename = os.path.join(output_directory, f"{instance_name}.ommx")

    # Optimized file handling
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create OMMX Artifact
    builder = ArtifactBuilder.new_archive_unnamed(output_filename)
    builder.add_instance(ommx_instance)
    if solution is not None:
        builder.add_solution(solution)
    else:
        print(
            f"No solution provided for {instance_name}, skipping solution attachment."
        )
    builder.build()

    # Aggressive memory cleanup
    del ommx_instance, builder, data
    gc.collect()


def get_node_count_from_dat_file(instance_path: str) -> int:
    """Read node count from .dat file.

    Args:
        instance_path (str): Path to .dat file

    Returns:
        int: Number of nodes, or 0 if file not found or parsing fails
    """
    try:
        data = load_topology_instance(instance_path)
        return data.get("nodes", 0)
    except Exception:
        return 0


def batch_process_instances(
    instances_directory: str = "../../instances",
    solution_directory: str = "../../solutions",
    output_directory: str = "./ommx_output",
    max_nodes: int | None = None,
) -> None:
    """Process all Topology instances sequentially.

    Args:
        instances_directory (str): Path to directory containing instance .dat files
        solution_directory (str): Path to directory containing solution files for verification
        output_directory (str): Path where .ommx files will be saved
        max_nodes (int | None): Maximum number of nodes to process (for memory management). Default is None (no limit).

    Note:
        - Only processes .dat files
        - Skips files with more than max_nodes nodes
        - Provides progress reporting and timing statistics
    """
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # HARD CODING: Use specific instances because of memory issues with large instances
    instances_path = Path(instances_directory)
    dat_files = [
        instances_path / "topology_15_3.dat",
        instances_path / "topology_15_4.dat",
        instances_path / "topology_20_3.dat",
        instances_path / "topology_20_4.dat",
        instances_path / "topology_20_5.dat",
        instances_path / "topology_25_3.dat",
        instances_path / "topology_25_4.dat",
        instances_path / "topology_25_5.dat",
        instances_path / "topology_25_6.dat",
        instances_path / "topology_30_4.dat",
        instances_path / "topology_30_5.dat",
        instances_path / "topology_30_6.dat",
        instances_path / "topology_35_5.dat",
        instances_path / "topology_35_6.dat",
        instances_path / "topology_40_6.dat",
        instances_path / "topology_50_4.dat",
    ]

    for dat_file in instances_path.glob("*.dat"):
        if dat_file.is_file():
            node_count = get_node_count_from_dat_file(str(dat_file))
            if max_nodes is not None and 0 < node_count <= max_nodes:
                dat_files.append(str(dat_file))

    if not dat_files:
        print(f"No valid .dat files found in {instances_directory}", flush=True)
        return

    print(f"Found {len(dat_files)} topology instances", flush=True)
    print(f"Max nodes filter: <= {max_nodes}", flush=True)
    print(f"Output directory: {output_directory}", flush=True)
    print("=" * 80, flush=True)

    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Process instances sequentially
    for i, dat_file in enumerate(dat_files):
        instance_name = Path(dat_file).stem  # Remove .dat extension
        progress = ((i + 1) / len(dat_files)) * 100

        try:
            process_single_instance(dat_file, output_directory, solution_directory)

            processed_count += 1
            print(
                f"✓ [{i+1:3d}/{len(dat_files)}] ({progress:5.1f}%) {instance_name}",
                flush=True,
            )

        except Exception as e:
            error_count += 1
            print(
                f"✗ [{i+1:3d}/{len(dat_files)}] ({progress:5.1f}%) {instance_name} - Exception: {e}",
                flush=True,
            )

        # Force garbage collection after each instance
        gc.collect()

    elapsed_time = time.time() - start_time

    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total instances: {len(dat_files)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    if len(dat_files) > 0:
        print(f"  Average time per instance: {elapsed_time/len(dat_files):.2f} seconds")
        print(f"  Throughput: {len(dat_files)/elapsed_time:.2f} instances/second")
    print(f"  Output files saved to: {os.path.abspath(output_directory)}")


def main() -> None:
    """Main function to execute topology OMMX conversion.

    Supports both single instance and batch processing modes.
    """
    print("Batch Processing Topology Instances to OMMX")
    print("=" * 60)

    try:
        instances_dir = "../../instances"
        output_dir = "./ommx_output"
        solution_dir = "../../solutions"
        batch_process_instances(instances_dir, solution_dir, output_dir)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
