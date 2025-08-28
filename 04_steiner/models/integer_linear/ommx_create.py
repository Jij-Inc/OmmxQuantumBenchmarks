"""
Create OMMX Artifacts for Steiner Tree Packing Problem
Process all instances from the instances directory sequentially

Features:
- Memory management optimizations
- Efficient data structures
- Reduced object creation overhead
- Optimized file I/O
- Sequential processing
"""

from datetime import datetime
import gc
import glob
import os
from pathlib import Path
import time
import traceback

from dateutil.tz import tzlocal
import jijmodeling as jm
import ommx
from ommx.artifact import ArtifactBuilder

from dat_reader import load_steiner_instance
from model import create_steiner_tree_packing_model
from sol_reader import (
    parse_steiner_sol_file,
    read_steiner_solution_file_as_jijmodeling_format,
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
    solution_data = parse_steiner_sol_file(solution_path)

    # Store the original objective from the solution file
    result["file_objective"] = solution_data.get("objective")

    # Load solution in JijModeling format.
    jm_solution = read_steiner_solution_file_as_jijmodeling_format(
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

    # Process variables in batches by type for better performance
    y_mask = var_names == "y"
    x_mask = var_names == "x"
    z_mask = var_names == "z"

    # Get arc to index mapping for arc-based variables
    arc_to_index = {
        (tail, head): idx for idx, (tail, head) in enumerate(instance_data["A"])
    }

    # Process y variables - now arc-based: y[arc_idx, net_idx]
    y_ids = var_ids[y_mask]
    y_subscripts = var_subscripts[y_mask]
    for i, var_id in enumerate(y_ids):
        subscripts = y_subscripts[i]
        if len(subscripts) == 2:
            # New arc-based format: y[arc_idx, net_idx]
            arc_idx, k = subscripts[0], subscripts[1]
            solution_dict[var_id] = int(jm_solution["y"][arc_idx][k])
        elif len(subscripts) == 3:
            # Legacy node-based format: y[tail, head, net_idx] -> convert to arc_idx
            tail, head, k = subscripts[0], subscripts[1], subscripts[2]
            arc_idx = arc_to_index.get((tail, head))
            if arc_idx is not None:
                solution_dict[var_id] = int(jm_solution["y"][arc_idx][k])
            else:
                solution_dict[var_id] = 0  # Arc not found in instance
        else:
            raise ValueError(f"Invalid y variable subscripts: {subscripts}")

    # Process x variables - now arc-based: x[arc_idx, terminal_idx]
    x_ids = var_ids[x_mask]
    x_subscripts = var_subscripts[x_mask]
    for i, var_id in enumerate(x_ids):
        subscripts = x_subscripts[i]
        if len(subscripts) == 2:
            # New arc-based format: x[arc_idx, terminal_idx]
            arc_idx, t = subscripts[0], subscripts[1]
            solution_dict[var_id] = int(jm_solution["x"][arc_idx][t])
        elif len(subscripts) == 3:
            # Legacy node-based format: x[tail, head, terminal_idx] -> convert to arc_idx
            tail, head, t = subscripts[0], subscripts[1], subscripts[2]
            arc_idx = arc_to_index.get((tail, head))
            if arc_idx is not None:
                solution_dict[var_id] = int(jm_solution["x"][arc_idx][t])
            else:
                solution_dict[var_id] = 0  # Arc not found in instance
        else:
            raise ValueError(f"Invalid x variable subscripts: {subscripts}")

    # Process z variables - unchanged: z[root_idx, terminal_idx]
    z_ids = var_ids[z_mask]
    z_subscripts = var_subscripts[z_mask]
    for i, var_id in enumerate(z_ids):
        subscripts = z_subscripts[i]
        if len(subscripts) == 2:
            r, t = subscripts[0], subscripts[1]
            solution_dict[var_id] = int(jm_solution["z"][r][t])
        else:
            raise ValueError(f"Invalid z variable subscripts: {subscripts}")

    # Check for any unhandled variable types
    handled_mask = y_mask | x_mask | z_mask
    if not handled_mask.all():
        unhandled_vars = var_names[~handled_mask]
        raise ValueError(f"Unknown variable types: {set(unhandled_vars)}")

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
        os.path.join(solution_directory, f"{instance_name}*.sol")
    )

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
    """Process a single Steiner Tree Packing instance and create OMMX file.

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
    data = load_steiner_instance(instance_path)

    # Convert to OMMX instance
    print("Creating OMMX instance...", flush=True)
    problem = create_steiner_tree_packing_model()

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
    instance_name = Path(instance_path).name
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
            os.path.join(solution_directory, f"{instance_name}*.opt.sol")
        )
        if len(opt_solution_paths) > 0:
            solution = results[opt_solution_paths[0]]["ommx_solution"]
        # If no optimal solution, try the regular solution file.
        else:
            solution_paths = glob.glob(
                os.path.join(solution_directory, f"{instance_name}.sol")
            )
            if solution_paths:
                solution = results[solution_paths[0]]["ommx_solution"]
            else:
                print(
                    f"No solution files found for {instance_name}, skipping solution attachment."
                )
    solution_path = os.path.join(instance_path, "sol.txt")
    if os.path.isfile(solution_path):
        result = verify_solution_quality(
            ommx_instance=ommx_instance,
            solution_path=solution_path,
            instance_data=data,
        )
        if not result["feasible"]:
            raise ValueError(
                f"Solution is not feasible for instance {instance_name}."
                f" Result: {result}"
            )
        if not result["objective_match"]:
            raise ValueError(
                f"Computed objective does not match file objective for instance {instance_name}."
                f" Result: {result}"
            )
        # If solution is still None, use the result from sol.txt
        if solution is None:
            solution = result["ommx_solution"]

    # Create QOBLIB authors.
    qoblib_authors = [
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
    qoblib_authors_str = ", ".join(qoblib_authors)
    # Add annotations to the instance.
    ommx_instance.title = instance_name
    ommx_instance.license = "CC BY 4.0"
    ommx_instance.dataset = "Steiner Tree Packing Problem"
    ommx_instance.authors = qoblib_authors
    ommx_instance.num_variables = len(ommx_instance.decision_variables)
    ommx_instance.num_constraints = len(ommx_instance.constraints)
    ommx_instance.annotations["org.ommx.qoblib.url"] = (
        "https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/04-steiner?ref_type=heads"
    )
    ommx_instance.created = _CREATION_TIME

    # Create output filename
    output_filename = os.path.join(output_directory, f"{instance_name}.ommx")

    # Optimized file handling
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create OMMX Artifact
    builder = ArtifactBuilder.new_archive_unnamed(output_filename)
    instance_desc = builder.add_instance(ommx_instance)
    if solution is not None:
        solution.instance = instance_desc.digest
        solution.annotations["org.ommx.qoblib.authors"] = qoblib_authors_str
        builder.add_solution(solution)
    builder.build()

    # Aggressive memory cleanup
    del ommx_instance, builder, data
    gc.collect()


def get_node_count_from_param_dat(instance_path: str) -> int:
    """Read node count from param.dat file.

    Args:
        instance_path (str): Path to instance directory

    Returns:
        int: Number of nodes, or 0 if file not found or parsing fails
    """
    param_file = os.path.join(instance_path, "param.dat")
    try:
        with open(param_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nodes "):
                    return int(line.split()[1])
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0


def batch_process_instances(
    instances_directory: str = "../../instances",
    solution_directory: str = "../../solutions",
    output_directory: str = "./ommx_output",
) -> None:
    """Process all Steiner Tree Packing instances sequentially.

    Args:
        instances_directory (str): Path to directory containing instance subdirectories
        solution_directory (str): Path to directory containing solution files for verification
        output_directory (str): Path where .ommx files will be saved

    Note:
        - Only processes directories containing .dat files
        - Skips hidden directories (starting with '.')
        - Provides progress reporting and timing statistics
    """
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Directory scanning - filter by node count <= 4500 from param.dat
    instances_path = Path(instances_directory)
    instance_dirs = []

    for item_path in instances_path.iterdir():
        if (
            item_path.is_dir()
            and not item_path.name.startswith(".")
            and any(f.suffix == ".dat" for f in item_path.iterdir() if f.is_file())
        ):
            node_count = get_node_count_from_param_dat(str(item_path))
            if node_count > 0 and node_count <= 4500:
                instance_dirs.append(str(item_path))

    if not instance_dirs:
        print(f"No instance directories found in {instances_directory}", flush=True)
        return

    print(f"Found {len(instance_dirs)} instance directories", flush=True)
    print(f"Output directory: {output_directory}", flush=True)
    print("=" * 80, flush=True)

    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Process instances sequentially
    for i, instance_dir in enumerate(instance_dirs):
        instance_name = Path(instance_dir).name
        progress = ((i + 1) / len(instance_dirs)) * 100

        try:
            process_single_instance(instance_dir, output_directory, solution_directory)

            processed_count += 1
            print(
                f"✓ [{i+1:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name}",
                flush=True,
            )

        except Exception as e:
            error_count += 1
            print(
                f"✗ [{i+1:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name} - Exception: {e}",
                flush=True,
            )

        # Force garbage collection after each instance
        gc.collect()

    elapsed_time = time.time() - start_time

    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total instances: {len(instance_dirs)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"  Average time per instance: {elapsed_time/len(instance_dirs):.2f} seconds")
    print(f"  Throughput: {len(instance_dirs)/elapsed_time:.2f} instances/second")
    print(f"  Output files saved to: {os.path.abspath(output_directory)}")


def main() -> None:
    """Main function to execute batch processing of Steiner Tree instances.

    Entry point for the OMMX conversion script. Configures default paths
    and executes the batch processing pipeline with error handling.

    Default configuration:
    - Input: ../../instances (relative to script location)
    - Output: ./ommx_output (relative to script location)

    Raises:
        Exception: Any errors during batch processing are caught and reported
    """
    print("Batch Processing Steiner Tree Packing Instances to OMMX")
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
