from datetime import datetime
import gc
import glob
import os
from pathlib import Path

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
    instance_name = Path(instance_path).name
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

    # Add metadata (reuse global timestamp)
    ommx_instance.title = f"Topology: {instance_name}"
    ommx_instance.created = _CREATION_TIME

    # Create output filename
    output_filename = os.path.join(output_directory, f"{instance_name}.ommx")

    # Optimized file handling
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create OMMX Artifact
    builder = ArtifactBuilder.new_archive_unnamed(output_filename)
    builder.add_instance(ommx_instance)
    builder.build()

    # Aggressive memory cleanup
    del ommx_instance, builder, data
    gc.collect()
