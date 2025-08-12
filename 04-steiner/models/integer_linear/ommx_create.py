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

import gc
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from dateutil.tz import tzlocal
from ommx.artifact import ArtifactBuilder
import jijmodeling as jm
import ommx

from dat_reader import load_steiner_instance
from model import create_steiner_tree_packing_model
from sol_reader import (
    parse_steiner_sol_file,
    read_steiner_solution_file_as_jijmodeling_format,
)

# Global optimization: reuse timestamp for better performance
_CREATION_TIME = datetime.now(tzlocal())


def verify_solution_quality(
    instance_name: str,
    ommx_instance: ommx.v1.Instance,
    instance_data: dict[str, object],
    solution_directory: str,
) -> dict[str, object]:
    """Verify solution quality using OMMX Instance.evaluate() method.

    Args:
        instance_name (str): Name of the instance (e.g., "stp_s020_l2_t3_h2_rs24098")
        ommx_instance (ommx.v1.Instance): OMMX instance for evaluation
        instance_data (dict[str, object]): Loaded instance data
        solution_directory (str): Path to directory containing solution files

    Returns:
        dict[str, object]: Verification results containing:
            - found: Whether solution file was found
            - feasible: Whether solution is feasible (if found)
            - objective_match: Whether computed objective matches file objective
            - file_objective: Objective value from solution file
            - computed_objective: Computed objective value
            - error: Error message if any
    """
    result = {
        "found": False,
        "feasible": None,
        "objective_match": None,
        "file_objective": None,
        "computed_objective": None,
        "error": None,
    }

    try:
        # Look for solution file with different extensions
        solution_files = [
            f"{instance_name}.opt.sol",
            f"{instance_name}.bst.sol",
            f"{instance_name}.sol",
        ]

        solution_path = None
        for sol_file in solution_files:
            candidate_path = os.path.join(solution_directory, sol_file)
            if os.path.exists(candidate_path):
                solution_path = candidate_path
                break

        if not solution_path:
            result["error"] = f"No solution file found for {instance_name}"
            return result

        result["found"] = True

        # Parse solution file to get original objective
        print(f"Parsing solution file...")
        solution_data = parse_steiner_sol_file(solution_path)
        result["file_objective"] = solution_data.get("objective")

        # Load solution in JijModeling format
        jm_solution = read_steiner_solution_file_as_jijmodeling_format(
            solution_path, instance_data
        )

        try:
            # Create solution dictionary with simple index-based variable mapping
            solution_dict = {}
            var_idx = 0
            print("Mapping solution variables...")
            # Map y variables
            if "y" in jm_solution:
                for a in range(len(jm_solution["y"])):
                    for k in range(len(jm_solution["y"][a])):
                        solution_dict[var_idx] = int(jm_solution["y"][a][k])
                        var_idx += 1

            # Map x variables
            if "x" in jm_solution:
                for tail in range(len(jm_solution["x"])):
                    for head in range(len(jm_solution["x"][tail])):
                        solution_dict[var_idx] = int(jm_solution["x"][tail][head])
                        var_idx += 1

            # Evaluate solution using OMMX Instance.evaluate() method
            print("Evaluating solution...")
            ommx_solution = ommx_instance.evaluate(solution_dict)

            # Check if solution is feasible
            result["feasible"] = ommx_solution.is_feasible()

            # Get computed objective value
            if ommx_solution.is_feasible():
                result["computed_objective"] = ommx_solution.objective_value()
            else:
                result["computed_objective"] = None

            # Compare objectives
            if (
                result["file_objective"] is not None
                and result["computed_objective"] is not None
            ):
                diff = abs(result["computed_objective"] - result["file_objective"])
                result["objective_match"] = diff < 1e-6

        except Exception as eval_error:
            result["error"] = f"OMMX evaluation failed: {str(eval_error)}"
            result["feasible"] = False

    except Exception as e:
        result["error"] = f"Solution verification failed: {str(e)}"

    return result


def process_single_instance(
    instance_path: str, output_directory: str, solution_directory: str | None = None
) -> bool:
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

    Returns:
        bool: True if processing succeeded, False if any error occurred

    Note:
        Performs aggressive memory cleanup including garbage collection
        to optimize memory usage in parallel processing scenarios.
    """
    try:
        instance_name = Path(instance_path).name

        # Load instance data
        data = load_steiner_instance(instance_path)

        # Convert to OMMX instance
        print("Creating OMMX instance...")
        problem = create_steiner_tree_packing_model()
        interpreter = jm.Interpreter(data)
        print("Evaluating problem...")
        ommx_instance = interpreter.eval_problem(problem)

        # Verify solution quality if solution directory is provided
        verification_result = None
        if solution_directory:
            verification_result = verify_solution_quality(
                instance_name, ommx_instance, data, solution_directory
            )
            print(f"[{instance_name}] Solution verification:")
            print(f"  Found: {verification_result['found']}")
            if verification_result["found"]:
                print(f"  Feasible: {verification_result['feasible']}")
                print(f"  File objective: {verification_result['file_objective']}")
                print(
                    f"  Computed objective: {verification_result['computed_objective']}"
                )
                print(f"  Objective match: {verification_result['objective_match']}")
                if verification_result["error"]:
                    print(f"  Error: {verification_result['error']}")

        # Immediately free data memory
        del data
        gc.collect()  # Force garbage collection

        # Add metadata (reuse global timestamp)
        ommx_instance.title = f"Steiner Tree Packing: {instance_name}"
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
        del ommx_instance, builder
        gc.collect()

        return True

    except Exception:
        # Ensure memory cleanup even on failure
        gc.collect()
        return False


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

    # Directory scanning
    instances_path = Path(instances_directory)
    instance_dirs = [
        str(item_path)
        for item_path in instances_path.iterdir()
        if (
            item_path.is_dir()
            and not item_path.name.startswith(".")
            and any(f.suffix == ".dat" for f in item_path.iterdir() if f.is_file())
        )
    ]
    instance_dirs = instance_dirs[:1]

    if not instance_dirs:
        print(f"No instance directories found in {instances_directory}")
        return

    print(f"Found {len(instance_dirs)} instance directories")
    print(f"Output directory: {output_directory}")
    print("=" * 80)

    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Process instances sequentially
    for i, instance_dir in enumerate(sorted(instance_dirs)):
        instance_name = Path(instance_dir).name
        progress = ((i + 1) / len(instance_dirs)) * 100

        try:
            success = process_single_instance(
                instance_dir, output_directory, solution_directory
            )

            if success:
                processed_count += 1
                print(
                    f"✓ [{i+1:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name}"
                )
            else:
                error_count += 1
                print(
                    f"✗ [{i+1:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name}"
                )

        except Exception as e:
            error_count += 1
            print(
                f"✗ [{i+1:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name} - Exception: {e}"
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
