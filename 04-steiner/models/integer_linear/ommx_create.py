"""
Create OMMX Artifacts for Steiner Tree Packing Problem
Batch process all instances from the instances directory
Based on the JijModeling formulation from Section 2.2 of "Steiner tree packing revisited"

Features:
- Loads instances from 04-steiner/instances/
- Converts to JijModeling problems using model.py
- Creates OMMX artifacts in ./ommx_output/
"""

import os
import logging
import traceback
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from dateutil.tz import tzlocal
from ommx.artifact import ArtifactBuilder
import jijmodeling as jm

from dat_reader import load_steiner_instance
from model import create_steiner_tree_packing_model

logging.basicConfig(level=logging.INFO)


def jijmodeling_to_ommx_instance(data: dict[str, object]) -> object:
    """Convert Steiner Tree Packing problem data to OMMX Instance.

    Args:
        data: Dictionary containing the loaded Steiner instance data

    Returns:
        The converted OMMX instance
    """
    problem = create_steiner_tree_packing_model()
    interpreter = jm.Interpreter(data)
    return interpreter.eval_problem(problem)


def process_single_instance(instance_path: str, output_directory: str) -> bool:
    """Process a single Steiner Tree Packing instance and create OMMX file.

    Args:
        instance_path: Path to the Steiner instance directory
        output_directory: Path to save the .ommx file

    Returns:
        True if successful, False if error occurred
    """
    try:
        instance_name = Path(instance_path).name

        # Load instance data
        data = load_steiner_instance(instance_path)

        # Convert to OMMX instance
        ommx_instance = jijmodeling_to_ommx_instance(data)
        
        # Clear data to free memory (important for parallel processing)
        del data

        # Add metadata (reuse timestamp for better performance)
        ommx_instance.title = f"Steiner Tree Packing: {instance_name}"
        ommx_instance.created = datetime.now(tzlocal())

        # Create output filename
        output_filename = os.path.join(output_directory, f"{instance_name}.ommx")

        # Remove existing file if it exists (optimized: check once)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Create OMMX Artifact
        builder = ArtifactBuilder.new_archive_unnamed(output_filename)
        builder.add_instance(ommx_instance)
        builder.build()
        
        # Clear variables to free memory
        del ommx_instance, builder

        return True

    except Exception as e:
        return False


def process_instance_wrapper(args: tuple[str, str]) -> tuple[bool, str]:
    """Wrapper function for multiprocessing."""
    instance_dir, output_directory = args
    return (
        process_single_instance(instance_dir, output_directory),
        Path(instance_dir).name,
    )


def batch_process_instances(
    instances_directory: str = "../../instances",
    output_directory: str = "./ommx_output",
    max_workers: int | None = None,
) -> None:
    """Batch process all Steiner Tree Packing instances and convert them to OMMX files.

    Args:
        instances_directory: Path to the directory containing instance subdirectories
        output_directory: Path to save the .ommx files
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Find all instance directories (optimized: use pathlib and generator expression)
    instances_path = Path(instances_directory)
    instance_dirs = [
        str(item_path) 
        for item_path in instances_path.iterdir() 
        if (item_path.is_dir() and 
            not item_path.name.startswith(".") and
            any(f.suffix == ".dat" for f in item_path.iterdir() if f.is_file()))
    ]

    if not instance_dirs:
        print(f"No instance directories found in {instances_directory}")
        return

    # Set default number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(instance_dirs))

    print(f"Found {len(instance_dirs)} instance directories")
    print(f"Output directory: {output_directory}")
    print(f"Using {max_workers} parallel workers")
    print("=" * 80)

    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Prepare arguments for parallel processing
    args_list = [
        (instance_dir, output_directory) for instance_dir in sorted(instance_dirs)
    ]
    total_instances = len(args_list)

    # Process instances in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(process_instance_wrapper, args): args[0]
            for args in args_list
        }

        for future in as_completed(future_to_instance):
            instance_dir = future_to_instance[future]
            try:
                success, instance_name = future.result()
                completed = processed_count + error_count + 1
                progress = (completed / total_instances) * 100

                if success:
                    processed_count += 1
                    print(
                        f"✓ [{completed:3d}/{total_instances}] ({progress:5.1f}%) {instance_name}"
                    )
                else:
                    error_count += 1
                    print(
                        f"✗ [{completed:3d}/{total_instances}] ({progress:5.1f}%) {instance_name}"
                    )
            except Exception as e:
                error_count += 1
                completed = processed_count + error_count
                progress = (completed / total_instances) * 100
                instance_name = Path(instance_dir).name
                print(
                    f"✗ [{completed:3d}/{total_instances}] ({progress:5.1f}%) {instance_name} - Exception: {e}"
                )

    elapsed_time = time.time() - start_time

    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total instances: {total_instances}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"  Average time per instance: {elapsed_time/total_instances:.2f} seconds")
    print(f"  Output files saved to: {os.path.abspath(output_directory)}")


def main() -> None:
    """Main function to batch process all instances."""
    print("Batch Processing Steiner Tree Packing Instances to OMMX")
    print("=" * 60)

    try:
        instances_dir = "../../instances"
        output_dir = "./ommx_output"
        batch_process_instances(instances_dir, output_dir)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
