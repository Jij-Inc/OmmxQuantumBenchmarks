"""
Optimized OMMX creation with multiple performance improvements
- Memory management optimizations
- Efficient data structures
- Reduced object creation overhead
- Optimized file I/O
"""

import os
import logging
import traceback
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import gc  # Garbage collection for memory management

from dateutil.tz import tzlocal
from ommx.artifact import ArtifactBuilder
import jijmodeling as jm

from dat_reader import load_steiner_instance
from model import create_steiner_tree_packing_model

logging.basicConfig(level=logging.INFO)

# Global optimization: reuse timestamp for better performance
_CREATION_TIME = datetime.now(tzlocal())


def jijmodeling_to_ommx_instance_optimized(data: dict[str, object]) -> object:
    """Optimized conversion with memory management."""
    problem = create_steiner_tree_packing_model()
    interpreter = jm.Interpreter(data)
    ommx_instance = interpreter.eval_problem(problem)
    
    # Clear interpreter to free memory
    del interpreter, problem
    return ommx_instance


def process_single_instance_optimized(instance_path: str, output_directory: str) -> bool:
    """Optimized instance processing with memory management."""
    try:
        instance_name = Path(instance_path).name

        # Load instance data
        data = load_steiner_instance(instance_path)

        # Convert to OMMX instance (optimized)
        ommx_instance = jijmodeling_to_ommx_instance_optimized(data)
        
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

    except Exception as e:
        # Cleanup on error too
        gc.collect()
        return False


def process_instance_wrapper_optimized(args: tuple[str, str]) -> tuple[bool, str]:
    """Optimized wrapper with error handling."""
    instance_dir, output_directory = args
    try:
        result = process_single_instance_optimized(instance_dir, output_directory)
        return result, Path(instance_dir).name
    except Exception as e:
        return False, Path(instance_dir).name


def batch_process_instances_optimized(
    instances_directory: str = "../../instances",
    output_directory: str = "./ommx_output_optimized",
    max_workers: int | None = None,
    chunk_size: int | None = None
) -> None:
    """Optimized batch processing with chunking and memory management."""
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Optimized directory scanning
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

    # Optimized worker allocation
    if max_workers is None:
        max_workers = min(cpu_count(), len(instance_dirs))
    
    # Chunking for better memory management
    if chunk_size is None:
        chunk_size = max(1, len(instance_dirs) // (max_workers * 2))

    print(f"Found {len(instance_dirs)} instance directories")
    print(f"Output directory: {output_directory}")
    print(f"Using {max_workers} parallel workers")
    print(f"Chunk size: {chunk_size}")
    print("=" * 80)

    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Process in chunks to manage memory better
    for chunk_start in range(0, len(instance_dirs), chunk_size * max_workers):
        chunk_end = min(chunk_start + chunk_size * max_workers, len(instance_dirs))
        chunk_dirs = instance_dirs[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//chunk_size + 1}: instances {chunk_start+1}-{chunk_end}")
        
        # Prepare arguments for this chunk
        args_list = [(instance_dir, output_directory) for instance_dir in sorted(chunk_dirs)]
        
        # Process chunk in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_instance = {
                executor.submit(process_instance_wrapper_optimized, args): args[0] 
                for args in args_list
            }
            
            for future in as_completed(future_to_instance):
                instance_dir = future_to_instance[future]
                try:
                    success, instance_name = future.result()
                    completed = processed_count + error_count + 1
                    progress = (completed / len(instance_dirs)) * 100
                    
                    if success:
                        processed_count += 1
                        print(f"✓ [{completed:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name}")
                    else:
                        error_count += 1
                        print(f"✗ [{completed:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name}")
                except Exception as e:
                    error_count += 1
                    completed = processed_count + error_count
                    progress = (completed / len(instance_dirs)) * 100
                    instance_name = Path(instance_dir).name
                    print(f"✗ [{completed:3d}/{len(instance_dirs)}] ({progress:5.1f}%) {instance_name} - Exception: {e}")
        
        # Force garbage collection between chunks
        gc.collect()
        print(f"Completed chunk {chunk_start//chunk_size + 1} - Memory cleanup done")
        print("-" * 40)

    elapsed_time = time.time() - start_time

    print("=" * 80)
    print("OPTIMIZED SUMMARY:")
    print(f"  Total instances: {len(instance_dirs)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"  Average time per instance: {elapsed_time/len(instance_dirs):.2f} seconds")
    print(f"  Throughput: {len(instance_dirs)/elapsed_time:.2f} instances/second")
    print(f"  Output files saved to: {os.path.abspath(output_directory)}")


def main() -> None:
    """Main function with optimization flags."""
    print("Batch Processing Steiner Tree Packing Instances to OMMX (OPTIMIZED)")
    print("=" * 70)
    
    try:
        instances_dir = "../../instances"
        output_dir = "./ommx_output_optimized"
        batch_process_instances_optimized(instances_dir, output_dir)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()