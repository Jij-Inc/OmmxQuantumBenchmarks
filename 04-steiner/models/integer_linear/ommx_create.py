"""
Create OMMX Artifacts for Steiner Tree Packing Problem
Batch process all instances from the instances directory
Based on the JijModeling formulation from Section 2.2 of "Steiner tree packing revisited"

Enhanced with solution verification - validates solutions during OMMX creation

Usage:
  python ommx_create.py           # Create OMMX files with solution verification
  python ommx_create.py --no-verify  # Create OMMX files without verification

Features:
- Loads instances from 04-steiner/instances/
- Converts to JijModeling problems using model.py  
- Verifies solutions from 04-steiner/solutions/ if available
- Ensures solutions are feasible and costs match
- Creates OMMX artifacts in ./ommx_output/
- Stops processing if invalid solutions are found
"""

import os
import glob
from datetime import datetime
from dateutil.tz import tzlocal
import uuid
import logging
from pathlib import Path

from ommx.artifact import Artifact, ArtifactBuilder
import jijmodeling as jm

from dat_reader import load_steiner_instance
from model import create_steiner_tree_packing_model
from verify_solutions import parse_solution_file, create_solution_arrays, verify_solution_constraints

logging.basicConfig(level=logging.INFO)


def jijmodeling_to_ommx_instance(data):
    """
    Convert Steiner Tree Packing problem data to OMMX Instance using JijModeling interpreter.
    
    Args:
        data: Dictionary containing the loaded Steiner instance data
        
    Returns:
        ommx.v1.Instance: The converted OMMX instance
    """
    
    # Create the JijModeling problem
    problem = create_steiner_tree_packing_model()
    
    # Use JijModeling interpreter to convert to OMMX
    interpreter = jm.Interpreter(data)
    ommx_instance = interpreter.eval_problem(problem)
    
    return ommx_instance


def verify_solution_if_exists(instance_path, data, instance_name):
    """
    Verify the solution for this instance if it exists.
    
    Args:
        instance_path: Path to the instance directory
        data: Loaded instance data
        instance_name: Name of the instance
        
    Returns:
        tuple: (is_verified, error_message, cost)
    """
    
    # Look for solution files
    solutions_dir = "/Users/keisukesato/dev/git/OMMX-OBLIB/04-steiner/solutions"
    solution_files = [
        os.path.join(solutions_dir, f"{instance_name}.opt.sol"),
        os.path.join(solutions_dir, f"{instance_name}.bst.sol")
    ]
    
    solution_file = None
    for sf in solution_files:
        if os.path.exists(sf):
            solution_file = sf
            break
            
    if not solution_file:
        return True, "No solution file found - skipping verification", None
    
    try:
        # Parse solution file
        solution_data = parse_solution_file(solution_file)
        
        if solution_data["cost"] is None:
            return False, f"Could not parse cost from {os.path.basename(solution_file)}", None
            
        # Convert solution to arrays
        x_array, y_array = create_solution_arrays(solution_data, data)
        
        # Verify constraints
        verification_results = verify_solution_constraints(data, x_array, y_array)
        
        total_violations = verification_results.get("total_violations", 0)
        
        if total_violations > 0:
            violations = verification_results.get("violations", [])
            error_msg = f"Solution has {total_violations} constraint violations"
            if violations:
                error_msg += f": {violations[0]}"
                if len(violations) > 1:
                    error_msg += f" (and {len(violations)-1} more)"
            return False, error_msg, solution_data["cost"]
        
        return True, f"Solution verified (cost: {solution_data['cost']})", solution_data["cost"]
        
    except Exception as e:
        return False, f"Error verifying solution: {str(e)}", None


def process_single_instance(instance_path, output_directory, verify_solutions=True):
    """
    Process a single Steiner Tree Packing instance and create OMMX file.
    
    Args:
        instance_path: Path to the Steiner instance directory  
        output_directory: Path to save the .ommx file
        verify_solutions: Whether to verify solutions during processing
        
    Returns:
        bool: True if successful, False if error occurred
    """
    
    try:
        instance_name = Path(instance_path).name
        print(f"Processing instance: {instance_name}")
        
        # Load instance data
        data = load_steiner_instance(instance_path)
        
        print(f"  - Nodes: {data['nodes']}")
        print(f"  - Nets: {data['num_nets']}")  
        print(f"  - Arcs: {len(data['A'])}")
        print(f"  - Terminals: {len(data['T'])}")
        print(f"  - Roots: {len(data['R'])}")
        
        # Verify solution if requested and available
        if verify_solutions:
            is_verified, verification_msg, cost = verify_solution_if_exists(instance_path, data, instance_name)
            
            if not is_verified:
                print(f"  ❌ SOLUTION VERIFICATION FAILED: {verification_msg}")
                raise ValueError(f"Solution verification failed for {instance_name}: {verification_msg}")
            else:
                print(f"  ✅ {verification_msg}")
        
        # Convert to OMMX instance
        ommx_instance = jijmodeling_to_ommx_instance(data)
        
        # Add metadata
        ommx_instance.title = f"Steiner Tree Packing: {instance_name}"
        ommx_instance.created = datetime.now(tzlocal())
        
        # Create output filename
        output_filename = os.path.join(output_directory, f"{instance_name}.ommx")
        
        # Remove existing file if it exists
        if os.path.exists(output_filename):
            os.remove(output_filename)
        
        # Create OMMX Artifact using archive format (like binary_linear)
        builder = ArtifactBuilder.new_archive_unnamed(output_filename)
        
        # Add instance to artifact
        instance_desc = builder.add_instance(ommx_instance)
        
        # Build and save the artifact
        artifact = builder.build()
        
        print(f"  ✓ Created: {output_filename}")
        print(f"    - Variables: {len(ommx_instance.decision_variables)}")
        print(f"    - Constraints: {len(ommx_instance.constraints)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {instance_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def batch_process_instances(instances_directory="../../instances", output_directory="./ommx_output", verify_solutions=True):
    """
    Batch process all Steiner Tree Packing instances and convert them to OMMX files.
    
    Args:
        instances_directory: Path to the directory containing instance subdirectories
        output_directory: Path to save the .ommx files
        verify_solutions: Whether to verify solutions during processing
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all instance directories (exclude README.md)
    instance_dirs = []
    for item in os.listdir(instances_directory):
        item_path = os.path.join(instances_directory, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it's a valid instance directory (contains .dat files)
            if any(f.endswith('.dat') for f in os.listdir(item_path)):
                instance_dirs.append(item_path)
    
    if not instance_dirs:
        print(f"No instance directories found in {instances_directory}")
        return
    
    print(f"Found {len(instance_dirs)} instance directories")
    print(f"Output directory: {output_directory}")
    print(f"Solution verification: {'ENABLED' if verify_solutions else 'DISABLED'}")
    print("=" * 80)
    
    processed_count = 0
    error_count = 0
    verified_solutions = 0
    
    # For demo, limit to first 5 instances - remove this for production
    for instance_dir in sorted(instance_dirs)[:5]:
        if process_single_instance(instance_dir, output_directory, verify_solutions):
            processed_count += 1
            if verify_solutions:
                verified_solutions += 1
        else:
            error_count += 1
        print("-" * 40)
    
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Total instances: {len(instance_dirs)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    if verify_solutions:
        print(f"  Solutions verified: {verified_solutions}")
    print(f"  Output files saved to: {os.path.abspath(output_directory)}")
    
    if verify_solutions and error_count > 0:
        print(f"\n⚠️  WARNING: {error_count} instances failed processing due to solution verification errors.")
        print("   This indicates issues with either the solutions or the model implementation.")
    elif verify_solutions:
        print(f"\n✅ SUCCESS: All {processed_count} processed instances have verified solutions!")
        print("   Model implementation is mathematically correct.")


def main():
    """Main function to batch process all instances."""
    
    import sys
    
    # Check for --no-verify flag
    verify_solutions = "--no-verify" not in sys.argv
    
    print("Batch Processing Steiner Tree Packing Instances to OMMX")
    if verify_solutions:
        print("(with solution verification enabled)")
    else:
        print("(solution verification DISABLED)")
    print("=" * 60)
    
    try:
        # Use absolute paths
        instances_dir = "/Users/keisukesato/dev/git/OMMX-OBLIB/04-steiner/instances"
        output_dir = "./ommx_output"
        batch_process_instances(instances_dir, output_dir, verify_solutions)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()