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
from datetime import datetime
from pathlib import Path

from dateutil.tz import tzlocal
from ommx.artifact import ArtifactBuilder
import jijmodeling as jm

from dat_reader import load_steiner_instance
from model import create_steiner_tree_packing_model

logging.basicConfig(level=logging.INFO)


def jijmodeling_to_ommx_instance(data):
    """Convert Steiner Tree Packing problem data to OMMX Instance.
    
    Args:
        data (dict): Dictionary containing the loaded Steiner instance data
        
    Returns:
        OMMX instance: The converted OMMX instance
    """
    problem = create_steiner_tree_packing_model()
    interpreter = jm.Interpreter(data)
    return interpreter.eval_problem(problem)


def process_single_instance(instance_path, output_directory):
    """Process a single Steiner Tree Packing instance and create OMMX file.
    
    Args:
        instance_path (str): Path to the Steiner instance directory  
        output_directory (str): Path to save the .ommx file
        
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
        
        # Create OMMX Artifact
        builder = ArtifactBuilder.new_archive_unnamed(output_filename)
        builder.add_instance(ommx_instance)
        builder.build()
        
        print(f"  ✓ Created: {output_filename}")
        print(f"    - Variables: {len(ommx_instance.decision_variables)}")
        print(f"    - Constraints: {len(ommx_instance.constraints)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {instance_name}: {str(e)}")
        return False


def batch_process_instances(instances_directory="../../instances", output_directory="./ommx_output"):
    """Batch process all Steiner Tree Packing instances and convert them to OMMX files.
    
    Args:
        instances_directory (str): Path to the directory containing instance subdirectories
        output_directory (str): Path to save the .ommx files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all instance directories
    instance_dirs = []
    for item in os.listdir(instances_directory):
        item_path = os.path.join(instances_directory, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            if any(f.endswith('.dat') for f in os.listdir(item_path)):
                instance_dirs.append(item_path)
    
    if not instance_dirs:
        print(f"No instance directories found in {instances_directory}")
        return
    
    print(f"Found {len(instance_dirs)} instance directories")
    print(f"Output directory: {output_directory}")
    print("=" * 80)
    
    processed_count = 0
    error_count = 0
    
    for instance_dir in sorted(instance_dirs):
        if process_single_instance(instance_dir, output_directory):
            processed_count += 1
        else:
            error_count += 1
        print("-" * 40)
    
    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total instances: {len(instance_dirs)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output files saved to: {os.path.abspath(output_directory)}")


def main():
    """Main function to batch process all instances."""
    print("Batch Processing Steiner Tree Packing Instances to OMMX")
    print("=" * 60)
    
    try:
        instances_dir = "/Users/keisukesato/dev/git/OMMX-OBLIB/04-steiner/instances"
        output_dir = "./ommx_output"
        batch_process_instances(instances_dir, output_dir)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()