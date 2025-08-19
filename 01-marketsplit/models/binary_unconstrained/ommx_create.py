import os
import glob
from pathlib import Path
import jijmodeling as jm
import ommx.v1
import numpy as np
from typing import Dict, List, Tuple, Any
from ommx.artifact import ArtifactBuilder
from sol_reader import parse_sol_to_ordered_dict
from dat_reader import QOBLIBReader
from model import create_problem

def batch_process_files(dat_directory: str = "../../instances", 
                        sol_directory: str = "../../solutions", 
                        output_directory: str = "./ommx_output"):
    """
    Batch process .dat and .sol files from different directories and convert them into .ommx files.
    
    Parameters:
    - dat_directory: Path to the directory containing .dat files
    - sol_directory: Path to the directory containing .sol files
    - output_directory: Path to the directory where .ommx files will be saved
    """
    
    # Create output directory (if it does not exist)
    os.makedirs(output_directory, exist_ok=True)
    
    # Create the problem definition (shared by all instances)
    problem = create_problem()
    
    # Find all .dat files
    dat_files = glob.glob(os.path.join(dat_directory, "*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in directory {dat_directory}")
        return
    
    print(f"Found {len(dat_files)} .dat files in {dat_directory}")
    print(f"Solution files directory: {sol_directory}")
    print(f"Output directory: {output_directory}")
    print("-" * 50)
    
    processed_count = 0
    error_count = 0
    
    for dat_file in dat_files:
        try:
            # Get the base filename (without extension)
            base_name = Path(dat_file).stem
            
            # Construct possible .sol file paths in the solutions directory
            possible_sol_files = [
                os.path.join(sol_directory, f"{base_name}.opt.sol"),
                os.path.join(sol_directory, f"{base_name}.sol")
            ]
            
            sol_file = None
            for possible_sol in possible_sol_files:
                if os.path.exists(possible_sol):
                    sol_file = possible_sol
                    break
            
            if sol_file is None:
                print("Warning: Corresponding solution file not found. Tried the following:")
                for possible_sol in possible_sol_files:
                    print(f"  - {possible_sol}")
                continue
            
            print(f"Processing file pair: {dat_file} and {sol_file}")
            
            # Read the .dat file
            reader = QOBLIBReader(dat_file)
            instance_data = reader.read_dat_file()
            
            # Create an OMMX instance
            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)
            
            # Read and evaluate the solution
            try:
                solution_dict = parse_sol_to_ordered_dict(sol_file, 0)
                solution = ommx_instance.evaluate(solution_dict)
                
                print(f"Solution evaluation result: objective={solution.objective}, feasible={solution.feasible}")
            except Exception as sol_error:
                print(f"Error processing solution file: {str(sol_error)}")
                print("Skipping solution evaluation and only saving the instance...")
                solution = None
            
            # Construct the output filename
            output_filename = os.path.join(output_directory, f"{base_name}.ommx")
            
            # If the file already exists, remove it
            if os.path.exists(output_filename):
                os.remove(output_filename)
            
            # Create OMMX Artifact
            builder = ArtifactBuilder.new_archive_unnamed(output_filename)
            
            # Add the instance
            desc_instance = builder.add_instance(ommx_instance)
            
            # Add the solution (if available)
            if solution is not None:
                desc_solution = builder.add_solution(solution)
            
            # Build and save the artifact
            artifact = builder.build()
            
            print(f"Successfully created: {output_filename}")
            print("-" * 50)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing file {dat_file}: {str(e)}")
            error_count += 1
            continue

    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Number of errors: {error_count} files")
    print(f"OMMX files saved in: {output_directory}")
    print("-" * 50)

def process_single_file(dat_file: str, sol_file: str = None, 
                        dat_directory: str = "../../instances",
                        sol_directory: str = "../../solutions",
                        output_dir: str = "./ommx_output"):
    """
    Process a single .dat/.sol file pair.
    
    Parameters:
    - dat_file: Name of the .dat file (without path)
    - sol_file: Name of the .sol file (optional; will be inferred if not provided)
    - dat_directory: Directory where the .dat file is located
    - sol_directory: Directory where the .sol file is located
    - output_dir: Directory to save the output .ommx file
    """
    
    # Construct full path to the .dat file
    full_dat_path = os.path.join(dat_directory, dat_file)
    if not os.path.exists(full_dat_path):
        raise FileNotFoundError(f"Cannot find dat file: {full_dat_path}")
    
    if sol_file is None:
        base_name = Path(dat_file).stem
        
        # Attempt to find the corresponding .sol file
        possible_sol_files = [
            f"{base_name}.opt.sol",
            f"{base_name}.sol"
        ]
        
        sol_file = None
        for possible_sol in possible_sol_files:
            full_sol_path = os.path.join(sol_directory, possible_sol)
            if os.path.exists(full_sol_path):
                sol_file = possible_sol
                break
        
        if sol_file is None:
            raise FileNotFoundError("Cannot find the corresponding solution file")
    
    full_sol_path = os.path.join(sol_directory, sol_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create problem definition
    problem = create_problem()
    
    # Read and process data
    reader = QOBLIBReader(full_dat_path)
    instance_data = reader.read_dat_file()
    
    # Create OMMX instance
    ommx_instance = jm.Interpreter(instance_data).eval_problem(problem)
    
    # Read and evaluate solution
    solution_dict = parse_sol_to_ordered_dict(full_sol_path, 0)
    solution = ommx_instance.evaluate(solution_dict)
    
    # Construct output filename
    base_name = Path(dat_file).stem
    output_filename = os.path.join(output_dir, f"{base_name}.ommx")
    
    # Create OMMX Artifact
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    builder = ArtifactBuilder.new_archive_unnamed(output_filename)
    desc_instance = builder.add_instance(ommx_instance)
    desc_solution = builder.add_solution(solution)
    artifact = builder.build()
    
    print(f"Successfully created: {output_filename}")
    print(f"Solution evaluation result: objective={solution.objective}, feasible={solution.feasible}")
    return output_filename

if __name__ == "__main__":
    batch_process_files()