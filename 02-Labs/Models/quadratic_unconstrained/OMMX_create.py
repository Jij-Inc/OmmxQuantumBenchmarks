import os
import glob
from pathlib import Path
import jijmodeling as jm
import ommx.v1
import numpy as np
from typing import Dict, List, Tuple, Any
from ommx.artifact import ArtifactBuilder
from sol_reader import parse_sol_file
from model import create_problem

def create_instance(n):
    instance_data = {
        'I': np.arange(n),                 
        'K': np.arange(n-1),
        'P': 10000,
    }
    return instance_data

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
    
    # Create the output directory if it does not already exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Build the problem definition (shared by all instances)
    problem = create_problem()
           
    processed_count = 0
    error_count = 0
    
    # Generate instance_data directly for n from 2 to 100
    for n in range(2, 101):
        try:
            # Format base_name as labs002, labs003, … labs100
            base_name = f"labs{n:03d}"

            # Look for labsXXX.opt.sol or labsXXX.sol in the solutions directory
            possible_sol_files = [
                os.path.join(sol_directory, f"{base_name}.opt.sol"),
                os.path.join(sol_directory, f"{base_name}.bst.sol"),
                os.path.join(sol_directory, f"{base_name}.sol")
            ]
            sol_file = next((p for p in possible_sol_files if os.path.exists(p)), None)
            if sol_file is None:
                print(f"Warning: Corresponding solution file for {base_name} not found.")
                continue

            print(f"Processing solution file: {sol_file}")

            # Generate instance_data using create_instanc(n)
            instance_data = create_instanc(n)

            # Create an OMMX instance
            interpreter = jm.Interpreter(instance_data)
            ommx_instance = interpreter.eval_problem(problem)

            # Read and evaluate the solution
            solution = None
            try:
                energy_dict, entries_dict, solution_dict_x = parse_sol_file(sol_file, n)
                
                rows = n    
                cols = n - 1       
                # flat index k = i*cols + j,  i = k // cols, j = k % cols
                solution_dict_z = {
                    i*cols + j: solution_dict_x[i] * solution_dict_x.get((i+j+1),0)
                    for i in range(rows)
                    for j in range(cols)
                    }
                solution_dict = solution_dict_z.copy()
                start = max(solution_dict_z.keys()) + 1
                for i, v in enumerate(solution_dict_x.values(), start=start):
                    solution_dict[i] = v

                solution = ommx_instance.evaluate(solution_dict)
                if energy_dict['Energy']==solution.objective and solution.feasible:
                    print(f"  → objective={solution.objective}, feasible={solution.feasible}")
                else:
                    print("Objective or feasible Error")
            except Exception as sol_error:
                print(f"  ! Error evaluating solution: {sol_error}")
                print("    Skipping solution evaluation and only saving the instance...")

            # Write out the .ommx artifact
            output_filename = os.path.join(output_directory, f"{base_name}.ommx")
            if os.path.exists(output_filename):
                os.remove(output_filename)

            builder = ArtifactBuilder.new_archive_unnamed(output_filename)
            builder.add_instance(ommx_instance)
            if solution is not None:
                builder.add_solution(solution)
            builder.build()

            print(f"  ✓ Successfully created: {output_filename}")
            print("-" * 50)
            processed_count += 1

        except Exception as e:
            print(f"Error processing labs n={n}: {e}")
            error_count += 1

    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Number of errors: {error_count} files")
    print(f"OMMX files saved in: {output_directory}")
    print("-" * 50)

if __name__ == "__main__":
    batch_process_files()
