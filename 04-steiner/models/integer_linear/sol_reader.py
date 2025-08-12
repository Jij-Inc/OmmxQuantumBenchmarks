"""Solution reader for Steiner Tree Packing Problem."""

import re


def parse_steiner_sol_file(sol_file_path: str) -> dict[str, object]:
    """Parse a Steiner Tree Packing solution file.
    
    Args:
        sol_file_path (str): Path to the solution file
        
    Returns:
        dict: Dictionary containing solution variables and metadata
    """
    
    solution_data = {
        'objective': None,
        'x_vars': {},  # Flow variables x[a,t] 
        'y_vars': {},  # Arc usage variables y[a,k]
        'total_vars': 0
    }
    
    with open(sol_file_path, 'r') as file:
        content = file.read().strip()
        
        # Parse objective value (if present)
        obj_match = re.search(r'objective[:\s]+([0-9.-]+)', content, re.IGNORECASE)
        if obj_match:
            solution_data['objective'] = float(obj_match.group(1))
        
        # Parse variable assignments
        # Look for patterns like: x_a_t = value or y_a_k = value
        var_patterns = [
            r'x_(\d+)_(\d+)\s*=\s*([0-9.-]+)',  # x variables
            r'y_(\d+)_(\d+)\s*=\s*([0-9.-]+)'   # y variables
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Try to parse x variables
            x_match = re.search(var_patterns[0], line)
            if x_match:
                arc_idx = int(x_match.group(1))
                term_idx = int(x_match.group(2))
                value = float(x_match.group(3))
                solution_data['x_vars'][(arc_idx, term_idx)] = value
                solution_data['total_vars'] += 1
                continue
            
            # Try to parse y variables
            y_match = re.search(var_patterns[1], line)
            if y_match:
                arc_idx = int(y_match.group(1))
                net_idx = int(y_match.group(2))
                value = float(y_match.group(3))
                solution_data['y_vars'][(arc_idx, net_idx)] = value
                solution_data['total_vars'] += 1
                continue
    
    return solution_data


def convert_steiner_solution_to_jijmodeling_format(
    solution_data: dict[str, object], 
    instance_data: dict[str, object]
) -> dict[str, object]:
    """Convert parsed Steiner solution data to JijModeling variable format.
    
    Args:
        solution_data: Parsed solution data from parse_steiner_sol_file
        instance_data: Instance data containing problem structure
        
    Returns:
        Dictionary in JijModeling format for solution evaluation
    """
    
    jm_solution = {}
    
    # Convert x variables (flow variables)
    num_arcs = len(instance_data['A'])
    num_terminals = len(instance_data['T'])
    
    # Initialize x array
    x_values = [[0.0 for _ in range(num_terminals)] for _ in range(num_arcs)]
    for (arc_idx, term_idx), value in solution_data['x_vars'].items():
        if 0 <= arc_idx < num_arcs and 0 <= term_idx < num_terminals:
            x_values[arc_idx][term_idx] = value
    
    jm_solution['x'] = x_values
    
    # Convert y variables (arc usage variables)  
    num_nets = instance_data['num_nets']
    
    # Initialize y array
    y_values = [[0.0 for _ in range(num_nets)] for _ in range(num_arcs)]
    for (arc_idx, net_idx), value in solution_data['y_vars'].items():
        if 0 <= arc_idx < num_arcs and 0 <= net_idx < num_nets:
            y_values[arc_idx][net_idx] = value
    
    jm_solution['y'] = y_values
    
    return jm_solution


def read_steiner_solution_file(
    sol_file_path: str, 
    instance_data: dict[str, object]
) -> dict[str, object]:
    """Complete solution reading pipeline for Steiner Tree Packing problems.
    
    Args:
        sol_file_path: Path to the solution file
        instance_data: Instance data from dat_reader
        
    Returns:
        Solution in JijModeling format ready for evaluation
    """
    
    # Parse the solution file
    solution_data = parse_steiner_sol_file(sol_file_path)
    
    # Convert to JijModeling format
    jm_solution = convert_steiner_solution_to_jijmodeling_format(solution_data, instance_data)
    
    return jm_solution


if __name__ == "__main__":
    # Test with example solution file
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from dat_reader import load_steiner_instance
    
    # Example usage
    instance_path = "../../instances/stp_s020_l2_t3_h2_rs24098"
    sol_path = os.path.join(instance_path, "sol.txt")
    
    if os.path.exists(sol_path):
        try:
            # Load instance data
            instance_data = load_steiner_instance(instance_path)
            
            # Read solution
            solution = read_steiner_solution_file(sol_path, instance_data)
            
            print("Solution reading test:")
            print(f"- X variables shape: {len(solution['x'])} x {len(solution['x'][0])}")
            print(f"- Y variables shape: {len(solution['y'])} x {len(solution['y'][0])}")
            
            # Count non-zero variables
            x_nonzero = sum(1 for row in solution['x'] for val in row if val > 0.5)
            y_nonzero = sum(1 for row in solution['y'] for val in row if val > 0.5)
            print(f"- Non-zero X variables: {x_nonzero}")
            print(f"- Non-zero Y variables: {y_nonzero}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Solution file not found: {sol_path}")