# Dataset Usage Example (using Marketsplit)

This example demonstrates how to work with datasets in OMMX Quantum Benchmarks using Marketsplit as a representative case.

**Important**: The patterns shown here apply to ALL datasets in the collection (Labs, Portfolio, Topology, etc.). Simply replace `Marketsplit` with any other dataset class name - the API is identical across all datasets.

## Dataset Overview

```python
from ommx_quantum_benchmarks.qoblib import Marketsplit

# Initialize the dataset
dataset = Marketsplit()

print(f"Dataset: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Available models: {dataset.model_names}")

# Check available instances
for model, instances in dataset.available_instances.items():
    print(f"{model}: {len(instances)} instances")
```

Output:
```
Dataset: 01_marketsplit
Available models: ['binary_linear', 'binary_unconstrained']
binary_linear: 160 instances
binary_unconstrained: 160 instances
```

## Working with Binary Linear Model

```python
# Load a specific instance
model_name = "binary_linear"
instance_name = "ms_03_050_002"

instance, solution = dataset(model_name, instance_name)

print(f"Loaded instance: {instance_name}")
print(f"Instance type: {type(instance)}")
print(f"Solution available: {solution is not None}")

if solution:
    print(f"Objective value: {solution.objective}")
    print(f"Feasible: {solution.feasible}")
    print(f"Number of variables: {len(solution.state.entries)}")
```

## Solution Verification

```python
if solution is not None:
    # Evaluate the solution using the instance
    evaluated = instance.evaluate(solution.state)
    
    print("Solution Verification:")
    print(f"Original objective: {solution.objective}")
    print(f"Evaluated objective: {evaluated.objective}")
    print(f"Objectives match: {solution.objective == evaluated.objective}")
    
    print(f"Original feasibility: {solution.feasible}")  
    print(f"Evaluated feasibility: {evaluated.feasible}")
    print(f"Feasibility matches: {solution.feasible == evaluated.feasible}")
    
    # Check state consistency
    state_match = solution.state.entries == evaluated.state.entries
    print(f"States match: {state_match}")
```

## Analyzing Multiple Instances

```python
# Analyze first 5 instances of each size category
def analyze_marketsplit_instances():
    results = []
    
    # Group instances by size (extract size from name)
    size_groups = {}
    for instance_name in dataset.available_instances["binary_linear"]:
        # Extract size info from name like "ms_03_050_002"
        parts = instance_name.split('_')
        if len(parts) >= 3:
            size_key = f"{parts[1]}_{parts[2]}"  # e.g., "03_050"
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append(instance_name)
    
    # Analyze one instance from each size group
    for size_key, instances in list(size_groups.items())[:5]:
        instance_name = instances[0]  # Take first instance of this size
        
        try:
            instance, solution = dataset("binary_linear", instance_name)
            
            if solution:
                evaluated = instance.evaluate(solution.state)
                results.append({
                    'instance': instance_name,
                    'size_category': size_key,
                    'objective': solution.objective,
                    'feasible': solution.feasible,
                    'variables': len(solution.state.entries),
                    'verification_passed': (
                        solution.objective == evaluated.objective and
                        solution.feasible == evaluated.feasible
                    )
                })
                
        except Exception as e:
            print(f"Error with {instance_name}: {e}")
    
    return results

# Run analysis
results = analyze_marketsplit_instances()

print("\\nAnalysis Results:")
print(f"{'Instance':<15} {'Size':<8} {'Variables':<10} {'Objective':<12} {'Feasible':<9} {'Verified':<9}")
print("-" * 70)

for r in results:
    print(f"{r['instance']:<15} {r['size_category']:<8} {r['variables']:<10} "
          f"{r['objective']:<12.2f} {str(r['feasible']):<9} {str(r['verification_passed']):<9}")
```

## Performance Comparison: Binary Linear vs Unconstrained

```python
import time

def compare_model_performance():
    \"\"\"Compare loading performance between binary linear and unconstrained models.\"\"\"
    test_instances = [
        "ms_03_050_002", "ms_04_050_001", "ms_05_050_001"
    ]
    
    results = {
        "binary_linear": [],
        "binary_unconstrained": []
    }
    
    for model in ["binary_linear", "binary_unconstrained"]:
        for instance_name in test_instances:
            start_time = time.time()
            try:
                instance, solution = dataset(model, instance_name)
                load_time = time.time() - start_time
                results[model].append({
                    'instance': instance_name,
                    'load_time': load_time,
                    'has_solution': solution is not None,
                    'success': True
                })
            except Exception as e:
                results[model].append({
                    'instance': instance_name,
                    'error': str(e),
                    'success': False
                })
    
    return results

# Run comparison
perf_results = compare_model_performance()

for model, results in perf_results.items():
    print(f"\\n{model.upper()} Model:")
    successful = [r for r in results if r.get('success', False)]
    if successful:
        avg_time = sum(r['load_time'] for r in successful) / len(successful)
        print(f"Average load time: {avg_time:.3f} seconds")
        print(f"Success rate: {len(successful)}/{len(results)}")
    else:
        print("No successful loads")
```

## Integration with Quantum Algorithms

```python
# Example: Preparing Marketsplit data for QAOA
def prepare_for_qaoa(instance, solution):
    \"\"\"
    Prepare Marketsplit instance for QAOA algorithm.
    This is a conceptual example - actual implementation depends on your quantum framework.
    \"\"\"
    
    if solution is None:
        print("No solution available for comparison")
        return None
    
    # Extract problem structure (conceptual)
    problem_info = {
        'num_variables': len(solution.state.entries),
        'optimal_value': solution.objective,
        'optimal_feasible': solution.feasible,
        'variable_assignments': {
            entry.id: entry.value for entry in solution.state.entries
        }
    }
    
    print(f"Prepared QAOA problem with {problem_info['num_variables']} variables")
    print(f"Target objective: {problem_info['optimal_value']}")
    
    return problem_info

# Usage
instance, solution = dataset("binary_unconstrained", "ms_03_050_002")
qaoa_problem = prepare_for_qaoa(instance, solution)
```

## Error Handling and Robustness

```python
def robust_marketsplit_loader(dataset, model_name, instance_pattern=None, max_instances=10):
    \"\"\"Robustly load Marketsplit instances with comprehensive error handling.\"\"\"
    
    # Validate model name
    if model_name not in dataset.model_names:
        raise ValueError(f"Invalid model '{model_name}'. Available: {dataset.model_names}")
    
    available_instances = dataset.available_instances[model_name]
    
    # Filter instances if pattern provided
    if instance_pattern:
        available_instances = [
            name for name in available_instances 
            if instance_pattern in name
        ]
    
    # Limit number of instances
    test_instances = available_instances[:max_instances]
    
    successful_loads = []
    failed_loads = []
    
    for instance_name in test_instances:
        try:
            instance, solution = dataset(model_name, instance_name)
            
            # Verify solution if available
            verification_ok = True
            if solution:
                try:
                    evaluated = instance.evaluate(solution.state)
                    verification_ok = (
                        solution.objective == evaluated.objective and
                        solution.feasible == evaluated.feasible
                    )
                except Exception as e:
                    verification_ok = False
                    print(f"Verification failed for {instance_name}: {e}")
            
            successful_loads.append({
                'instance_name': instance_name,
                'has_solution': solution is not None,
                'verification_ok': verification_ok,
                'objective': solution.objective if solution else None
            })
            
        except Exception as e:
            failed_loads.append({
                'instance_name': instance_name,
                'error': str(e)
            })
    
    return successful_loads, failed_loads

# Usage
successful, failed = robust_marketsplit_loader(
    dataset, 
    "binary_linear", 
    instance_pattern="ms_03",  # Only instances with "ms_03" in name
    max_instances=5
)

print(f"Successfully loaded: {len(successful)} instances")
print(f"Failed to load: {len(failed)} instances")

if successful:
    solutions_available = sum(1 for s in successful if s['has_solution'])
    verified_solutions = sum(1 for s in successful if s['verification_ok'])
    print(f"Solutions available: {solutions_available}/{len(successful)}")
    print(f"Solutions verified: {verified_solutions}/{solutions_available}")
```

This example demonstrates the key patterns for working with the Marketsplit dataset, including basic usage, solution verification, performance analysis, and robust error handling.