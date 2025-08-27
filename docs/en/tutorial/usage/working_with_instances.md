# Working with Instances

This guide covers advanced techniques for working with OMMX instances and solutions.

## Understanding OMMX Format

OMMX (Optimization Model Exchange) instances contain:
- **Variables**: Decision variables with domains and bounds
- **Objective**: Function to minimize or maximize  
- **Constraints**: Feasibility conditions
- **Metadata**: Problem information and annotations

## Accessing Instance Data

### Instance Properties

```python
from ommx_quantum_benchmarks.qoblib import Marketsplit

dataset = Marketsplit()
instance, solution = dataset("binary_linear", "ms_03_050_002")

# Basic instance information
print(f"Instance type: {type(instance)}")
print(f"Has objective: {hasattr(instance, 'objective')}")
print(f"Has constraints: {hasattr(instance, 'constraints')}")
```

### Variable Information

```python
# Access instance variables (implementation depends on OMMX structure)
# This is a conceptual example - actual API may differ
if hasattr(instance, 'variables'):
    print(f"Number of variables: {len(instance.variables)}")
    for var in instance.variables[:5]:  # First 5 variables
        print(f"Variable: {var}")
```

## Solution Analysis

### Solution Structure

```python
if solution is not None:
    print(f"Objective value: {solution.objective}")
    print(f"Feasible: {solution.feasible}")  
    print(f"State entries: {len(solution.state.entries)}")
    
    # Examine variable assignments
    for i, entry in enumerate(solution.state.entries[:10]):
        print(f"Variable {entry.id} = {entry.value}")
        if i >= 9:  # Limit output
            break
```

### Solution Evaluation

```python
# Evaluate solution with the instance
evaluated = instance.evaluate(solution.state)

print("Comparison:")
print(f"Original objective: {solution.objective}")
print(f"Evaluated objective: {evaluated.objective}")
print(f"Difference: {abs(solution.objective - evaluated.objective)}")

# Check feasibility
print(f"Original feasible: {solution.feasible}")
print(f"Evaluated feasible: {evaluated.feasible}")
```

## Advanced Usage Patterns

### Batch Loading with Error Handling

```python
def load_instances_safely(dataset, model_name, instance_names):
    \"\"\"Load multiple instances with comprehensive error handling.\"\"\"
    results = {}
    errors = {}
    
    for name in instance_names:
        try:
            instance, solution = dataset(model_name, name)
            results[name] = {
                'instance': instance,
                'solution': solution,
                'has_solution': solution is not None
            }
        except Exception as e:
            errors[name] = str(e)
    
    return results, errors

# Usage
marketsplit = Marketsplit()
instance_names = ["ms_03_050_002", "ms_03_050_005", "ms_04_050_001"]
results, errors = load_instances_safely(marketsplit, "binary_linear", instance_names)

print(f"Loaded: {len(results)} instances")
print(f"Errors: {len(errors)} instances")
```

### Solution Quality Analysis

```python
def analyze_solution_quality(dataset, model_name, instance_names):
    \"\"\"Analyze solution quality across multiple instances.\"\"\"
    analysis = []
    
    for name in instance_names:
        try:
            instance, solution = dataset(model_name, name)
            if solution is None:
                continue
                
            evaluated = instance.evaluate(solution.state)
            
            # Check consistency
            obj_consistent = abs(solution.objective - evaluated.objective) < 1e-10
            feas_consistent = solution.feasible == evaluated.feasible
            
            analysis.append({
                'instance': name,
                'objective': solution.objective,
                'feasible': solution.feasible,
                'obj_consistent': obj_consistent,
                'feas_consistent': feas_consistent,
                'num_variables': len(solution.state.entries)
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    return analysis

# Usage and analysis
results = analyze_solution_quality(
    marketsplit, 
    "binary_linear", 
    marketsplit.available_instances["binary_linear"][:10]
)

# Summary statistics
total = len(results)
consistent_obj = sum(1 for r in results if r['obj_consistent'])
consistent_feas = sum(1 for r in results if r['feas_consistent'])
feasible_count = sum(1 for r in results if r['feasible'])

print(f"Total instances analyzed: {total}")
print(f"Objective consistency: {consistent_obj}/{total}")
print(f"Feasibility consistency: {consistent_feas}/{total}")
print(f"Feasible solutions: {feasible_count}/{total}")
```

### Performance Benchmarking Setup

```python
import time

def benchmark_loading_time(dataset, model_name, num_instances=10):
    \"\"\"Benchmark instance loading performance.\"\"\"
    available = dataset.available_instances[model_name]
    test_instances = available[:num_instances]
    
    times = []
    for instance_name in test_instances:
        start = time.time()
        try:
            instance, solution = dataset(model_name, instance_name)
            load_time = time.time() - start
            times.append(load_time)
        except Exception as e:
            print(f"Failed to load {instance_name}: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average loading time: {avg_time:.3f} seconds")
        print(f"Total time for {len(times)} instances: {sum(times):.3f} seconds")
        print(f"Min time: {min(times):.3f}s, Max time: {max(times):.3f}s")
    
    return times

# Benchmark different datasets
datasets = {
    'Marketsplit': Marketsplit(),
    'Labs': Labs()
}

for name, dataset in datasets.items():
    print(f"\nBenchmarking {name}:")
    if dataset.available_instances:
        model = list(dataset.available_instances.keys())[0]
        benchmark_loading_time(dataset, model, 5)
    else:
        print("No instances available")
```

## Integration with Optimization Solvers

### Preparing Data for External Solvers

```python
def extract_problem_data(instance):
    \"\"\"Extract problem data for use with external solvers.\"\"\"
    # This is conceptual - actual implementation depends on OMMX structure
    problem_data = {
        'variables': [],
        'objective': None,
        'constraints': [],
        'bounds': []
    }
    
    # Extract variable information
    # if hasattr(instance, 'variables'):
    #     for var in instance.variables:
    #         problem_data['variables'].append({
    #             'name': var.name,
    #             'type': var.type,  # binary, integer, continuous
    #             'bounds': (var.lower_bound, var.upper_bound)
    #         })
    
    return problem_data

# Usage
instance, _ = marketsplit("binary_linear", "ms_03_050_002")
# problem_data = extract_problem_data(instance)
# print(f"Extracted data: {problem_data}")
```

This advanced guide should help you work effectively with OMMX instances and solutions for research and optimization tasks.