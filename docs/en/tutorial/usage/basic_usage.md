# Basic Usage

This page covers common usage patterns and best practices when working with OMMX OBLIB.

## Dataset Instantiation

All dataset classes follow the same pattern:

```python
from ommx_quantum_benchmarks.qoblib import Marketsplit, Labs, Portfolio

# Create dataset instances
marketsplit = Marketsplit()
labs = Labs()  
portfolio = Portfolio()
```

## Accessing Dataset Properties

```python
dataset = Marketsplit()

# Basic information
print(f"Dataset ID: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Available models: {dataset.model_names}")

# Container registry information
print(f"Base URL: {dataset.base_url}")
print(f"Model URLs: {dataset.model_url}")
```

## Instance Management

### Listing Available Instances

```python
# Get all instances for all models
for model_name, instances in dataset.available_instances.items():
    print(f"Model '{model_name}': {len(instances)} instances")
    
# Get instances for specific model
binary_linear_instances = dataset.available_instances["binary_linear"]
print(f"Binary linear instances: {binary_linear_instances[:5]}")  # First 5
```

### Loading Specific Instances

```python
# Load instance and solution
model_name = "binary_linear"
instance_name = "ms_03_050_002"

try:
    instance, solution = dataset(model_name, instance_name)
    print("Successfully loaded instance and solution")
except FileNotFoundError as e:
    print(f"Instance not found: {e}")
except ValueError as e:
    print(f"Invalid model name: {e}")
```

## Working with Solutions

### Solution Validation

```python
if solution is not None:
    # Evaluate solution using the instance
    evaluated = instance.evaluate(solution.state)
    
    # Compare original and evaluated solutions
    obj_match = solution.objective == evaluated.objective
    feas_match = solution.feasible == evaluated.feasible
    state_match = solution.state.entries == evaluated.state.entries
    
    print(f"Objective values match: {obj_match}")
    print(f"Feasibility matches: {feas_match}")
    print(f"State entries match: {state_match}")
```

### Accessing Solution Data

```python
if solution is not None:
    print(f"Objective value: {solution.objective}")
    print(f"Is feasible: {solution.feasible}")
    print(f"Number of variables: {len(solution.state.entries)}")
    
    # Access variable assignments
    for entry in solution.state.entries:
        print(f"Variable {entry.id}: {entry.value}")
```

## Error Handling Best Practices

### Handle Missing Instances

```python
def safe_load_instance(dataset, model_name, instance_name):
    try:
        return dataset(model_name, instance_name)
    except FileNotFoundError:
        print(f"Instance {instance_name} not found in model {model_name}")
        # Show available alternatives
        available = dataset.available_instances.get(model_name, [])
        if available:
            print(f"Available instances: {available[:3]}...")
        return None, None
    except ValueError as e:
        print(f"Invalid model name: {e}")
        print(f"Available models: {dataset.model_names}")
        return None, None
```

### Validate Model Names

```python
def validate_model(dataset, model_name):
    if model_name not in dataset.model_names:
        raise ValueError(f"Model '{model_name}' not available. "
                        f"Choose from: {dataset.model_names}")
    return True
```

## Performance Tips

### Reuse Dataset Instances

```python
# Good: Reuse dataset instance
dataset = Marketsplit()
instances = []
for instance_name in ["ms_03_050_002", "ms_03_050_005", "ms_03_050_007"]:
    instance, solution = dataset("binary_linear", instance_name)
    instances.append((instance, solution))

# Avoid: Creating new dataset instances repeatedly
for instance_name in instance_names:
    dataset = Marketsplit()  # Inefficient!
    instance, solution = dataset("binary_linear", instance_name)
```

### Batch Processing

```python
def load_all_instances(dataset, model_name, max_instances=None):
    \"\"\"Load all available instances for a model.\"\"\"
    instances = dataset.available_instances.get(model_name, [])
    if max_instances:
        instances = instances[:max_instances]
    
    loaded = []
    for instance_name in instances:
        try:
            instance, solution = dataset(model_name, instance_name)
            loaded.append((instance_name, instance, solution))
        except Exception as e:
            print(f"Failed to load {instance_name}: {e}")
    
    return loaded
```