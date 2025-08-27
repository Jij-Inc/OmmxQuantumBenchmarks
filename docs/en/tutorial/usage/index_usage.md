# Usage Guide

This section provides detailed information about working with the quantum optimization benchmark datasets available through OMMX Quantum Benchmarks.

## Overview

OMMX Quantum Benchmarks provides access to optimization benchmark datasets converted to OMMX format. Currently, this includes selected datasets from the QOBLIB repository, with plans to expand to additional benchmark sources. The available datasets follow a consistent interface pattern:

1. **Dataset Classes**: Each problem category has its own class (e.g., `Marketsplit`, `Labs`, `Portfolio`)
2. **Model Types**: Each dataset supports different model formulations (e.g., binary linear, quadratic unconstrained)
3. **Instance Loading**: Access specific problem instances using the dataset call interface
4. **Solution Evaluation**: Verify and work with provided optimal or near-optimal solutions

## Key Concepts

### Datasets
A dataset represents a collection of related optimization problems. Each dataset has:
- A unique name and description
- Multiple model formulations
- Available instances for each model
- Standardized access methods

### Models
Different mathematical formulations of the same problem type:
- **Binary Linear**: Binary variables with linear constraints
- **Binary Unconstrained**: Binary variables without constraints  
- **Integer Linear**: Integer variables with linear constraints
- **Quadratic Unconstrained**: Quadratic objective without constraints
- **Mixed Integer Linear**: Mix of continuous and integer variables

### Instances
Specific problem instances within a dataset, each with:
- Unique identifier (instance name)
- Problem data in OMMX format
- Optional optimal or best-known solution

## Working with Datasets

### Basic Dataset Information

```python
from ommx_quantum_benchmarks.qoblib import Marketsplit

dataset = Marketsplit()
print(f"Name: {dataset.name}")
print(f"Models: {dataset.model_names}")  
print(f"Description: {dataset.description}")
```

### Exploring Available Instances

```python
# Get all available instances
for model, instances in dataset.available_instances.items():
    print(f"{model}: {len(instances)} instances")
    
# Check if specific instance exists
model_name = "binary_linear"
if "ms_03_050_002" in dataset.available_instances[model_name]:
    print("Instance found!")
```

### Loading Instances and Solutions

```python
# Load instance and solution
instance, solution = dataset(model_name, instance_name)

# Check if solution is available
if solution is not None:
    print(f"Solution objective: {solution.objective}")
    print(f"Solution feasible: {solution.feasible}")
else:
    print("No solution available for this instance")
```

## Next Topics

- [Basic Usage](basic_usage.md) - Common patterns and best practices
- [Dataset Overview](dataset_overview.md) - Detailed information about each dataset
- [Working with Instances](working_with_instances.md) - Advanced instance manipulation