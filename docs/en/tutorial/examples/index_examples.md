# Examples

This section provides practical examples for working with OMMX Quantum Benchmarks datasets.

**Note**: All datasets from the QOBLIB collection follow the same API pattern. The examples shown here using Marketsplit apply to other QOBLIB datasets (Labs, Portfolio, Topology, etc.) with only the class name and available instances differing.

## Representative Example

- [Marketsplit Example](marketsplit_example.md) - Complete usage example showing all common patterns

## Common Usage Pattern for QOBLIB Datasets

All QOBLIB datasets follow this consistent interface:

```python
from ommx_quantum_benchmarks.qoblib import DatasetName

# Instantiate any dataset
dataset = DatasetName()

# Check properties
print(f"Name: {dataset.name}")
print(f"Models: {dataset.model_names}")
print(f"Available instances: {dataset.available_instances}")

# Load instance and solution (if available)
if dataset.available_instances[model_name]:
    instance, solution = dataset(model_name, instance_name)
```

## What the Example Covers

The Marketsplit example demonstrates all common patterns that apply to QOBLIB datasets:

1. **Dataset instantiation and exploration**
2. **Instance loading and validation**
3. **Solution analysis and verification**
4. **Error handling and robustness**
5. **Performance considerations**
6. **Integration patterns for optimization workflows**

These patterns work identically across all QOBLIB dataset classes, making it easy to switch between different QOBLIB problem types while using the same code structure.