# Dataset Overview

OMMX Quantum Benchmarks provides access to optimization benchmark datasets converted to OMMX format. This page describes the current status of available dataset categories.

**Current Sources**: The initial release includes selected datasets from QOBLIB, with framework designed for expansion to additional benchmark sources in the future.

## Market Split (`01_marketsplit`)

**Problem Type**: Market segmentation optimization  
**Models**: Binary linear, Binary unconstrained  
**Instances**: 160+ instances across different sizes

Market split problems involve partitioning markets to optimize various objectives while satisfying constraints.

```python
from ommx_quantum_benchmarks.qoblib import Marketsplit

dataset = Marketsplit()
print(f"Available models: {dataset.model_names}")
# ['binary_linear', 'binary_unconstrained']
```

## Labs (`02_labs`)

**Problem Type**: Laboratory assignment problems  
**Models**: Integer, Quadratic unconstrained  
**Instances**: 99 instances (labs002 - labs100)

Laboratory problems typically involve resource allocation and scheduling in laboratory environments.

```python
from ommx_quantum_benchmarks.qoblib import Labs

dataset = Labs()
print(f"Instance range: {min(dataset.available_instances['integer'])} to {max(dataset.available_instances['integer'])}")
```

## Other Dataset Categories

The following dataset categories are defined in the framework but currently contain no instances. These represent problem types that may be expanded in future releases:

- **Birkhoff** (`03_birkhoff`) - Integer linear problems related to doubly stochastic matrices
- **Steiner** (`04_steiner`) - Integer linear Steiner tree problems
- **Sports** (`05_sports`) - Mixed integer linear sports scheduling problems
- **Portfolio** (`06_portfolio`) - Binary quadratic and quadratic unconstrained portfolio optimization
- **Independent Set** (`07_independent_set`) - Binary linear and unconstrained graph problems
- **Network** (`08_network`) - Integer linear network optimization
- **Routing** (`09_routing`) - Integer linear vehicle routing problems  
- **Topology** (`10_topology`) - Network topology problems with multiple formulations

**Note**: These datasets can be instantiated but will return empty instance lists. Check the `available_instances` property to see current availability.

## Current Status Summary

| Dataset | Models | Instance Count | Status |
|---------|--------|----------------|---------|
| Marketsplit | 2 | 160+ | ✅ Available |
| Labs | 2 | 99 | ✅ Available |
| Birkhoff | 1 | 0 | 🚧 Defined, no instances |
| Steiner | 1 | 0 | 🚧 Defined, no instances |
| Sports | 1 | 0 | 🚧 Defined, no instances |
| Portfolio | 2 | 0 | 🚧 Defined, no instances |
| IndependentSet | 2 | 0 | 🚧 Defined, no instances |
| Network | 1 | 0 | 🚧 Defined, no instances |
| Routing | 1 | 0 | 🚧 Defined, no instances |
| Topology | 3 | 0 | 🚧 Defined, no instances |

**Legend**: 
- ✅ Available: Instances have been converted and are accessible
- 🚧 Defined, no instances: Dataset classes exist but no instances are currently available