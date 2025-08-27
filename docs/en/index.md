# Welcome to OMMX Quantum Benchmarks

OMMX Quantum Benchmarks provides access to quantum optimization benchmark datasets in [OMMX](https://jij-inc.github.io/ommx/en/introduction.html) format for easier integration with quantum and classical optimization workflows.

## About This Project

This repository collects optimization benchmark datasets and converts them to OMMX (Optimization Model Exchange) format. Currently, the collection includes selected datasets from [QOBLIB (Quantum Optimization Benchmarking Library)](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library), with plans to expand to additional benchmark sources in the future.

**Current Status**: The initial release focuses on converting datasets from QOBLIB. Additional benchmark collections may be added in future releases.

## What's Included

- **Benchmark Collection**: Quantum optimization datasets converted to OMMX format
- **Python API**: Simple interface for accessing benchmark instances
- **GitHub Container Registry**: Distributed storage for easy access

## Currently Available Datasets

### From QOBLIB Collection

**Available with instances**:
- **Market Split** (`01_marketsplit`) - 160+ instances across binary linear and unconstrained models
- **Labs** (`02_labs`) - 99 instances for integer and quadratic unconstrained models

**Framework defined for future expansion**:
- Birkhoff, Steiner, Sports, Portfolio, Independent Set, Network, Routing, and Topology categories are implemented but may not contain instances yet

**Note**: You can check current instance availability for any dataset using the `available_instances` property.

### Future Benchmark Sources

The framework is designed to accommodate additional benchmark collections beyond QOBLIB as they become available.

## Quick Start

To get started with OMMX Quantum Benchmarks, please see the [Quick Start Guide](quickstart.ipynb) for installation instructions and basic usage examples.

## Learn More

Explore our documentation to dive deeper into the benchmark collection capabilities:

- [Quick Start Guide](quickstart.ipynb): Installation and basic usage examples
- [Usage Guide](tutorial/usage/index_usage.md): Detailed information on working with datasets and instances
- [Examples](tutorial/examples/index_examples.md): Practical examples for different problem types
- [API Reference](autoapi/index.md): Complete documentation of the Python API

## Installation

Since this package is currently in development and not yet published on PyPI, please install directly from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/Jij-Inc/OMMX-OBLIB.git
cd OMMX-OBLIB

# Install in development mode with pip
pip install -e .

# Or install in development mode with uv
uv pip install -e .
```

Alternatively, you can install directly from GitHub without cloning:

```bash
# Using pip
pip install git+https://github.com/Jij-Inc/OMMX-OBLIB.git

# Using uv
uv pip install git+https://github.com/Jij-Inc/OMMX-OBLIB.git
```

## Basic Usage

```python
from ommx_quantum_benchmarks.qoblib import Marketsplit

# Load a dataset
dataset = Marketsplit()

# Get an instance and solution
model_name = "binary_linear"
instance_name = "ms_03_050_002"
instance, solution = dataset(model_name, instance_name)

# Evaluate the solution
evaluated_solution = instance.evaluate(solution.state)
print(f"Objective value: {evaluated_solution.objective}")
print(f"Feasible: {evaluated_solution.feasible}")
```

## Attribution

This project includes data derived from [QOBLIB - Quantum Optimization Benchmarking Library](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library):
- **Original authors**: Thorsten Koch, David E. Bernal Neira, Ying Chen, Giorgio Cortiana, Daniel J. Egger, Raoul Heese, Narendra N. Hegade, Alejandro Gomez Cadavid, Rhea Huang, Toshinari Itoko, Thomas Kleinert, Pedro Maciel Xavier, Naeimeh Mohseni, Jhon A. Montanez-Barrera, Koji Nakano, Giacomo Nannicini, Corey O'Meara, Justin Pauckert, Manuel Proissl, Anurag Ramesh, Maximilian Schicker, Noriaki Shimada, Mitsuharu Takeori, Victor Valls, David Van Bulck, Stefan Woerner, and Christa Zoufal.
- **License**: CC BY 4.0

The instance data has been converted to OMMX format with additional modifications as needed.

## Support

For issues with the benchmark datasets or OMMX Quantum Benchmarks library, please file an issue on our [GitHub repository](https://github.com/Jij-Inc/OmmxQuantumBenchmarks).

For questions about the original datasets or problem formulations, please refer to the respective source repositories (e.g., [original QOBLIB repository](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library) for QOBLIB-derived datasets).