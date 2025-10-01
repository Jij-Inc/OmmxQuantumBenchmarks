# OMMX Quantum Benchmarks
OMMX Quantum Benchmarks provides access to quantum optimization benchmark datasets in [OMMX](https://jij-inc.github.io/ommx/en/introduction.html) format for easier integration with quantum and classical optimization workflows.

Documentation: https://jij-inc.github.io/OmmxQuantumBenchmarks/

## Quick Start
### Installation

```bash
# Clone and install
git clone https://github.com/Jij-Inc/OmmxQuantumBenchmarks.git
cd OmmxQuantumBenchmarks
pip install -e .
```

### Basic Usage

```python
from ommx_quantum_benchmarks.qoblib import Labs

# Load a dataset
dataset = Labs()
instance, solution = dataset("integer", "labs002")

# Evaluate the solution
evaluated = instance.evaluate(solution.state)
print(f"Objective: {evaluated.objective}, Feasible: {evaluated.feasible}")
```

You should get the following result.

```bash
Objective: 1.0, Feasible: True
```

The `instance` is `ommx.v1.Instance`. OMMX (Open Mathematical prograMming eXchange) is an open data format and SDK designed to simplify data exchange between software and people when applying mathematical optimization to real-world problems. For more details about OMMX, please have a look at the documentation: [Documentation](https://jij-inc.github.io/ommx/en/introduction.html).

## Datasets
Note that, although this repository currently contains only one dataset, QOBLIB, we are planning to add more datasets for quantum optimisation in OMMX format.

### QOBLIB
QOBLIB stands for Quantum Optimization Benchmarking Library. In this repository we provide instance data given in the original QOBLIB repository in `ommx` format leveraging the power of Github Container Registry. Note that currently, this directory does not provide all datasets, such as the fourth dataset Steiner Tree Packing pointed out in [issue 8](https://github.com/Jij-Inc/OMMX-OBLIB/issues/8). One can see which instance data are available accessing `available_instances` property.

#### Data Attribution
This project includes data derived from [QOBLIB - Quantum Optimization Benchmarking Library](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library):
- Original authors: Thorsten Koch, David E. Bernal Neira, Ying Chen, Giorgio Cortiana, Daniel J. Egger, Raoul Heese, Narendra N. Hegade, Alejandro Gomez Cadavid, Rhea Huang, Toshinari Itoko, Thomas Kleinert, Pedro Maciel Xavier, Naeimeh Mohseni, Jhon A. Montanez-Barrera, Koji Nakano, Giacomo Nannicini, Corey O’Meara, Justin Pauckert, Manuel Proissl, Anurag Ramesh, Maximilian Schicker, Noriaki Shimada, Mitsuharu Takeori, Victor Valls, David Van Bulck, Stefan Woerner, and Christa Zoufal.
- License: CC BY 4.0

The instance data has been converted to `ommx` format with additional modifications if needed.

#### Best-practice for solution reporting
Please refer to the original [contribution guidelines](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/blob/main/CONTRIBUTING.md?ref_type=heads) for further information.

#### Best-practice for hardware implementation
A collection of guidelines to run quantum optimization algorithms with Qiskit on hardware that is based on superconducting qubits can be found [here](https://github.com/qiskit-community/qopt-best-practices).

