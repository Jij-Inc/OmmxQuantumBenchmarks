# Release Notes

## Version 0.1.0

**Initial Release - QOBLIB Dataset Collection**

### Overview
This initial release establishes OMMX Quantum Benchmarks as a collection of optimization benchmark datasets in OMMX format. The first collection includes selected datasets from the [QOBLIB (Quantum Optimization Benchmarking Library)](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library).

**Scope**: This release focuses on QOBLIB datasets, with framework designed to accommodate additional benchmark sources in future releases.

### Features
- Python API for accessing quantum optimization benchmark datasets
- OMMX format conversion and standardization
- GitHub Container Registry integration for distributed data access
- Consistent interface across all dataset categories

### Currently Available Data
**From QOBLIB Collection**:
- **Marketsplit** (01): 160+ instances with binary linear and unconstrained formulations
- **Labs** (02): 99 instances with integer and quadratic unconstrained formulations

### Attribution
All converted data is derived from the original QOBLIB repository created by Thorsten Koch, David E. Bernal Neira, and colleagues, licensed under CC BY 4.0.
