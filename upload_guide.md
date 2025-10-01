# Upload Guide
This documentation is only for those who have the right to write GitHub Packages of OmmxQuantumBenchmarks. In this documentation, the way to upload the ommx file is explained.

## QOBLIB
In `ommx_quantum_benchmarks/qoblib`, there are 10 directories starting from `01_marketsplit` to `10_topology`. Each directory corresponds to the original [qoblib-quantum-optimization-benchmarking-library](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library).

Hereby, we will talk about `01_marketsplit` as a representative example. 

### OMMX file creation

1. Copy `instances` and `solutions` from the original qoblib-quantum-optimization-benchmarking-library to right under `01_marketsplit`.
2. Change the current directory into the target model directory (in `01_marketsplit` case, either `models/binary_linear` or `models/binary_unconstrained`).
3. Run `ommx_create.py` script: `uv run ommx_create.py`.

After the execution, you must have an `ommx_output` directory in the target model directory, which contains `.ommx` files. Those `.ommx` files contain an instance and the corresponding solution.

### OMMX file upload
What you basically need to do is use the class `ommx_quantum_benchmarks.qoblib.Uploader` and run the function `push_ommx` with the target `.ommx` files. To do this, there is a notebook [notebooks/qoblib.ipynb](./notebooks/qoblib.ipynb). You can see Uploader section in the notebook. The first cell of the section is for defining the path to the target model directory as follows.

```python
# DEFINE THE PATH TO THE DIRECTORY CONTAINING THE MODELS, CHANGE IT BY YOURSELF.
models_dir_path = "./../ommx_quantum_benchmarks/qoblib/02_labs/models"
```

All you need to do is change this path to the target model directory and run all the cells of the section. Once the execution is done, you will be able to see them in [GitHub Packages](https://github.com/orgs/Jij-Inc/packages?repo_name=OmmxQuantumBenchmarks).

If you have ever uploaded the same models, you may have an error saying the package already exists. In that case and yet you would like to re-upload the models, then just go to the path that the error message tells you and remove the target `.ommx` files that you are about to upload again. Also, if the change is massive, then **you might want to think about changing the image name itself**. The image name is defined as `IMAGE_NAME` in `ommx_quantum_benchmarks/qoblib/definitions.py`. Since the all classes must use this variable, what you need to do must be only change the variable into like `qoblib_v2`.