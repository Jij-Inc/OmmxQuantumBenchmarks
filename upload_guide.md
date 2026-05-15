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

## History: minto 1.x → 2.x migration (qoblib → qoblib_v2)

The QOBLIB benchmark artifacts were originally published to the `qoblib` image on GHCR using the minto 1.x storage layout (both instance and solution stored at experiment level). When this project migrated to minto 2.x, where solutions are stored only inside runs, a new image name `qoblib_v2` was introduced so that the published `ommx-quantum-benchmarks 0.0.1` users on the old layout would not be broken.

The data on `qoblib_v2` was produced as a one-time in-place migration from the existing `qoblib` package, without regenerating from the upstream QOBLIB source:

1. For every (dataset, model, instance) listed in `available_instances`, the script pulled the corresponding artifact from `ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib:<tag>` via `minto.Experiment.load_from_registry()`.
2. The instance and solution were extracted from `experiment.dataspace.experiment_datastore` (where the minto 1.x reader had placed them).
3. A fresh `minto.Experiment` was built with the 2.x layout: `experiment.log_global_instance(...)` for the instance and `with experiment.run() as run: run.log_solution(...)` for the solution.
4. The new experiment was pushed to `ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib_v2:<same-tag>` via `experiment.push_github()`.

The migration tooling (`scripts/upload_qoblib.py` and `.github/workflows/upload-qoblib-v2.yml`) was intentionally removed after the migration completed. If the same operation ever needs to be performed again (for example to recreate `qoblib_v2` from scratch), refer to git history on the `fix/support-minto-v2` branch (commits `0bb2102`, `ab50aac`, `0f213ee`) to recover the script and workflow.

Future regenerations from the upstream QOBLIB source (not from the existing GHCR data) should go through the model definitions (`model.py` / `ommx_create.py`). This path is currently broken under jijmodeling 2.x because the model files still use the 1.x API; that migration will be addressed in a follow-up PR.
