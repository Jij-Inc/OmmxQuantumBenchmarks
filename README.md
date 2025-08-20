# OMMX-OBLIB
This directory is dedicated to distributing the dataset [QOBLIB - Quantum Optimization Benchmarking Library](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library) in `ommx` format. Note that currently, this directory does not provide all datasets, such as the fourth dataset Steiner Tree Packing pointed out in [issue 8](https://github.com/Jij-Inc/OMMX-OBLIB/issues/8). Please have a look at [OMMX-OBLIB/Packages](https://github.com/orgs/Jij-Inc/packages?repo_name=OMMX-OBLIB) to see which datasets are distributed.

## How to Download Datasets
You can access each dataset from [OMMX-OBLIB/Packages](https://github.com/orgs/Jij-Inc/packages?repo_name=OMMX-OBLIB) using [Minto](https://jij-inc.github.io/minto). The code below shows an example of how to load a dataset.

```python
import minto
import ommx

image_name = "url_to_image_you_would_like_to_download"
experiment = minto.Experiment.load_from_registry(image_name)

datastore = experiment.get_current_datastore()
instance_dict: dict[str, ommx.v1.Instance] = datastore.instances
solution_dict: dict[str, ommx.v1.Solution] = datastore.solutions  # If provided
```

The same key works for both `instance_dict` and `solution_dict`. For instance, if you would like to get a solution for an instance with the key `"labs002"`, execute the following code.

```python
key = "labs002"
instance = instance_dict[key]
solution = solution_dict[key]
```

## How to Upload a Dataset
You can upload a dataset using the `./misc/upload_ommx.py` script. To upload any dataset, you need to prepare a GitHub personal access token (PAT). Moreover, this script leverages `ommx.artifact.Artifact` (through `Minto` [Ref.](https://jij-inc.github.io/minto/en/tutorials/github_push.html)). To use this upload script, you need to setup OMMX CLI first to save information. The following steps are what you need to go through.

1. Setup rust environment to use `cargo`.
2. Setup OMMX CLI: `cargo install ommx`.
3. Login OMMX: `ommx login https://ghcr.io/v2/Jij-Inc/OMMX-OBLIB --username [your_username] --password [your_PAT]`.
4. Run the script at `misc` directory:

```bash
uv run upload_ommx.py \
    --model_dir_path [target_model_path] \
    --dataset_name [dataset_name]
```

This script assumes the following directory tree.

```
target_model_path --- directory(1) --- ommx_output --- `.ommx` files
                   |- directory(2) --- ommx_output --- `.ommx` files
                   *
                   |- directory(n) --- ommx_output --- `.ommx` files
```

Each `directory(i)` will be processed as one package.

## How to Create `.ommx` files
Each dataset directory should contain a `model` directory and some sub-directories under the `model` directory. Those sub-directories should have an `ommx_create.py` script. Simply running the script `uv run ommx_create.py` creates an `ommx_output` directory and produces `.ommx` files in it.
