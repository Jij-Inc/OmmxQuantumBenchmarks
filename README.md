# OMMX-OBLIB
This directory is dedicated to distributing the dataset [QOBLIB - Quantum Optimization Benchmarking Library](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library) in `ommx` format. Note that currently, this directory does not provide all datasets, such as the fourth dataset Steiner Tree Packing pointed out in [issue 8](https://github.com/Jij-Inc/OMMX-OBLIB/issues/8). Please have a look at [OMMX-OBLIB/Packages](https://github.com/orgs/Jij-Inc/packages?repo_name=OMMX-OBLIB) to see which datasets are distributed.

## How to Download Datasets and Usage of the Datasets
### Download
You can download each dataset in `ommx` format using `misc/download_ommx.py` script. `misc/requirement.txt` provides the package information that `download_ommx.py` requires. Simply go to `misc` directory and run the following command.

```bash
python download_ommx.py \
    --dataset_names dataset_names_that_you_would_like_to_download \
    --output_dir path_to_output_dir \
    --models '{"dataset_name1": ["model1", "model2"], "dataset_name2": ...}'
```

`--dataset_name` must be chosen in `marketsplit`, `labs`, `birkhoff`, `steiner`, `sports`, `portfolio`, `independent_set`, `network`, `routing`, `topology` and `all`. If you choose `all`, then all the 10 datasets will be downloaded.

`--output_dir` is a path to the output directory.

Each dataset may have multiple modeling as follows.

- `marketsplit`: `binary_linear` and `binary_unconstrained`.
- `labs`: `integer` and `quadratic_unconstrained`.
- `birkhoff`: `integer_linear`.
- `steiner`: `integer_linear`.
- `sports`: `mixed_integer_linear`.
- `portfolio`: `binary_quadratic` and `quadratic_unconstrained`.
- `independent_set`: `binary_linear` and `binary_unconstrained`.
- `network`: `integer_linear`.
- `routing`: `integer_linear`.
- `topology`: `flow_mip`, `seidel_linear` and `seidel_quadratic`.

You can specify which models you would like to download by `--models` like `'{"labs": ["integer", "quadratic_unconstrained"]}"` if needed. If you didn't specify models, then all the models contained in the datasets specified by `--dataset_names` will be downloaded.

The table below is the summary of the arguments.

| name | optional | number | default value |
| --- | --- | --- | --- |
| `--dataset_names` | False | as many as you wish | N/A |
| `--output_dir` | True | 1 as string | `./downloaded_ommx` |
| `--models` | True | 1 as dictionary | `None` |

For instance, if you would like to download `quadratic_unconstrained` model of `labs`, then the command will be:

```bash
python download_ommx.py \
    --dataset_names labs \
    --models '{"labs": ["quadratic_unconstrained"]}'
```

### Usage
After downloading datasets, which are in `ommx` format, you can load the data using `minto`.

```python
import minto

path = "downloaded_ommx_file_path"
experiment = minto.Experiment.load_from_ommx_archive(path)

datastore = experiment.get_current_datastore()
instance_dict: dict[str, ommx.v1.Instance] = datastore.instances  # Instance Data
solution_dict: dict[str, ommx.v1.Solution] = datastore.solutions  # Solutions
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
