"""This script uploads OMMX data to a GitHub repository using Minto."""

import argparse
import json
from pathlib import Path
import time

import minto

import datasets


DATASETS = datasets.get_all_datasets()
DATASET_NAMES = datasets.get_all_dataset_names()
DATASET_MODELS = datasets.get_all_dataset_models()


def parse_models_json(value: str) -> dict[str, str | list[str]]:
    """Parse json string to a dictionary.

    Args:
        value (str): JSON string to parse.

    Returns:
        dict[str, str | list[str]]: Parsed dictionary.
    """
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("models must be a JSON object")

        for key, val in parsed.items():
            if not isinstance(val, (str, list)):
                raise argparse.ArgumentTypeError(
                    f"models['{key}'] must be a string or list of strings"
                )
            if isinstance(val, list) and not all(isinstance(item, str) for item in val):
                raise argparse.ArgumentTypeError(
                    f"models['{key}'] list must contain only strings"
                )

        return parsed
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")


def validate_dataset_names(names: list[str]) -> None:
    """Validate if the dataset names are valid.

    Args:
        names (list[str]): List of dataset names to validate.

    Raises:
        ValueError: If any dataset name is invalid.
    """
    for name in names:
        if name not in DATASET_NAMES:
            raise ValueError(
                f"Invalid dataset name '{name}'. Valid names are: {DATASET_NAMES}."
            )


def validate_dataset_models(model_dict: dict[str, list[str]]) -> None:
    """Validate if the dataset models are valid.

    Args:
        model_dict (dict[str, list[str]]): Dictionary of dataset names and their models.

    Raises:
        ValueError: If any dataset name or model is invalid.
    """
    # Check the keys are valid dataset names.
    for dataset_name in model_dict.keys():
        if dataset_name not in DATASET_NAMES:
            raise ValueError(
                f"Invalid dataset name '{dataset_name}' is specified in models' key."
                f"Valid names are: {DATASET_NAMES}."
            )
    # Check the models are valid for each dataset.
    for dataset_name, model_list in model_dict.items():
        for model in model_list:
            valid_models = DATASET_MODELS[dataset_name]
            if model not in valid_models:
                raise ValueError(
                    f"Invalid model '{model}' for dataset '{dataset_name}'."
                    f"Valid models are: {valid_models}."
                )


def get_experiment(dataset_name: str, model_name: str) -> minto.Experiment:
    """Get OMMX data for a specific dataset from the Github Packages.

    Args:
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model.

    Returns:
        minto.Experiment: The Minto experiment containing OMMX data.
    """
    dataset_index = DATASET_NAMES.index(dataset_name)
    dataset = DATASETS[dataset_index]
    experiment = minto.Experiment.load_from_registry(dataset.model_url[model_name])
    return experiment


def download_ommx(
    dataset_names: str | list[str],
    output_dir: str,
    model_dict: dict[str, str | list[str]] | None = None,
) -> None:
    """Download Minto experiments having OMMX data from GitHub repository.

    Args:
        dataset_names (str | list[str]): The names of the dataset to download.
        output_dir (str): The directory where the downloaded data will be saved.
        model_dict (dict[str, str | list[str]] | None): Optional models to filter the datasets.

    Raises:
        ValueError: If an invalid dataset name is specified.
    """
    # Make datasett_names a list if it is a string.
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    # Convert "all" to a list of all dataset names.
    if "all" in dataset_names:
        dataset_names = DATASET_NAMES
    # Validate the dataset names.
    validate_dataset_names(dataset_names)

    # Convert None to all models specified in dataset_names.
    if model_dict is None:
        model_dict = {name: DATASET_MODELS[name] for name in dataset_names}
    # Convert model_dict to a list if it is a string.
    for dataset_name, _models in model_dict.items():
        if isinstance(_models, str):
            model_dict[dataset_name] = [_models]
    # Validate the dataset models.
    validate_dataset_models(model_dict)

    # Create the output directory if it does not exist.
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("===========================")
    print(f"Dataset names: {dataset_names}")
    print(f"Output directory: {output_dir}")
    print("===========================")

    for dataset_name in dataset_names:
        # If no models are specified for the dataset, use all models.
        model_names = model_dict.get(dataset_name, None)
        if model_names is None:
            model_names = DATASETS[dataset_name].models

        # Create a directory for the dataset.
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            print(f"Downloading {dataset_name}/{model_name} from GitHub...")
            start_time = time.time()

            # Get the Minto experiment for the dataset.
            experiment = get_experiment(
                dataset_name=dataset_name, model_name=model_name
            )
            # Save the OMMX data to the directory.
            dataset_dir = output_dir / dataset_name
            ommx_file_path = dataset_dir / f"{model_name}.ommx"
            experiment.save_as_ommx_archive(ommx_file_path)

            end_time = time.time()
            print(f"Downloaded {dataset_name} OMMX data to {ommx_file_path}")
            print(f"Download time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OMMX data from GitHub.")
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="The name of the dataset to download. Use 'all' to download all datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="downloaded_ommx",
        help="The directory where the downloaded data will be saved.",
    )
    parser.add_argument(
        "--models",
        type=parse_models_json,
        default=None,
        help='Optional JSON string specifying models for each dataset. If it\'s None, all models will be downloaded. Example: \'{"network": ["integer_linear"]}\'.',
    )
    args = parser.parse_args()

    download_ommx(
        dataset_names=args.dataset_names,
        output_dir=args.output_dir,
        model_dict=args.models,
    )
