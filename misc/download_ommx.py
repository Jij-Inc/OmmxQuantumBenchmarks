"""This script uploads OMMX data to a GitHub repository using Minto."""

import argparse
import json
from pathlib import Path
import time

import minto

import datasets


DATASETS = datasets.get_all_datasets()
DATASET_NAMES = datasets.get_all_dataset_names()
DATASET_SUBITEMS = datasets.get_all_dataset_subitems()


def parse_subitems_json(value: str) -> dict[str, str | list[str]]:
    """Parse json string to a dictionary.

    Args:
        value (str): JSON string to parse.

    Returns:
        dict[str, str | list[str]]: Parsed dictionary.
    """
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("subitems must be a JSON object")

        for key, val in parsed.items():
            if not isinstance(val, (str, list)):
                raise argparse.ArgumentTypeError(
                    f"subitems['{key}'] must be a string or list of strings"
                )
            if isinstance(val, list) and not all(isinstance(item, str) for item in val):
                raise argparse.ArgumentTypeError(
                    f"subitems['{key}'] list must contain only strings"
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


def validate_dataset_subitems(subitems: dict[str, list[str]]) -> None:
    """Validate if the dataset subitems are valid.

    Args:
        subitems (dict[str, list[str]]): Dictionary of dataset names and their subitems.

    Raises:
        ValueError: If any dataset name or subitem is invalid.
    """
    # Check the keys are valid dataset names.
    for dataset_name in subitems.keys():
        if dataset_name not in DATASET_NAMES:
            raise ValueError(
                f"Invalid dataset name '{dataset_name}' is specified in subitems' key."
                f"Valid names are: {DATASET_NAMES}."
            )
    # Check the subitems are valid for each dataset.
    for dataset_name, subitem_list in subitems.items():
        for subitem in subitem_list:
            valid_subitems = DATASET_SUBITEMS[dataset_name]
            if subitem not in valid_subitems:
                raise ValueError(
                    f"Invalid subitem '{subitem}' for dataset '{dataset_name}'."
                    f"Valid subitems are: {valid_subitems}."
                )


def get_experiment(dataset_name: str, subitem_name: str) -> minto.Experiment:
    """Get OMMX data for a specific dataset from the Github Packages.

    Args:
        dataset_name (str): The name of the dataset.
        subitem_name (str): The name of the subitem.

    Returns:
        minto.Experiment: The Minto experiment containing OMMX data.
    """
    dataset_index = DATASET_NAMES.index(dataset_name)
    dataset = DATASETS[dataset_index]
    experiment = minto.Experiment.load_from_registry(dataset.subitem_url[subitem_name])
    return experiment


def download_ommx(
    dataset_names: str | list[str],
    output_dir: str,
    subitems: dict[str, str | list[str]] | None = None,
) -> None:
    """Download Minto experiments having OMMX data from GitHub repository.

    Args:
        dataset_names (str | list[str]): The names of the dataset to download.
        output_dir (str): The directory where the downloaded data will be saved.
        subitems (dict[str, str | list[str]] | None): Optional subitems to filter the datasets.

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

    # Convert None to all subitems specified in dataset_names.
    if subitems is None:
        subitems = {name: DATASET_SUBITEMS[name] for name in dataset_names}
    # Convert subitems to a list if it is a string.
    for dataset_name, _subitems in subitems.items():
        if isinstance(_subitems, str):
            subitems[dataset_name] = [_subitems]
    # Validate the dataset subitems.
    validate_dataset_subitems(subitems)

    # Create the output directory if it does not exist.
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("===========================")
    print(f"Dataset names: {dataset_names}")
    print(f"Output directory: {output_dir}")
    print("===========================")

    for dataset_name, subitem_names in subitems.items():
        # Create a directory for the dataset.
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for subitem_name in subitem_names:
            print(f"Downloading {dataset_name}/{subitem_name} from GitHub...")
            start_time = time.time()

            # Get the Minto experiment for the dataset.
            experiment = get_experiment(
                dataset_name=dataset_name, subitem_name=subitem_name
            )
            # Save the OMMX data to the directory.
            dataset_dir = output_dir / dataset_name
            ommx_file_path = dataset_dir / f"{subitem_name}.ommx"
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
        "--subitems",
        type=parse_subitems_json,
        default=None,
        help='Optional JSON string specifying subitems for each dataset. If it\'s None, all subitems will be downloaded. Example: \'{"network": ["integer_linear"]}\'.',
    )
    args = parser.parse_args()

    download_ommx(
        dataset_names=args.dataset_names,
        output_dir=args.output_dir,
        subitems=args.subitems,
    )
