from typing import Final

IMAGE_NAME: Final[str] = "qoblib"
BASE_URL: Final[str] = f"ghcr.io/jij-inc/ommxquantumbenchmarks/{IMAGE_NAME}"


def get_instance_tag(dataset_name: str, model_name: str, instance_name: str) -> str:
    """Get the instance tag from dataset name, model name, and instance name.
    For instance, if dataset_name is "02_labs", model_name is "integer", and instance_name is "labs002",
    the instance tag will be "02_labs-integer-labs002".

    Args:
        dataset_name (str): the dataset name
        model_name (str): the model name
        instance_name (str): the instance name

    Returns:
        str: the instance tag
    """
    return f"{dataset_name}-{model_name}-{instance_name}"
