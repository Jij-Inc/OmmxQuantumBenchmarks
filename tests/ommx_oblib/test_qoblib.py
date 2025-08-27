import pytest

from ommx_quantum_benchmarks.qoblib.qoblib import *
from .mock import *


def test_base_dataset_creation():
    """Create a mock BaseDataset instance and check its member variables.

    Check if
    1. no assertion error is raised during the creation of the instance,
    2. its base_url is "ghcr.io/jij-inc/ommx-oblib/qoblib",
    3. its model_url is a dict,
    4. each item of its model_url is model_name: f"{self.base_url}:{self.name}-{model_name}".
    """
    # 1. no assertion error is raised during the creation of the instance,
    # 2. its base_url is "ghcr.io/jij-inc/ommx-oblib/qoblib",
    # 3. its model_url is a dict,
    # 4. each item of its model_url is model_name: f"{self.base_url}:{self.name}-{model_name}".


def test_base_dataset_creation_with_invalid_name():
    """Create a mock BaseDataset instance with an empty name.

    Check if
    1. AssertionError is raised during the creation of the instance.
    """
    # 1. AssertionError is raised during the creation of the instance.


def test_base_dataset_creation_with_empty_model_names():
    """Create a mock BaseDataset instance with empty model_names.

    Check if
    1. AssertionError is raised during the creation of the instance.
    """
    # 1. AssertionError is raised during the creation of the instance.


def test_base_dataset_creation_with_changed_base_url():
    """Create a mock BaseDataset instance with a changed base_url.

    Check if
    1. AssertionError is raised during the creation of the instance.
    """
    # 1. AssertionError is raised during the creation of the instance.


def test_get_instance_url():
    """Run get_instance_url method of a mock BaseDataset instance.

    Check if
    1. the returned value is str,
    2. the returned value is f"{self.model_url[model_name]}-{instance_name}".
    """
    # 1. the returned value is str,
    # 2. the returned value is f"{self.model_url[model_name]}-{instance_name}".


def test_get_instance_url_with_invalid_model_name():
    """Run get_instance_url method of a mock BaseDataset instance with an invalid model name.

    Check if
    1. ValueError is raised.
    """
    # 1. ValueError is raised.


def test_get_experiment_with_invalid_url():
    """Run get_experiment method of a mock BaseDataset instance with an invalid instance URL.

    Check if
    1. FileNotFoundError is raised.
    """
    # 1. RuntimeError is raised.


def test_marketsplit():
    """Create a Marketsplit instance and get each instances in its avialable_instances.

    Check if
    1. its name is "01_marketsplit",
    2. its model_names is ["binary_linear", "binary_unconstrained"],
    3. its availabe_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
