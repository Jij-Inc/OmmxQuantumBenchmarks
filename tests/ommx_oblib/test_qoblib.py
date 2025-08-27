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
    # 1. its name is "01_marketsplit",
    # 2. its model_names is ["binary_linear", "binary_unconstrained"],
    # 3. its availabe_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_labs():
    """Create a Labs instance and get each instances in its avialable_instances.

    Check if
    1. its name is "02_labs",
    2. its model_names is ["integer", "quadratic_unconstrained"],
    3. its availabe_instances is dict whose key are "integer" and "quadratic_unconstrained",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "02_labs",
    # 2. its model_names is ["integer", "quadratic_unconstrained"],
    # 3. its availabe_instances is dict whose key are "integer" and "quadratic_unconstrained",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_birkhoff():
    """Create a Birkhoff instance and get each instances in its avialable_instances.

    Check if
    1. its name is "03_birkhoff",
    2. its model_names is ["integer_linear"],
    3. its availabe_instances is dict whose key are "integer_linear",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "03_birkhoff",
    # 2. its model_names is ["integer_linear"],
    # 3. its availabe_instances is dict whose key are "integer_linear",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_steiner():
    """Create a Steiner instance and get each instances in its avialable_instances.

    Check if
    1. its name is "04_steiner",
    2. its model_names is ["integer_linear"],
    3. its availabe_instances is dict whose key are "integer_linear",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "04_steiner",
    # 2. its model_names is ["integer_linear"],
    # 3. its availabe_instances is dict whose key are "integer_linear",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_sports():
    """Create a Sports instance and get each instances in its avialable_instances.

    Check if
    1. its name is "05_sports",
    2. its model_names is ["mixed_integer_linear"],
    3. its availabe_instances is dict whose key are "mixed_integer_linear",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "05_sports",
    # 2. its model_names is ["mixed_integer_linear"],
    # 3. its availabe_instances is dict whose key are "mixed_integer_linear",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_portfolio():
    """Create a Portfolio instance and get each instances in its avialable_instances.

    Check if
    1. its name is "06_portfolio",
    2. its model_names is ["binary_quadratic", "quadratic_unconstrained"],
    3. its availabe_instances is dict whose key are "binary_quadratic" and "quadratic_unconstrained",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "06_portfolio",
    # 2. its model_names is ["binary_quadratic", "quadratic_unconstrained"],
    # 3. its availabe_instances is dict whose key are "binary_quadratic" and "quadratic_unconstrained",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_independent_set():
    """Create a IndependentSet instance and get each instances in its avialable_instances.

    Check if
    1. its name is "07_independent_set",
    2. its model_names is ["binary_linear", "binary_unconstrained"],
    3. its availabe_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "07_independent_set",
    # 2. its model_names is ["binary_linear", "binary_unconstrained"],
    # 3. its availabe_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_network():
    """Create a Network instance and get each instances in its avialable_instances.

    Check if
    1. its name is "08_network",
    2. its model_names is ["integer_linear"],
    3. its availabe_instances is dict whose key are "integer_linear",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "08_network",
    # 2. its model_names is ["integer_linear"],
    # 3. its availabe_instances is dict whose key are "integer_linear",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_routing():
    """Create a Routing instance and get each instances in its avialable_instances.

    Check if
    1. its name is "09_routing",
    2. its model_names is ["integer_linear"],
    3. its availabe_instances is dict whose key are "integer_linear",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "09_routing",
    # 2. its model_names is ["integer_linear"],
    # 3. its availabe_instances is dict whose key are "integer_linear",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.


def test_topology():
    """Create a Topology instance and get each instances in its avialable_instances.

    Check if
    1. its name is "10_topology",
    2. its model_names is ["flow_mip", "seidel_linear", "seidel_quadratic"],
    3. its availabe_instances is dict whose key are "flow_mip", "seidel_linear" and "seidel_quadratic",
    4. each value of its availabe_instances is a list of str,
    5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    6. the evaluated solution with its instance and solution is the same as the original solution.
    """
    # 1. its name is "10_topology",
    # 2. its model_names is ["flow_mip", "seidel_linear", "seidel_quadratic"],
    # 3. its availabe_instances is dict whose key are "flow_mip", "seidel_linear" and "seidel_quadratic",
    # 4. each value of its availabe_instances is a list of str,
    # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
    # 6. the evaluated solution with its instance and solution is the same as the original solution.
