import pytest
import ommx.v1

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
    dataset = MockDataset()
    # 2. its base_url is "ghcr.io/jij-inc/ommx-oblib/qoblib",
    assert dataset.base_url == "ghcr.io/jij-inc/ommx-oblib/qoblib"
    # 3. its model_url is a dict,
    assert isinstance(dataset.model_url, dict)
    # 4. each item of its model_url is model_name: f"{self.base_url}:{self.name}-{model_name}".
    for model_name in dataset.model_names:
        expected_url = f"{dataset.base_url}:{dataset.name}-{model_name}"
        assert dataset.model_url[model_name] == expected_url


def test_base_dataset_creation_with_invalid_name():
    """Create a mock BaseDataset instance with an empty name.

    Check if
    1. AssertionError is raised during the creation of the instance.
    """
    # 1. AssertionError is raised during the creation of the instance.
    with pytest.raises(AssertionError):
        MockDatasetWithEmptyName()


def test_base_dataset_creation_with_empty_model_names():
    """Create a mock BaseDataset instance with empty model_names.

    Check if
    1. AssertionError is raised during the creation of the instance.
    """
    # 1. AssertionError is raised during the creation of the instance.
    with pytest.raises(AssertionError):
        MockDatasetWithEmptyModelNames()


def test_base_dataset_creation_with_changed_base_url():
    """Create a mock BaseDataset instance with a changed base_url.

    Check if
    1. AssertionError is raised during the creation of the instance.
    """
    # 1. AssertionError is raised during the creation of the instance.
    with pytest.raises(AssertionError):
        MockDatasetChangedBaseURL()


def test_get_instance_url():
    """Run get_instance_url method of a mock BaseDataset instance.

    Check if
    1. the returned value is str,
    2. the returned value is f"{self.model_url[model_name]}-{instance_name}".
    """
    # 1. the returned value is str,
    dataset = MockDataset()
    model_name = "model1"
    instance_name = "instance1"
    result = dataset.get_instance_url(model_name, instance_name)
    assert isinstance(result, str)
    # 2. the returned value is f"{self.model_url[model_name]}-{instance_name}".
    expected_url = f"{dataset.model_url[model_name]}-{instance_name}"
    assert result == expected_url


def test_get_instance_url_with_invalid_model_name():
    """Run get_instance_url method of a mock BaseDataset instance with an invalid model name.

    Check if
    1. ValueError is raised.
    """
    # 1. ValueError is raised.
    dataset = MockDataset()
    with pytest.raises(ValueError):
        dataset.get_instance_url("invalid_model", "instance1")


def test_get_experiment_with_invalid_url():
    """Run get_experiment method of a mock BaseDataset instance with an invalid instance URL.

    Check if
    1. FileNotFoundError is raised.
    """
    # 1. FileNotFoundError is raised.
    dataset = MockDataset()
    with pytest.raises(FileNotFoundError):
        dataset.get_experiment("model1", "invalid_instance")


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
    dataset = Marketsplit()
    assert dataset.name == "01_marketsplit"
    # 2. its model_names is ["binary_linear", "binary_unconstrained"],
    assert dataset.model_names == ["binary_linear", "binary_unconstrained"]
    # 3. its availabe_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "binary_linear" in dataset.available_instances
    assert "binary_unconstrained" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Labs()
    assert dataset.name == "02_labs"
    # 2. its model_names is ["integer", "quadratic_unconstrained"],
    assert dataset.model_names == ["integer", "quadratic_unconstrained"]
    # 3. its availabe_instances is dict whose key are "integer" and "quadratic_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "integer" in dataset.available_instances
    assert "quadratic_unconstrained" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Birkhoff()
    assert dataset.name == "03_birkhoff"
    # 2. its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # 3. its availabe_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Steiner()
    assert dataset.name == "04_steiner"
    # 2. its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # 3. its availabe_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Sports()
    assert dataset.name == "05_sports"
    # 2. its model_names is ["mixed_integer_linear"],
    assert dataset.model_names == ["mixed_integer_linear"]
    # 3. its availabe_instances is dict whose key are "mixed_integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "mixed_integer_linear" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Portfolio()
    assert dataset.name == "06_portfolio"
    # 2. its model_names is ["binary_quadratic", "quadratic_unconstrained"],
    assert dataset.model_names == ["binary_quadratic", "quadratic_unconstrained"]
    # 3. its availabe_instances is dict whose key are "binary_quadratic" and "quadratic_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "binary_quadratic" in dataset.available_instances
    assert "quadratic_unconstrained" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = IndependentSet()
    assert dataset.name == "07_independent_set"
    # 2. its model_names is ["binary_linear", "binary_unconstrained"],
    assert dataset.model_names == ["binary_linear", "binary_unconstrained"]
    # 3. its availabe_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "binary_linear" in dataset.available_instances
    assert "binary_unconstrained" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Network()
    assert dataset.name == "08_network"
    # 2. its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # 3. its availabe_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Routing()
    assert dataset.name == "09_routing"
    # 2. its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # 3. its availabe_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


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
    dataset = Topology()
    assert dataset.name == "10_topology"
    # 2. its model_names is ["flow_mip", "seidel_linear", "seidel_quadratic"],
    assert dataset.model_names == ["flow_mip", "seidel_linear", "seidel_quadratic"]
    # 3. its availabe_instances is dict whose key are "flow_mip", "seidel_linear" and "seidel_quadratic",
    assert isinstance(dataset.available_instances, dict)
    assert "flow_mip" in dataset.available_instances
    assert "seidel_linear" in dataset.available_instances
    assert "seidel_quadratic" in dataset.available_instances
    # 4. each value of its availabe_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # 5. the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its availabe_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # 6. the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries
