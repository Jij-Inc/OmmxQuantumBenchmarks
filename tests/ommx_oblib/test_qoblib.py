import pytest

from ommx_quantum_benchmarks.qoblib.qoblib import *
from ommx_quantum_benchmarks.qoblib.definitions import BASE_URL, get_instance_tag
from .mock import *


def test_base_dataset_creation():
    """Create a mock BaseDataset instance and check its member variables.

    Check if
    - no assertion error is raised during the creation of the instance,
    """
    # - no assertion error is raised during the creation of the instance,
    dataset = MockDataset()


def test_base_dataset_creation_with_invalid_name():
    """Create a mock BaseDataset instance with an empty name.

    Check if
    - AssertionError is raised during the creation of the instance.
    """
    # - AssertionError is raised during the creation of the instance.
    with pytest.raises(AssertionError):
        MockDatasetWithEmptyName()


def test_base_dataset_creation_with_empty_model_names():
    """Create a mock BaseDataset instance with empty model_names.

    Check if
    - AssertionError is raised during the creation of the instance.
    """
    # - AssertionError is raised during the creation of the instance.
    with pytest.raises(AssertionError):
        MockDatasetWithEmptyModelNames()


def test_get_instance_url():
    """Run get_instance_url method of a mock BaseDataset instance.

    Check if
    - the returned value is str,
    - the returned value is f"{BASE_URL}:{instance_tag}".
    """
    # - the returned value is str,
    dataset = MockDataset()
    model_name = "model1"
    instance_name = "instance1"
    result = dataset.get_instance_url(model_name, instance_name)
    assert isinstance(result, str)
    # - the returned value is f"{BASE_URL}:{instance_tag}".
    expected_url = (
        f"{BASE_URL}:{get_instance_tag(dataset.name, model_name, instance_name)}"
    )
    assert result == expected_url


def test_get_instance_url_with_invalid_model_name():
    """Run get_instance_url method of a mock BaseDataset instance with an invalid model name.

    Check if
    - ValueError is raised.
    """
    # - ValueError is raised.
    dataset = MockDataset()
    with pytest.raises(ValueError):
        dataset.get_instance_url("invalid_model", "instance1")


def test_get_experiment_with_invalid_url():
    """Run get_experiment method of a mock BaseDataset instance with an invalid instance URL.

    Check if
    - FileNotFoundError is raised.
    """
    # - FileNotFoundError is raised.
    dataset = MockDataset()
    with pytest.raises(FileNotFoundError):
        dataset.get_experiment("model1", "invalid_instance")


def test_marketsplit():
    """Create a Marketsplit instance and get each instances in its available_instances.

    Check if
    - its name is "01_marketsplit",
    - its model_names is ["binary_linear", "binary_unconstrained"],
    - its available_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "01_marketsplit",
    dataset = Marketsplit()
    assert dataset.name == "01_marketsplit"
    # - its model_names is ["binary_linear", "binary_unconstrained"],
    assert dataset.model_names == ["binary_linear", "binary_unconstrained"]
    # - its available_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "binary_linear" in dataset.available_instances
    assert "binary_unconstrained" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_labs():
    """Create a Labs instance and get each instances in its available_instances.

    Check if
    - its name is "02_labs",
    - its model_names is ["integer", "quadratic_unconstrained"],
    - its available_instances is dict whose key are "integer" and "quadratic_unconstrained",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "02_labs",
    dataset = Labs()
    assert dataset.name == "02_labs"
    # - its model_names is ["integer", "quadratic_unconstrained"],
    assert dataset.model_names == ["integer", "quadratic_unconstrained"]
    # - its available_instances is dict whose key are "integer" and "quadratic_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "integer" in dataset.available_instances
    assert "quadratic_unconstrained" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_birkhoff():
    """Create a Birkhoff instance and get each instances in its available_instances.

    Check if
    - its name is "03_birkhoff",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "03_birkhoff",
    dataset = Birkhoff()
    assert dataset.name == "03_birkhoff"
    # - its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # - its available_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_steiner():
    """Create a Steiner instance and get each instances in its available_instances.

    Check if
    - its name is "04_steiner",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "04_steiner",
    dataset = Steiner()
    assert dataset.name == "04_steiner"
    # - its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # - its available_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_sports():
    """Create a Sports instance and get each instances in its available_instances.

    Check if
    - its name is "05_sports",
    - its model_names is ["mixed_integer_linear"],
    - its available_instances is dict whose key are "mixed_integer_linear",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "05_sports",
    dataset = Sports()
    assert dataset.name == "05_sports"
    # - its model_names is ["mixed_integer_linear"],
    assert dataset.model_names == ["mixed_integer_linear"]
    # - its available_instances is dict whose key are "mixed_integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "mixed_integer_linear" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)

    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_portfolio():
    """Create a Portfolio instance and get each instances in its available_instances.

    Check if
    - its name is "06_portfolio",
    - its model_names is ["binary_quadratic", "quadratic_unconstrained"],
    - its available_instances is dict whose key are "binary_quadratic" and "quadratic_unconstrained",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "06_portfolio",
    dataset = Portfolio()
    assert dataset.name == "06_portfolio"
    # - its model_names is ["binary_quadratic", "quadratic_unconstrained"],
    assert dataset.model_names == ["binary_quadratic", "quadratic_unconstrained"]
    # - its available_instances is dict whose key are "binary_quadratic" and "quadratic_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "binary_quadratic" in dataset.available_instances
    assert "quadratic_unconstrained" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_independent_set():
    """Create a IndependentSet instance and get each instances in its available_instances.

    Check if
    - its name is "07_independent_set",
    - its model_names is ["binary_linear", "binary_unconstrained"],
    - its available_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "07_independent_set",
    dataset = IndependentSet()
    assert dataset.name == "07_independent_set"
    # - its model_names is ["binary_linear", "binary_unconstrained"],
    assert dataset.model_names == ["binary_linear", "binary_unconstrained"]
    # - its available_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    assert isinstance(dataset.available_instances, dict)
    assert "binary_linear" in dataset.available_instances
    assert "binary_unconstrained" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_network():
    """Create a Network instance and get each instances in its available_instances.

    Check if
    - its name is "08_network",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "08_network",
    dataset = Network()
    assert dataset.name == "08_network"
    # - its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # - its available_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_routing():
    """Create a Routing instance and get each instances in its available_instances.

    Check if
    - its name is "09_routing",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "09_routing",
    dataset = Routing()
    assert dataset.name == "09_routing"
    # - its model_names is ["integer_linear"],
    assert dataset.model_names == ["integer_linear"]
    # - its available_instances is dict whose key are "integer_linear",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_linear" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries


def test_topology():
    """Create a Topology instance and get each instances in its available_instances.

    Check if
    - its name is "10_topology",
    - its model_names is ["flow_mip", "seidel_linear", "seidel_quadratic"],
    - its available_instances is dict whose key are "flow_mip", "seidel_linear" and "seidel_quadratic",
    - each value of its available_instances is a list of str,
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    # - its name is "10_topology",
    dataset = Topology()
    assert dataset.name == "10_topology"
    # - its model_names is ["flow_mip", "seidel_linear", "seidel_quadratic"],
    assert dataset.model_names == ["flow_mip", "seidel_linear", "seidel_quadratic"]
    # - its available_instances is dict whose key are "flow_mip", "seidel_linear" and "seidel_quadratic",
    assert isinstance(dataset.available_instances, dict)
    assert "flow_mip" in dataset.available_instances
    assert "seidel_linear" in dataset.available_instances
    assert "seidel_quadratic" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    for model_name, instances in dataset.available_instances.items():
        if instances:  # Only test if instances are available
            instance_name = instances[0]  # Test with first instance
            result = dataset(model_name, instance_name)
            # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution) using each values of its available_instances,
            assert isinstance(result, tuple)
            assert len(result) == 2
            instance, solution = result
            assert isinstance(instance, ommx.v1.Instance)
            if solution is not None:
                assert isinstance(solution, ommx.v1.Solution)
                # - the evaluated solution with its instance and solution is the same as the original solution.
                evaluated_solution = instance.evaluate(solution.state)
                assert evaluated_solution.feasible == solution.feasible
                assert evaluated_solution.objective == solution.objective
                assert evaluated_solution.state.entries == solution.state.entries
