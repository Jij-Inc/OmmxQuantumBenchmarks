import pytest

from ommx_quantum_benchmarks.qoblib.qoblib import *
from ommx_quantum_benchmarks.qoblib.definitions import BASE_URL, get_instance_tag
from .mock import *


NUM_CASES = 1  # Limit the number of test cases (instances) for each instance test to reduce test time.


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
    """Create a Labs instance and check its basic properties.

    Check if
    - its name is "02_labs",
    - its model_names is ["integer", "quadratic_unconstrained"],
    - its available_instances is dict whose key are "integer" and "quadratic_unconstrained",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["integer"] is 99,
    - the length of its available_instances["quadratic_unconstrained"] is 99,
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
    # - the length of its available_instances["integer"] is 99,
    assert len(dataset.available_instances["integer"]) == 99
    # - the length of its available_instances["quadratic_unconstrained"] is 99,
    assert len(dataset.available_instances["quadratic_unconstrained"]) == 99


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in Labs().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_labs_instance(model_name, instance_name):
    """Test individual Labs instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = Labs()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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
    """Create a Birkhoff instance and check its basic properties.

    Check if
    - its name is "03_birkhoff",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["integer_linear"] is 800,
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
    # - the length of its available_instances["integer_linear"] is 800,
    assert len(dataset.available_instances["integer_linear"]) == 800


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in Birkhoff().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_birkhoff_instance(model_name, instance_name):
    """Test individual Birkhoff instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = Birkhoff()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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
    """Create a Steiner instance and check its basic properties.

    Check if
    - its name is "04_steiner",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["integer_linear"] is 31,
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
    # - the length of its available_instances["integer_linear"] is 31,
    assert len(dataset.available_instances["integer_linear"]) == 31


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in Steiner().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_steiner_instance(model_name, instance_name):
    """Test individual Steiner instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = Steiner()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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


def test_independentset():
    """Create a IndependentSet instance and check its basic properties.

    Check if
    - its name is "07_independentset",
    - its model_names is ["binary_linear", "binary_unconstrained"],
    - its available_instances is dict whose key are "binary_linear" and "binary_unconstrained",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["binary_linear"] is 42,
    - the length of its available_instances["binary_unconstrained"] is 42,
    """
    # - its name is "07_independentset",
    dataset = IndependentSet()
    assert dataset.name == "07_independentset"
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
    # - the length of its available_instances["binary_linear"] is 42,
    assert len(dataset.available_instances["binary_linear"]) == 42
    # - the length of its available_instances["binary_unconstrained"] is 42,
    assert len(dataset.available_instances["binary_unconstrained"]) == 42


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in IndependentSet().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_independentset_instance(model_name, instance_name):
    """Test individual IndependentSet instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = IndependentSet()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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
    """Create a Network instance and check its basic properties.

    Check if
    - its name is "08_network",
    - its model_names is ["integer_lp"],
    - its available_instances is dict whose key are "integer_lp",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["integer_lp"] is 20,
    """
    # - its name is "08_network",
    dataset = Network()
    assert dataset.name == "08_network"
    # - its model_names is ["integer_lp"],
    assert dataset.model_names == ["integer_lp"]
    # - its available_instances is dict whose key are "integer_lp",
    assert isinstance(dataset.available_instances, dict)
    assert "integer_lp" in dataset.available_instances
    # - each value of its available_instances is a list of str,
    for model_name, instances in dataset.available_instances.items():
        assert isinstance(instances, list)
        for instance in instances:
            assert isinstance(instance, str)
    # - the length of its available_instances["integer_lp"] is 20,
    assert len(dataset.available_instances["integer_lp"]) == 20


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in Network().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_network_instance(model_name, instance_name):
    """Test individual Network instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = Network()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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
    """Create a Routing instance and check its basic properties.

    Check if
    - its name is "09_routing",
    - its model_names is ["integer_linear"],
    - its available_instances is dict whose key are "integer_linear",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["integer_linear"] is 55,
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
    # - the length of its available_instances["integer_linear"] is 55,
    assert len(dataset.available_instances["integer_linear"]) == 55


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in Routing().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_routing_instance(model_name, instance_name):
    """Test individual Routing instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = Routing()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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
    """Create a Topology instance and check its basic properties.

    Check if
    - its name is "10_topology",
    - its model_names is ["flow_mip", "seidel_linear", "seidel_quadratic"],
    - its available_instances is dict whose key are "flow_mip", "seidel_linear" and "seidel_quadratic",
    - each value of its available_instances is a list of str,
    - the length of its available_instances["flow_mip"] is 16,
    - the length of its available_instances["seidel_linear"] is 16,
    - the length of its available_instances["seidel_quadratic"] is 16,
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
    # - the length of its available_instances["flow_mip"] is 16,
    assert len(dataset.available_instances["flow_mip"]) == 16
    # - the length of its available_instances["seidel_linear"] is 16,
    assert len(dataset.available_instances["seidel_linear"]) == 16
    # - the length of its available_instances["seidel_quadratic"] is 16,
    assert len(dataset.available_instances["seidel_quadratic"]) == 16


@pytest.mark.parametrize(
    "model_name,instance_name",
    [
        (model_name, instance_name)
        for model_name, instances in Topology().available_instances.items()
        for instance_name in instances
    ][:NUM_CASES],
)
def test_topology_instance(model_name, instance_name):
    """Test individual Topology instance.

    Check if
    - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution),
    - the evaluated solution with its instance and solution is the same as the original solution.
    """
    dataset = Topology()
    result = dataset(model_name, instance_name)
    # - the returned value of __call__ is a tuple of (ommx.v1.instance, ommx.v1.solution)
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
