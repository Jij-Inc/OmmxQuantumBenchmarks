from abc import ABC
from dataclasses import dataclass, field
import numbers
from typing import Final

import minto
import ommx.v1


@dataclass
class BaseDataset(ABC):
    """Base class for datasets."""

    # Define the base URL, which will be not changed in subclasses.
    base_url: Final[str] = "ghcr.io/jij-inc/ommx-oblib/ommx_datasets"
    # Define variables that are set when the class is defined.
    number: int
    name: str
    description: str
    model_names: list[str] = field(default_factory=list)
    # Define variable that is set in __post_init__.
    model_url: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set model_url based on the member variables and assert the member variables."""
        self.model_url = {
            f"{self.base_url}:{self.number:02d}-{self.name}-{model_name}": model_name
            for model_name in self.model_names
        }

        # Assert the member variables.
        meesage_prefix = "[FOR DEVELOPER] "
        assert isinstance(self.number, numbers.Integral) and self.number > 0, (
            meesage_prefix
            + f"Dataset number must be a positive integer, but got {self.number}."
        )
        assert isinstance(self.name, str) and self.name, (
            meesage_prefix
            + f"Dataset name must be a non-empty string, but got {self.name}."
        )
        assert isinstance(self.description, str) and self.description, (
            meesage_prefix
            + f"Dataset description must be a non-empty string, but got {self.description}."
        )
        assert (
            isinstance(self.model_names, list)
            and self.model_names
            and all(isinstance(model_name, str) for model_name in self.model_names)
        ), (
            meesage_prefix
            + f"Dataset model_names must be a non-empty list of strings, but got {self.model_names}."
        )
        assert self.base_url == "ghcr.io/jij-inc/ommx-oblib/ommx_datasets", (
            meesage_prefix
            + f"Dataset base_url must be 'ghcr.io/jij-inc/ommx-oblib/ommx_datasets', but got {self.base_url}."
        )

    @property
    def models(self) -> list[str]:
        """Return the models of the dataset."""
        return list(self.model_url.keys())

    @property
    def urls(self) -> list[str]:
        """Return the URLs of the dataset models."""
        return list(self.model_url.values())

    @property
    def instance_url(self, model_name: str, instance_name: str) -> str:
        """Return the URL of the instance data specified by the given model and instance names.

        Args:
            model_name (str): a model name.
            instance_name (str): an instance name.

        Returns:
            str: the URL of the instance data.
        """
        base_url = self.model_url.get(model_name)
        instance_url = f"{base_url}-{instance_name}"
        return instance_url

    def get_experiment(self, model_name: str, instance_name: str) -> minto.Experiment:
        """Get OMMX data for a specific dataset from the Github Packages.

        Args:
            model_name (str): The name of the model.
            instance_name (str): The name of the instance.

        Returns:
            minto.Experiment: The Minto experiment containing OMMX data.
        """
        instance_url = self.instance_url(
            model_name=model_name, instance_name=instance_name
        )
        experiment = minto.Experiment.load_from_registry(instance_url)
        return experiment

    def __call__(
        self, model_name: str, instance_name: str
    ) -> tuple[ommx.v1.Instance, ommx.v1.Solution]:
        """Get the OMMX instance and solution for a specific Labs dataset.

        Args:
            instance_name (str): The name of the instance.
            model (str): The model to use.

        Returns:
            tuple[ommx.v1.Instance, ommx.v1.Solution]: The OMMX instance and solution.
        """
        experiment = self.get_experiment(
            model_name=model_name, instance_name=instance_name
        )
        instance = experiment.get_instance(instance_name)
        solution = experiment.get_solution(instance_name)

        return (instance, solution)


@dataclass
class MarketSplit(BaseDataset):
    """Class representing a market split dataset."""

    number: int = 1
    name: str = "marketsplit"
    description: str = (
        "Marketsplit dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/01-marketsplit?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_linear", "binary_unconstrained"]
    )


@dataclass
class Labs(BaseDataset):
    """Class representing a labs dataset."""

    number: int = 2
    name: str = "labs"
    description: str = (
        "Labs dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/02-labs?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["integer", "quadratic_unconstrained"]
    )


@dataclass
class Birkhoff(BaseDataset):
    """Class representing a Birkhoff dataset."""

    number: int = 3
    name: str = "birkhoff"
    description: str = (
        "Birkhoff dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/03-birkhoff?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])


@dataclass
class Steiner(BaseDataset):
    """Class representing a Steiner dataset."""

    number: int = 4
    name: str = "steiner"
    description: str = (
        "Steiner dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/04-steiner?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])


@dataclass
class Sports(BaseDataset):
    """Class representing a Sports dataset."""

    number: int = 5
    name: str = "sports"
    description: str = (
        "Sports dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/05-sports?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["mixed_integer_linear"])


@dataclass
class Portfolio(BaseDataset):
    """Class representing a Portfolio dataset."""

    number: int = 6
    name: str = "portfolio"
    description: str = (
        "Portfolio dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/06-portfolio?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_quadratic", "quadratic_unconstrained"]
    )


@dataclass
class IndependentSet(BaseDataset):
    """Class representing an Independent Set dataset."""

    number: int = 7
    name: str = "independent_set"
    description: str = (
        "Independent Set dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/07-independentset?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_linear", "binary_unconstrained"]
    )


@dataclass
class Network(BaseDataset):
    """Class representing a Network dataset."""

    number: int = 8
    name: str = "network"
    description: str = (
        "Network dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/08-network?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])


@dataclass
class Routing(BaseDataset):
    """Class representing a Routing dataset."""

    number: int = 9
    name: str = "routing"
    description: str = (
        "Routing dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/09-routing?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])


@dataclass
class Topology(BaseDataset):
    """Class representing a Topology dataset."""

    number: int = 10
    name: str = "topology"
    description: str = (
        "Topology dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/10-topology?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["flow_mip", "seidel_linear", "seidel_quadratic"]
    )
