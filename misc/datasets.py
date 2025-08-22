from abc import ABC
from dataclasses import dataclass, field


@dataclass
class BaseDataset(ABC):
    """Base class for datasets."""

    number: int
    name: str
    description: str
    model_url: dict[str, str] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        """Return the models of the dataset."""
        return list(self.model_url.keys())

    @property
    def urls(self) -> list[str]:
        """Return the URLs of the dataset models."""
        return list(self.model_url.values())


@dataclass
class MarketSplit(BaseDataset):
    """Class representing a market split dataset."""

    number: int = 1
    name: str = "marketsplit"
    description: str = (
        "Marketsplit dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/01-marketsplit?ref_type=heads."
    )
    model_url: dict[str, str] = field(
        default_factory=lambda: {"binary_linear": "", "binary_unconstrained": ""}
    )


@dataclass
class Labs(BaseDataset):
    """Class representing a labs dataset."""

    number: int = 2
    name: str = "labs"
    description: str = (
        "Labs dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/02-labs?ref_type=heads."
    )
    model_url: dict[str, str] = field(
        default_factory=lambda: {
            "integer": "ghcr.io/jij-inc/ommx-oblib/02_labs/integer:20250820181458",
            "quadratic_unconstrained": "ghcr.io/jij-inc/ommx-oblib/02_labs/quadratic_unconstrained:20250820180941",
        }
    )


@dataclass
class Birkhoff(BaseDataset):
    """Class representing a Birkhoff dataset."""

    number: int = 3
    name: str = "birkhoff"
    description: str = (
        "Birkhoff dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/03-birkhoff?ref_type=heads."
    )
    model_url: dict[str, str] = field(default_factory=lambda: {"integer_linear": ""})


@dataclass
class Steiner(BaseDataset):
    """Class representing a Steiner dataset."""

    number: int = 4
    name: str = "steiner"
    description: str = (
        "Steiner dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/04-steiner?ref_type=heads."
    )
    model_url: dict[str, str] = field(default_factory=lambda: {"integer_linear": ""})


@dataclass
class Sports(BaseDataset):
    """Class representing a Sports dataset."""

    number: int = 5
    name: str = "sports"
    description: str = (
        "Sports dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/05-sports?ref_type=heads."
    )
    model_url: dict[str, str] = field(
        default_factory=lambda: {"mixed_integer_linear": ""}
    )


@dataclass
class Portfolio(BaseDataset):
    """Class representing a Portfolio dataset."""

    number: int = 6
    name: str = "portfolio"
    description: str = (
        "Portfolio dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/06-portfolio?ref_type=heads."
    )
    model_url: dict[str, str] = field(
        default_factory=lambda: {
            "binary_quadratic": "",
            "quadratic_unconstrained": "",
        }
    )


@dataclass
class IndependentSet(BaseDataset):
    """Class representing an Independent Set dataset."""

    number: int = 7
    name: str = "independent_set"
    description: str = (
        "Independent Set dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/07-independentset?ref_type=heads."
    )
    model_url: dict[str, str] = field(
        default_factory=lambda: {"binary_linear": "", "binary_unconstrained": ""}
    )


@dataclass
class Network(BaseDataset):
    """Class representing a Network dataset."""

    number: int = 8
    name: str = "network"
    description: str = (
        "Network dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/08-network?ref_type=heads."
    )
    model_url: dict[str, str] = field(default_factory=lambda: {"integer_linear": ""})


@dataclass
class Routing(BaseDataset):
    """Class representing a Routing dataset."""

    number: int = 9
    name: str = "routing"
    description: str = (
        "Routing dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/09-routing?ref_type=heads."
    )
    model_url: dict[str, str] = field(default_factory=lambda: {"integer_linear": ""})


@dataclass
class Topology(BaseDataset):
    """Class representing a Topology dataset."""

    number: int = 10
    name: str = "topology"
    description: str = (
        "Topology dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/10-topology?ref_type=heads."
    )
    model_url: dict[str, str] = field(
        default_factory=lambda: {
            "flow_mip": "",
            "seidel_linear": "",
            "seidel_quadratic": "",
        }
    )


def get_all_datasets() -> list[BaseDataset]:
    """Get all datasets as a list of BaseDataset instances."""
    return [
        MarketSplit(),
        Labs(),
        Birkhoff(),
        Steiner(),
        Sports(),
        Portfolio(),
        IndependentSet(),
        Network(),
        Routing(),
        Topology(),
    ]


def get_all_dataset_names():
    """Get all dataset names as a list of strings."""
    return [dataset.name for dataset in get_all_datasets()]


def get_all_dataset_models() -> dict[str, list[str]]:
    """Get all dataset models as a list of strings."""
    return {dataset.name: dataset.models for dataset in get_all_datasets()}
