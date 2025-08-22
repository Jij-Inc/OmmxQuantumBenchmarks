from abc import ABC
from dataclasses import dataclass


@dataclass
class BaseDataset(ABC):
    """Base class for datasets."""

    number: int
    name: str
    description: str
    subitem_url: dict[str, str]

    @property
    def subitems(self) -> list[str]:
        """Return the subitems of the dataset."""
        return list(self.subitem_url.keys())

    @property
    def urls(self) -> list[str]:
        """Return the URLs of the dataset subitems."""
        return list(self.subitem_url.values())


@dataclass
class MarketSplit(BaseDataset):
    """Class representing a market split dataset."""

    number: int = 1
    name: str = "marketsplit"
    description: str = (
        "Marketsplit dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/01-marketsplit?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"binary_linear": "", "binary_unconstrained": ""}


@dataclass
class Labs(BaseDataset):
    """Class representing a labs dataset."""

    number: int = 2
    name: str = "labs"
    description: str = (
        "Labs dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/02-labs?ref_type=heads."
    )
    subitem_url: dict[str, str] = {
        "integer": "docker pull ghcr.io/jij-inc/ommx-oblib/02_labs/integer:20250820181458",
        "quadratic_unconstrained": "docker pull ghcr.io/jij-inc/ommx-oblib/02_labs/quadratic_unconstrained:20250820180941",
    }


@dataclass
class Birkhoff(BaseDataset):
    """Class representing a Birkhoff dataset."""

    number: int = 3
    name: str = "birkhoff"
    description: str = (
        "Birkhoff dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/03-birkhoff?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"integer_linear": ""}


@dataclass
class Steiner(BaseDataset):
    """Class representing a Steiner dataset."""

    number: int = 4
    name: str = "steiner"
    description: str = (
        "Steiner dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/04-steiner?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"integer_linear": ""}


@dataclass
class Sports(BaseDataset):
    """Class representing a Sports dataset."""

    number: int = 5
    name: str = "sports"
    description: str = (
        "Sports dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/05-sports?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"mixed_integer_linear": ""}


@dataclass
class Portfolio(BaseDataset):
    """Class representing a Portfolio dataset."""

    number: int = 6
    name: str = "portfolio"
    description: str = (
        "Portfolio dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/06-portfolio?ref_type=heads."
    )
    subitem_url: dict[str, str] = {
        "binary_quadratic": "",
        "quadratic_unconstrained": "",
    }


@dataclass
class IndependentSet(BaseDataset):
    """Class representing an Independent Set dataset."""

    number: int = 7
    name: str = "independent_set"
    description: str = (
        "Independent Set dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/07-independentset?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"binary_linear": "", "binary_unconstrained": ""}


@dataclass
class Network(BaseDataset):
    """Class representing a Network dataset."""

    number: int = 8
    name: str = "network"
    description: str = (
        "Network dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/08-network?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"integer_linear": ""}


@dataclass
class Routing(BaseDataset):
    """Class representing a Routing dataset."""

    number: int = 9
    name: str = "routing"
    description: str = (
        "Routing dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/09-routing?ref_type=heads."
    )
    subitem_url: dict[str, str] = {"integer_linear": ""}


@dataclass
class Topology(BaseDataset):
    """Class representing a Topology dataset."""

    number: int = 10
    name: str = "topology"
    description: str = (
        "Topology dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/10-topology?ref_type=heads."
    )
    subitem_url: dict[str, str] = {
        "flow_mip": "",
        "seidel_linear": "",
        "seidel_quadratic": "",
    }


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


def get_all_dataset_subitems() -> dict[str, list[str]]:
    """Get all dataset subitems as a list of strings."""
    return {dataset.name: dataset.subitems for dataset in get_all_datasets()}
