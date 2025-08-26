from abc import ABC
from dataclasses import dataclass, field
from typing import Final

import minto
import ommx.v1


@dataclass
class BaseDataset(ABC):
    """Base class for datasets."""

    # Define member variables that are set in subclasses.
    name: str
    description: str
    model_names: list[str] = field(default_factory=list)
    # Define the base URL, which will be not changed in subclasses.
    base_url: Final[str] = "ghcr.io/jij-inc/ommx-oblib/qoblib"
    available_instances: dict[str, list[str]] = field(default_factory=dict)
    # Define variable that is set in __post_init__.
    model_url: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set model_url based on the member variables and assert the member variables."""
        self.model_url = {
            model_name: f"{self.base_url}:{self.name}-{model_name}"
            for model_name in self.model_names
        }

        # Assert the member variables.
        meesage_prefix = "[FOR DEVELOPER] "
        assert isinstance(self.name, str) and self.name, (
            meesage_prefix
            + f"Dataset name must be a non-empty string, but got {self.name}."
        )
        assert (
            isinstance(self.model_names, list)
            and self.model_names
            and all(isinstance(model_name, str) for model_name in self.model_names)
        ), (
            meesage_prefix
            + f"Dataset model_names must be a non-empty list of strings, but got {self.model_names}."
        )
        assert self.base_url == "ghcr.io/jij-inc/ommx-oblib/qoblib", (
            meesage_prefix
            + f"Dataset base_url must be 'ghcr.io/jij-inc/ommx-oblib/qoblib', but got {self.base_url}."
        )

    def get_instance_url(self, model_name: str, instance_name: str) -> str:
        """Get the URL of the instance data specified by the given model and instance names.

        Args:
            model_name (str): a model name.
            instance_name (str): an instance name.

        Returns:
            str: the URL of the instance data.
        """
        base_url = self.model_url.get(model_name)
        return f"{base_url}-{instance_name}"

    def get_experiment(self, model_name: str, instance_name: str) -> minto.Experiment:
        """Get OMMX data for a specific dataset from the Github Packages.

        Args:
            model_name (str): The name of the model.
            instance_name (str): The name of the instance.

        Returns:
            minto.Experiment: The Minto experiment containing OMMX data.
        """
        instance_url = self.get_instance_url(
            model_name=model_name, instance_name=instance_name
        )
        # Try to load the experiment from the Github Container Registry.
        try:
            experiment = minto.Experiment.load_from_registry(instance_url)
        except RuntimeError as e:
            error_messge_1 = f"Invalid dataset name: {instance_url}. Choose from the available datasets:\n"
            error_message_2 = ""
            for model, instances in self.available_instances.items():
                error_message_2 += f"- Model: {model}, Instances: {', '.join(instances) if instances else 'All available instances'}\n"
            error_message_3 = "Or please have a look at the Package: https://github.com/Jij-Inc/OMMX-OBLIB/pkgs/container/ommx-oblib%2Fqoblib"
            error_message = f"{error_messge_1}{error_message_2}{error_message_3}"

            # If the error is 404 not found, raise FileNotFoundError with a user-friendly message.
            if "status code 404" in str(e):
                raise FileNotFoundError(error_message) from e
            elif "Invalid name" in str(e):
                raise FileNotFoundError(error_message) from e
            else:
                # Raise the original error for other cases.
                raise

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
        # Load the only instance in the experiment.
        instances = experiment.get_current_datastore().instances
        if len(instances) != 1:
            raise ValueError(
                f"Number of instances in the given experiment is not one: {len(instances)}."
            )
        else:
            instance = list(instances.values())[0]

        # Load the only solution in the experiment if it exists.
        solutions = experiment.get_current_datastore().solutions
        if len(solutions) > 1:
            raise ValueError(
                f"Number of solutions in the given experiment is more than one: {len(solutions)}."
            )
        elif len(solutions) == 1:
            solution = list(solutions.values())[0]
        else:
            solution = None

        return (instance, solution)


@dataclass
class MarketSplit(BaseDataset):
    """Class representing a market split dataset."""

    name: str = "01_marketsplit"
    description: str = (
        "Marketsplit dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/01-marketsplit?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_linear", "binary_unconstrained"]
    )
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"binary_linear": [], "binary_unconstrained": []}
    )


@dataclass
class Labs(BaseDataset):
    """Class representing a labs dataset."""

    name: str = "02_labs"
    description: str = (
        "Labs dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/02-labs?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["integer", "quadratic_unconstrained"]
    )
    available_instances: Final[dict[str, list[str]]] = field(
        default_factory=lambda: {
            "integer": [
                "labs002",
                "labs003",
                "labs004",
                "labs005",
                "labs006",
                "labs007",
                "labs008",
                "labs009",
                "labs010",
                "labs011",
                "labs012",
                "labs013",
                "labs014",
                "labs015",
                "labs016",
                "labs017",
                "labs018",
                "labs019",
                "labs020",
                "labs021",
                "labs022",
                "labs023",
                "labs024",
                "labs025",
                "labs026",
                "labs027",
                "labs028",
                "labs029",
                "labs030",
                "labs031",
                "labs032",
                "labs033",
                "labs034",
                "labs035",
                "labs036",
                "labs037",
                "labs038",
                "labs039",
                "labs040",
                "labs041",
                "labs042",
                "labs043",
                "labs044",
                "labs045",
                "labs046",
                "labs047",
                "labs048",
                "labs049",
                "labs050",
                "labs051",
                "labs052",
                "labs053",
                "labs054",
                "labs055",
                "labs056",
                "labs057",
                "labs058",
                "labs059",
                "labs060",
                "labs061",
                "labs062",
                "labs063",
                "labs064",
                "labs065",
                "labs066",
                "labs067",
                "labs068",
                "labs069",
                "labs070",
                "labs071",
                "labs072",
                "labs073",
                "labs074",
                "labs075",
                "labs076",
                "labs077",
                "labs078",
                "labs079",
                "labs080",
                "labs081",
                "labs082",
                "labs083",
                "labs084",
                "labs085",
                "labs086",
                "labs087",
                "labs088",
                "labs089",
                "labs090",
                "labs091",
                "labs092",
                "labs093",
                "labs094",
                "labs095",
                "labs096",
                "labs097",
                "labs098",
                "labs099",
                "labs100",
            ],
            "quadratic_unconstrained": [
                "labs002",
                "labs003",
                "labs004",
                "labs005",
                "labs006",
                "labs007",
                "labs008",
                "labs009",
                "labs010",
                "labs011",
                "labs012",
                "labs013",
                "labs014",
                "labs015",
                "labs016",
                "labs017",
                "labs018",
                "labs019",
                "labs020",
                "labs021",
                "labs022",
                "labs023",
                "labs024",
                "labs025",
                "labs026",
                "labs027",
                "labs028",
                "labs029",
                "labs030",
                "labs031",
                "labs032",
                "labs033",
                "labs034",
                "labs035",
                "labs036",
                "labs037",
                "labs038",
                "labs039",
                "labs040",
                "labs041",
                "labs042",
                "labs043",
                "labs044",
                "labs045",
                "labs046",
                "labs047",
                "labs048",
                "labs049",
                "labs050",
                "labs051",
                "labs052",
                "labs053",
                "labs054",
                "labs055",
                "labs056",
                "labs057",
                "labs058",
                "labs059",
                "labs060",
                "labs061",
                "labs062",
                "labs063",
                "labs064",
                "labs065",
                "labs066",
                "labs067",
                "labs068",
                "labs069",
                "labs070",
                "labs071",
                "labs072",
                "labs073",
                "labs074",
                "labs075",
                "labs076",
                "labs077",
                "labs078",
                "labs079",
                "labs080",
                "labs081",
                "labs082",
                "labs083",
                "labs084",
                "labs085",
                "labs086",
                "labs087",
                "labs088",
                "labs089",
                "labs090",
                "labs091",
                "labs092",
                "labs093",
                "labs094",
                "labs095",
                "labs096",
                "labs097",
                "labs098",
                "labs099",
                "labs100",
            ],
        }
    )


@dataclass
class Birkhoff(BaseDataset):
    """Class representing a Birkhoff dataset."""

    name: str = "03_birkhoff"
    description: str = (
        "Birkhoff dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/03-birkhoff?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"integer_linear": []}
    )


@dataclass
class Steiner(BaseDataset):
    """Class representing a Steiner dataset."""

    name: str = "04_steiner"
    description: str = (
        "Steiner dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/04-steiner?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"integer_linear": []}
    )


@dataclass
class Sports(BaseDataset):
    """Class representing a Sports dataset."""

    name: str = "05_sports"
    description: str = (
        "Sports dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/05-sports?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["mixed_integer_linear"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"mixed_integer_linear": []}
    )


@dataclass
class Portfolio(BaseDataset):
    """Class representing a Portfolio dataset."""

    name: str = "06_portfolio"
    description: str = (
        "Portfolio dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/06-portfolio?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_quadratic", "quadratic_unconstrained"]
    )
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"binary_quadratic": [], "quadratic_unconstrained": []}
    )


@dataclass
class IndependentSet(BaseDataset):
    """Class representing an Independent Set dataset."""

    name: str = "07_independent_set"
    description: str = (
        "Independent Set dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/07-independentset?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_linear", "binary_unconstrained"]
    )
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"binary_linear": [], "binary_unconstrained": []}
    )


@dataclass
class Network(BaseDataset):
    """Class representing a Network dataset."""

    name: str = "08_network"
    description: str = (
        "Network dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/08-network?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"integer_linear": []}
    )


@dataclass
class Routing(BaseDataset):
    """Class representing a Routing dataset."""

    name: str = "09_routing"
    description: str = (
        "Routing dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/09-routing?ref_type=heads."
    )
    model_names: list[str] = field(default_factory=lambda: ["integer_linear"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {"integer_linear": []}
    )


@dataclass
class Topology(BaseDataset):
    """Class representing a Topology dataset."""

    name: str = "10_topology"
    description: str = (
        "Topology dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/10-topology?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["flow_mip", "seidel_linear", "seidel_quadratic"]
    )
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {
            "flow_mip": [],
            "seidel_linear": [],
            "seidel_quadratic": [],
        }
    )
