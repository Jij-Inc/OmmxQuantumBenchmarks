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
    base_url: Final[str] = "ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib"
    # Define available instances for each model, which will be set in subclasses.
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
        message_prefix = "[FOR DEVELOPER] "
        assert isinstance(self.name, str) and self.name, (
            message_prefix
            + f"Dataset name must be a non-empty string, but got {self.name}."
        )
        assert (
            isinstance(self.model_names, list)
            and self.model_names
            and all(isinstance(model_name, str) for model_name in self.model_names)
        ), (
            message_prefix
            + f"Dataset model_names must be a non-empty list of strings, but got {self.model_names}."
        )
        assert self.base_url == "ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib", (
            message_prefix
            + f"Dataset base_url must be 'ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib', but got {self.base_url}."
        )

    def get_instance_url(self, model_name: str, instance_name: str) -> str:
        """Get the URL of the instance data specified by the given model and instance names.

        Args:
            model_name (str): a model name.
            instance_name (str): an instance name.

        Raises:
            ValueError: if the given model name is not valid.

        Returns:
            str: the URL of the instance data.
        """
        if model_name not in self.model_names:
            raise ValueError(
                f"Invalid model name: {model_name}. Choose from {self.model_names}."
            )
        base_url = self.model_url[model_name]
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
            error_messge_1 = f"Invalid instance name: {instance_url}. Choose from the available instances:\n"
            error_message_2 = ""
            for model, instances in self.available_instances.items():
                error_message_2 += f"- Model: {model}, Instances: {', '.join(instances) if instances else 'All available instances'}."
            error_message = f"{error_messge_1}{error_message_2}"

            # If the error is 404 not found, raise FileNotFoundError with a user-friendly message.
            if "status code 404" in str(e):
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
        # The uploaded instance should be only one. Thus, if this error is raised, it is a bug of the uploader.
        assert (
            len(instances) == 1
        ), f"[FOR DEVELOPER] Number of instances obtained by model_name={model_name} and instance_name={instance_name} is not one: {len(instances)}."
        instance = list(instances.values())[0]

        # Load the only solution in the experiment if it exists.
        solutions = experiment.get_current_datastore().solutions
        # The uploaded solutions should be at most one. Thus, if this error is raised, it is a bug of the uploader.
        assert (
            0 <= len(solutions) <= 1
        ), f"[FOR DEVELOER] Number of solutions obtained by model_name={model_name} and instance_name={instance_name} is more than one: {len(solutions)}."
        if len(solutions) == 1:
            solution = list(solutions.values())[0]
        else:
            solution = None

        return (instance, solution)


@dataclass
class Marketsplit(BaseDataset):
    """Class representing a market split dataset."""

    name: str = "01_marketsplit"
    description: str = (
        "Marketsplit dataset in ommx format, originally provided by https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library/-/tree/main/01-marketsplit?ref_type=heads."
    )
    model_names: list[str] = field(
        default_factory=lambda: ["binary_linear", "binary_unconstrained"]
    )
    # available_instances: dict[str, list[str]] = field(
    #     default_factory=lambda: {
    #         "binary_linear": [
    #             "ms_03_050_002",
    #             "ms_03_050_005",
    #             "ms_03_050_007",
    #             "ms_03_050_009",
    #             "ms_03_100_001",
    #             "ms_03_100_012",
    #             "ms_03_100_019",
    #             "ms_03_100_022",
    #             "ms_03_200_050",
    #             "ms_03_200_068",
    #             "ms_03_200_161",
    #             "ms_03_200_177",
    #             "ms_04_050_001",
    #             "ms_04_050_003",
    #             "ms_04_050_004",
    #             "ms_04_050_005",
    #             "ms_04_100_003",
    #             "ms_04_100_009",
    #             "ms_04_100_013",
    #             "ms_04_100_015",
    #             "ms_04_200_030",
    #             "ms_04_200_150",
    #             "ms_04_200_174",
    #             "ms_04_200_176",
    #             "ms_05_050_001",
    #             "ms_05_050_002",
    #             "ms_05_050_003",
    #             "ms_05_050_004",
    #             "ms_05_100_003",
    #             "ms_05_100_006",
    #             "ms_05_100_013",
    #             "ms_05_100_015",
    #             "ms_05_200_070",
    #             "ms_05_200_095",
    #             "ms_05_200_180",
    #             "ms_05_200_199",
    #             "ms_06_050_001",
    #             "ms_06_050_002",
    #             "ms_06_050_003",
    #             "ms_06_050_004",
    #             "ms_06_100_002",
    #             "ms_06_100_003",
    #             "ms_06_100_005",
    #             "ms_06_100_010",
    #             "ms_06_200_077",
    #             "ms_06_200_104",
    #             "ms_06_200_240",
    #             "ms_06_200_289",
    #             "ms_07_050_001",
    #             "ms_07_050_002",
    #             "ms_07_050_003",
    #             "ms_07_050_004",
    #             "ms_07_100_002",
    #             "ms_07_100_003",
    #             "ms_07_100_005",
    #             "ms_07_100_006",
    #             "ms_07_200_248",
    #             "ms_07_200_370",
    #             "ms_07_200_398",
    #             "ms_07_200_500",
    #             "ms_08_050_000",
    #             "ms_08_050_001",
    #             "ms_08_050_002",
    #             "ms_08_050_003",
    #             "ms_08_100_000",
    #             "ms_08_100_001",
    #             "ms_08_100_002",
    #             "ms_08_100_003",
    #             "ms_08_200_000",
    #             "ms_08_200_001",
    #             "ms_08_200_002",
    #             "ms_08_200_003",
    #             "ms_09_050_000",
    #             "ms_09_050_001",
    #             "ms_09_050_002",
    #             "ms_09_050_003",
    #             "ms_09_100_000",
    #             "ms_09_100_001",
    #             "ms_09_100_002",
    #             "ms_09_100_003",
    #             "ms_09_200_000",
    #             "ms_09_200_001",
    #             "ms_09_200_002",
    #             "ms_09_200_003",
    #             "ms_10_050_000",
    #             "ms_10_050_001",
    #             "ms_10_050_002",
    #             "ms_10_050_003",
    #             "ms_10_100_000",
    #             "ms_10_100_001",
    #             "ms_10_100_002",
    #             "ms_10_100_003",
    #             "ms_10_200_000",
    #             "ms_10_200_001",
    #             "ms_10_200_002",
    #             "ms_10_200_003",
    #             "ms_11_050_000",
    #             "ms_11_050_001",
    #             "ms_11_050_002",
    #             "ms_11_050_003",
    #             "ms_11_100_000",
    #             "ms_11_100_001",
    #             "ms_11_100_002",
    #             "ms_11_100_003",
    #             "ms_11_200_000",
    #             "ms_11_200_001",
    #             "ms_11_200_002",
    #             "ms_11_200_003",
    #             "ms_12_050_000",
    #             "ms_12_050_001",
    #             "ms_12_050_002",
    #             "ms_12_050_003",
    #             "ms_12_100_000",
    #             "ms_12_100_001",
    #             "ms_12_100_002",
    #             "ms_12_100_003",
    #             "ms_12_200_000",
    #             "ms_12_200_001",
    #             "ms_12_200_002",
    #             "ms_12_200_003",
    #             "ms_13_050_000",
    #             "ms_13_050_001",
    #             "ms_13_050_002",
    #             "ms_13_050_003",
    #             "ms_13_100_000",
    #             "ms_13_100_001",
    #             "ms_13_100_002",
    #             "ms_13_100_003",
    #             "ms_13_200_000",
    #             "ms_13_200_001",
    #             "ms_13_200_002",
    #             "ms_13_200_003",
    #             "ms_14_050_000",
    #             "ms_14_050_001",
    #             "ms_14_050_002",
    #             "ms_14_050_003",
    #             "ms_14_100_000",
    #             "ms_14_100_001",
    #             "ms_14_100_002",
    #             "ms_14_100_003",
    #             "ms_14_200_000",
    #             "ms_14_200_001",
    #             "ms_14_200_002",
    #             "ms_14_200_003",
    #             "ms_15_050_000",
    #             "ms_15_050_001",
    #             "ms_15_050_002",
    #             "ms_15_050_003",
    #             "ms_15_100_000",
    #             "ms_15_100_001",
    #             "ms_15_100_002",
    #             "ms_15_100_003",
    #             "ms_15_200_000",
    #             "ms_15_200_001",
    #             "ms_15_200_002",
    #             "ms_15_200_003",
    #         ],
    #         "binary_unconstrained": [
    #             "ms_03_050_002",
    #             "ms_03_050_005",
    #             "ms_03_050_007",
    #             "ms_03_050_009",
    #             "ms_03_100_001",
    #             "ms_03_100_012",
    #             "ms_03_100_019",
    #             "ms_03_100_022",
    #             "ms_03_200_050",
    #             "ms_03_200_068",
    #             "ms_03_200_161",
    #             "ms_03_200_177",
    #             "ms_04_050_001",
    #             "ms_04_050_003",
    #             "ms_04_050_004",
    #             "ms_04_050_005",
    #             "ms_04_100_003",
    #             "ms_04_100_009",
    #             "ms_04_100_013",
    #             "ms_04_100_015",
    #             "ms_04_200_030",
    #             "ms_04_200_150",
    #             "ms_04_200_174",
    #             "ms_04_200_176",
    #             "ms_05_050_001",
    #             "ms_05_050_002",
    #             "ms_05_050_003",
    #             "ms_05_050_004",
    #             "ms_05_100_003",
    #             "ms_05_100_006",
    #             "ms_05_100_013",
    #             "ms_05_100_015",
    #             "ms_05_200_070",
    #             "ms_05_200_095",
    #             "ms_05_200_180",
    #             "ms_05_200_199",
    #             "ms_06_050_001",
    #             "ms_06_050_002",
    #             "ms_06_050_003",
    #             "ms_06_050_004",
    #             "ms_06_100_002",
    #             "ms_06_100_003",
    #             "ms_06_100_005",
    #             "ms_06_100_010",
    #             "ms_06_200_077",
    #             "ms_06_200_104",
    #             "ms_06_200_240",
    #             "ms_06_200_289",
    #             "ms_07_050_001",
    #             "ms_07_050_002",
    #             "ms_07_050_003",
    #             "ms_07_050_004",
    #             "ms_07_100_002",
    #             "ms_07_100_003",
    #             "ms_07_100_005",
    #             "ms_07_100_006",
    #             "ms_07_200_248",
    #             "ms_07_200_370",
    #             "ms_07_200_398",
    #             "ms_07_200_500",
    #             "ms_08_050_000",
    #             "ms_08_050_001",
    #             "ms_08_050_002",
    #             "ms_08_050_003",
    #             "ms_08_100_000",
    #             "ms_08_100_001",
    #             "ms_08_100_002",
    #             "ms_08_100_003",
    #             "ms_08_200_000",
    #             "ms_08_200_001",
    #             "ms_08_200_002",
    #             "ms_08_200_003",
    #             "ms_09_050_000",
    #             "ms_09_050_001",
    #             "ms_09_050_002",
    #             "ms_09_050_003",
    #             "ms_09_100_000",
    #             "ms_09_100_001",
    #             "ms_09_100_002",
    #             "ms_09_100_003",
    #             "ms_09_200_000",
    #             "ms_09_200_001",
    #             "ms_09_200_002",
    #             "ms_09_200_003",
    #             "ms_10_050_000",
    #             "ms_10_050_001",
    #             "ms_10_050_002",
    #             "ms_10_050_003",
    #             "ms_10_100_000",
    #             "ms_10_100_001",
    #             "ms_10_100_002",
    #             "ms_10_100_003",
    #             "ms_10_200_000",
    #             "ms_10_200_001",
    #             "ms_10_200_002",
    #             "ms_10_200_003",
    #             "ms_11_050_000",
    #             "ms_11_050_001",
    #             "ms_11_050_002",
    #             "ms_11_050_003",
    #             "ms_11_100_000",
    #             "ms_11_100_001",
    #             "ms_11_100_002",
    #             "ms_11_100_003",
    #             "ms_11_200_000",
    #             "ms_11_200_001",
    #             "ms_11_200_002",
    #             "ms_11_200_003",
    #             "ms_12_050_000",
    #             "ms_12_050_001",
    #             "ms_12_050_002",
    #             "ms_12_050_003",
    #             "ms_12_100_000",
    #             "ms_12_100_001",
    #             "ms_12_100_002",
    #             "ms_12_100_003",
    #             "ms_12_200_000",
    #             "ms_12_200_001",
    #             "ms_12_200_002",
    #             "ms_12_200_003",
    #             "ms_13_050_000",
    #             "ms_13_050_001",
    #             "ms_13_050_002",
    #             "ms_13_050_003",
    #             "ms_13_100_000",
    #             "ms_13_100_001",
    #             "ms_13_100_002",
    #             "ms_13_100_003",
    #             "ms_13_200_000",
    #             "ms_13_200_001",
    #             "ms_13_200_002",
    #             "ms_13_200_003",
    #             "ms_14_050_000",
    #             "ms_14_050_001",
    #             "ms_14_050_002",
    #             "ms_14_050_003",
    #             "ms_14_100_000",
    #             "ms_14_100_001",
    #             "ms_14_100_002",
    #             "ms_14_100_003",
    #             "ms_14_200_000",
    #             "ms_14_200_001",
    #             "ms_14_200_002",
    #             "ms_14_200_003",
    #             "ms_15_050_000",
    #             "ms_15_050_001",
    #             "ms_15_050_002",
    #             "ms_15_050_003",
    #             "ms_15_100_000",
    #             "ms_15_100_001",
    #             "ms_15_100_002",
    #             "ms_15_100_003",
    #             "ms_15_200_000",
    #             "ms_15_200_001",
    #             "ms_15_200_002",
    #             "ms_15_200_003",
    #         ],
    #     }
    # )
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
