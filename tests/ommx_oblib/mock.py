from dataclasses import dataclass, field

from ommx_quantum_benchmarks.qoblib.qoblib import BaseDataset


@dataclass
class MockDataset(BaseDataset):
    name: str = "mock"
    description: str = "This is a mock dataset."
    model_names: list[str] = field(default_factory=lambda: ["model1", "model2"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {
            "model1": ["instance1", "instance2"],
            "model2": ["instanceA", "instanceB"],
        }
    )


@dataclass
class MockDatasetWithEmptyName(BaseDataset):
    name: str = ""
    description: str = "This is a mock dataset."
    model_names: list[str] = field(default_factory=lambda: ["model1", "model2"])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {
            "model1": ["instance1", "instance2"],
            "model2": ["instanceA", "instanceB"],
        }
    )


@dataclass
class MockDatasetWithEmptyModelNames(BaseDataset):
    name: str = "mock"
    description: str = "This is a mock dataset."
    model_names: list[str] = field(default_factory=lambda: [])
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {
            "model1": ["instance1", "instance2"],
            "model2": ["instanceA", "instanceB"],
        }
    )


@dataclass
class MockDatasetChangedBaseURL(BaseDataset):
    name: str = "mock"
    description: str = "This is a mock dataset."
    model_names: list[str] = field(default_factory=lambda: ["model1", "model2"])
    base_url: str = "custom/base/url"
    available_instances: dict[str, list[str]] = field(
        default_factory=lambda: {
            "model1": ["instance1", "instance2"],
            "model2": ["instanceA", "instanceB"],
        }
    )
