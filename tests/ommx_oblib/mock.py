from ommx_quantum_benchmarks.qoblib.qoblib import BaseDataset


class MockDataset(BaseDataset):
    name: str = "mock"
    description: str = "This is a mock dataset."
    model_names: list[str] = ["model1", "model2"]
    available_instances: dict[str, list[str]] = {
        "model1": ["instance1", "instance2"],
        "model2": ["instanceA", "instanceB"],
    }


class MockDatasetWithEmptyName(BaseDataset):
    name: str = ""
    description: str = "This is a mock dataset."
    model_names: list[str] = ["model1", "model2"]
    available_instances: dict[str, list[str]] = {
        "model1": ["instance1", "instance2"],
        "model2": ["instanceA", "instanceB"],
    }


class MockDatasetWithEmptyModelNames(BaseDataset):
    name: str = "mock"
    description: str = "This is a mock dataset."
    model_names: list[str] = []
    available_instances: dict[str, list[str]] = {
        "model1": ["instance1", "instance2"],
        "model2": ["instanceA", "instanceB"],
    }


class MockDatasetChangedBaseURL(BaseDataset):
    name: str = "mock"
    description: str = "This is a mock dataset."
    model_names: list[str] = ["model1", "model2"]
    base_url: str = "custom/base/url"
    available_instances: dict[str, list[str]] = {
        "model1": ["instance1", "instance2"],
        "model2": ["instanceA", "instanceB"],
    }
