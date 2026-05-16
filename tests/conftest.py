import pytest
from wildfire_simulator.datasets import WildfireDataset

@pytest.fixture(scope="session")
def dataset():
    return WildfireDataset()
