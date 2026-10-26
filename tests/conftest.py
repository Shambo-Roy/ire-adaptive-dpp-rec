import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptdiv.config import PopulationConfig, SimulationConfig  # noqa: E402
from adaptdiv.simulator import generate  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg() -> SimulationConfig:
    """Default design, fewer users, for fast tests."""
    return replace(SimulationConfig(), population=PopulationConfig(users_per_cell=20, train_users=100))


@pytest.fixture(scope="session")
def small_data(small_cfg):
    return generate(seed=0, cfg=small_cfg)
