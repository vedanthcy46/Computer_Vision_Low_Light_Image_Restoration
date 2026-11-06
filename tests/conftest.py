import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lowlight.data import make_synthetic_pairs  # noqa: E402
from lowlight.dl.infer import DLEnhancer  # noqa: E402
from lowlight.dl.zero_dce import DCENet  # noqa: E402


@pytest.fixture(scope="session")
def pairs():
    return make_synthetic_pairs(3, size=128, seed=5)


@pytest.fixture(scope="session")
def dl():
    import torch
    torch.manual_seed(0)
    return DLEnhancer(model=DCENet().eval())   # random weights: tests check behaviour, not quality


@pytest.fixture
def dark(pairs):
    return pairs[0][0]
