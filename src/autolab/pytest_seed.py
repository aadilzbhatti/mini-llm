"""Pytest plugin the evaluation cascade loads (`-p autolab.pytest_seed`): seed every test.

Some protected tests draw unseeded random inits/tokens (tests/test_model.py::
test_overfits_one_batch fails ~1 in 3 on the unmodified base code). A flaky gate
rejects good candidates at random, so the cascade makes every test deterministic
instead of skipping any. The owner's test files are untouched.
"""

import random

import pytest


@pytest.fixture(autouse=True)
def _autolab_seed():
    import torch

    random.seed(0)
    torch.manual_seed(0)
    yield
