from collections.abc import Iterator
from .linear_algebra import Vector
import math
import numpy as np
import pytest


@pytest.fixture
def vectors():
    return (np.random.rand(i) for i in range(1, 10))


def compare(a: Vector, b: list[float]):
    for i in range(len(b)):
        assert math.isclose(a[i], b[i])

MAX_P = 10

def test_norm(vectors: Iterator[list[float]]):
    for vector in vectors:
        for p in range(1, MAX_P):
            assert math.isclose(Vector.norm(vector, p), np.linalg.norm(vector, p))

