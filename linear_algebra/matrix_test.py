from functools import reduce
from collections.abc import Iterator
from .matrix import Matrix
import math
import numpy as np
import pytest


@pytest.fixture
def matrices():
    return (np.random.rand(i, i) for i in range(1, 10))


@pytest.fixture()
def diagonalizable_matrices():
    return [
        [[2.92, 0.86, -1.15], [0.86, 6.51, 3.32], [-1.15, 3.32, 4.57]],
        [[1, 0], [0, 2]],
        [[3, 1], [0, 3]],
        [[2, 1], [1, 2]],
        [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
        [[0, 1], [-1, 0]],
        [[2, 0, 1], [0, 3, 0], [1, 0, 2]],
        [[7, 0], [0, -3]],
        [[1, 1], [1, 1]],
        [[5, 2], [2, 5]],
    ]


SAMPLE_COUNT = 10


@pytest.fixture
def matrices_with_dimension():
    return ([np.random.rand(i, i) for _ in range(SAMPLE_COUNT)] for i in range(1, 10))


def compare(a: Matrix, b: list[list[float]]):
    for i in range(len(b)):
        for j in range(len(b[0])):
            assert math.isclose(a[i, j], b[i][j])


def test_tranpose(matrices: Iterator[list[list[float]]]):
    for matrix in matrices:
        a = Matrix(matrix).tranpose()
        b = np.matrix.transpose(matrix)
        compare(a, b)


def test_add(matrices_with_dimension: Iterator[list[list[list[float]]]]):
    for matrices in matrices_with_dimension:
        for i in range(SAMPLE_COUNT):
            for j in range(SAMPLE_COUNT):
                a = Matrix(matrices[i]) + Matrix(matrices[j])
                b = np.array(matrices[i]) + np.array(matrices[j])
                compare(a, b)


def test_add_many(matrices_with_dimension: Iterator[list[list[list[float]]]]):
    for matrices in matrices_with_dimension:
        a = Matrix.add_many([Matrix(matrices[i]) for i in range(SAMPLE_COUNT)])
        b = sum([np.array(matrices[i]) for i in range(SAMPLE_COUNT)])
        compare(a, b)


def test_multiply(matrices_with_dimension: Iterator[list[list[list[float]]]]):
    for matrices in matrices_with_dimension:
        for i in range(SAMPLE_COUNT):
            for j in range(SAMPLE_COUNT):
                a = Matrix(matrices[i]) @ Matrix(matrices[j])
                b = np.matmul(np.array(matrices[i]), np.array(matrices[j]))
                compare(a, b)


def test_multiply_many(matrices_with_dimension: Iterator[list[list[list[float]]]]):
    for matrices in matrices_with_dimension:
        a = Matrix.multiply_many([Matrix(matrices[i]) for i in range(SAMPLE_COUNT)])
        b = reduce(
            lambda a, b: np.matmul(a, b),
            [np.array(matrices[i]) for i in range(SAMPLE_COUNT)],
        )
        compare(a, b)


def test_inverse(matrices: Iterator[list[list[float]]]):
    for matrix in matrices:
        a = Matrix(matrix).inverse()
        b = np.linalg.inv(matrix)
        compare(a, b)


def test_determinant(matrices: Iterator[list[list[float]]]):
    for matrix in matrices:
        assert math.isclose(Matrix(matrix).determinant(), np.linalg.det(matrix))


def test_rank(matrices: Iterator[list[list[float]]]):
    for matrix in matrices:
        assert math.isclose(Matrix(matrix).rank(), np.linalg.matrix_rank(matrix))


def test_eigenvalues(diagonalizable_matrices: Iterator[list[list[float]]]):
    for matrix in diagonalizable_matrices:
        a = Matrix(matrix).eigenvalues()
        b = np.linalg.eigvals(matrix)
        b.sort()
        b = b[::-1]
        for i in range(len(a)):
            assert math.isclose(a[i], b[i])


def test_frobenius_norm(matrices: Iterator[list[list[float]]]):
    for matrix in matrices:
        assert math.isclose(Matrix(matrix).frobenius_norm(), np.linalg.norm(matrix))
