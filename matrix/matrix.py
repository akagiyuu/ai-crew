from __future__ import annotations
from functools import reduce
from typing import override
from copy import deepcopy
import operator


class Matrix:
    row_count: int
    column_count: int
    data: list[complex]

    def __init__(self, raw_data: list[list[complex]]) -> None:
        self.row_count = len(raw_data)
        self.column_count = len(raw_data[0])

        for row in raw_data:
            assert len(row) == self.column_count

        self.data = [entry for row in raw_data for entry in row]

    @staticmethod
    def from_matrix(matrix: Matrix) -> Matrix:
        out_matrix = Matrix.__new__(Matrix)
        out_matrix.data = deepcopy(matrix.data)
        out_matrix.row_count = matrix.row_count
        out_matrix.column_count = matrix.column_count

        return out_matrix

    @staticmethod
    def zero(row_count: int, column_count: int) -> Matrix:
        out_matrix = Matrix.__new__(Matrix)
        out_matrix.data = [0 for _ in range(row_count * column_count)]
        out_matrix.row_count = row_count
        out_matrix.column_count = column_count

        return out_matrix

    @staticmethod
    def identity(size: int) -> Matrix:
        out_matrix = Matrix.__new__(Matrix)
        out_matrix.data = [0 for _ in range(size * size)]
        out_matrix.row_count = size
        out_matrix.column_count = size
        for i in range(size):
            out_matrix[i, i] = 1

        return out_matrix


    @override
    def __str__(self) -> str:
        return (
            "[\n"
            + "\n".join(
                [
                    "    " + ", ".join([str(self[i, j]) for j in range(self.column_count)])
                    for i in range(self.row_count)
                ]
            )
            + "\n]"
        )

    def __setitem__(self, key: tuple[int, int], value: complex) -> None:
        self.data[key[0] * self.column_count + key[1]] = value

    def __getitem__(self, key: tuple[int, int]) -> complex:
        return self.data[key[0] * self.column_count + key[1]]

    def tranpose(self) -> "Matrix":
        out_matrix = Matrix.from_matrix(self)
        out_matrix.row_count, out_matrix.column_count = out_matrix.column_count, out_matrix.row_count

        for i in range(out_matrix.row_count):
            for j in range(out_matrix.column_count):
                out_matrix[i, j] = self[j, i]

        return out_matrix

    def __add__(self, other: "Matrix") -> "Matrix":
        assert self.column_count == other.column_count
        assert self.row_count == other.row_count

        out_matrix = Matrix.from_matrix(self)
        for i in range(self.row_count):
            for j in range(self.column_count):
                out_matrix[i, j] = self[i, j] + other[i, j]

        return out_matrix

    @staticmethod
    def add_many(matrices: list["Matrix"]) -> "Matrix":
        zero = Matrix.zero(matrices[0].row_count, matrices[0].column_count)

        return reduce(operator.add, matrices, zero)

    def __mul__(self, other: "Matrix") -> "Matrix":
        assert self.column_count == other.row_count

        out_matrix = Matrix.zero(self.row_count, other.column_count)
        for i in range(self.row_count):
            for j in range(other.column_count):
                for k in range(self.column_count):
                    out_matrix[i, j] += self[i, k] * other[k, j]

        return out_matrix

    @staticmethod
    def multiply_many(matrices: list["Matrix"]) -> "Matrix":
        identity = Matrix.identity(matrices[0].row_count)

        return reduce(operator.mul, matrices, identity)

a = Matrix([[1, 2, 3], [4, 5, 6]])
b = Matrix([[7, 8], [9, 10], [11, 12]])
c = Matrix([[13, 14], [15, 16]])
print(Matrix.multiply_many([a, b, c]))
