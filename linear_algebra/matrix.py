from __future__ import annotations
from functools import reduce
from typing import override
from copy import deepcopy
import math
import sys
from .vector import Vector


def sign(i: float, j: float) -> float:
    return 1 if (i + j) % 2 == 0 else -1


class Matrix:
    row_count: int
    column_count: int
    data: list[float]

    def __init__(self, raw_data: list[list[float]]) -> None:
        self.row_count = len(raw_data)
        self.column_count = len(raw_data[0])

        for row in raw_data:
            assert len(row) == self.column_count

        self.data = [entry for row in raw_data for entry in row]

    @staticmethod
    def from_data_with_dimension(
        data: list[float], row_count: int, column_count: int
    ) -> Matrix:
        out_matrix = Matrix.__new__(Matrix)
        out_matrix.data = data
        out_matrix.row_count = row_count
        out_matrix.column_count = column_count

        return out_matrix

    @staticmethod
    def from_column_vector(column: list[Vector]) -> Matrix:
        row_count = len(column[0])
        column_count = len(column)
        data: list[float] = []
        for i in range(row_count):
            for j in range(column_count):
                data.append(column[j][i])
        return Matrix.from_data_with_dimension(data, row_count, column_count)

    @staticmethod
    def from_diagonal(diagonal: list[float]) -> Matrix:
        out_matrix = Matrix.zero(len(diagonal), len(diagonal))
        for i in range(len(diagonal)):
            out_matrix[i, i] = diagonal[i]
        return out_matrix

    @staticmethod
    def from_matrix(matrix: Matrix) -> Matrix:
        return Matrix.from_data_with_dimension(
            deepcopy(matrix.data), matrix.row_count, matrix.column_count
        )

    @staticmethod
    def zero(row_count: int, column_count: int) -> Matrix:
        data = [0.0 for _ in range(row_count * column_count)]
        return Matrix.from_data_with_dimension(data, row_count, column_count)

    @staticmethod
    def identity(size: int) -> Matrix:
        data = [0.0 for _ in range(size * size)]
        for i in range(size):
            data[i * size + i] = 1
        return Matrix.from_data_with_dimension(data, size, size)

    @override
    def __str__(self) -> str:
        return (
            "[\n"
            + "\n".join(
                [
                    "    "
                    + ", ".join([str(self[i, j]) for j in range(self.column_count)])
                    for i in range(self.row_count)
                ]
            )
            + "\n]"
        )

    @override
    def __repr__(self) -> str:
        return str(self)

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        self.data[key[0] * self.column_count + key[1]] = value

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self.data[key[0] * self.column_count + key[1]]

    def __cmp__(self, other: Matrix) -> bool:
        if self.row_count != other.row_count or self.column_count != other.column_count:
            return False

        error = min(self.error(), other.error())

        return all(
            abs(self.data[i] - other.data[i]) < error for i in range(len(self.data))
        )

    def error(self) -> float:
        return (
            max((abs(x) for x in self.data))
            * max(self.row_count, self.column_count)
            * sys.float_info.epsilon
        )

    def to_column_vector(self) -> list[Vector]:
        vectors = [Vector.zero(self.row_count) for _ in range(self.column_count)]
        for i in range(self.row_count):
            for j in range(self.column_count):
                vectors[j][i] = self[i, j]
        return vectors

    def tranpose(self) -> Matrix:
        out_matrix = Matrix.from_matrix(self)
        out_matrix.row_count, out_matrix.column_count = (
            out_matrix.column_count,
            out_matrix.row_count,
        )

        for i in range(out_matrix.row_count):
            for j in range(out_matrix.column_count):
                out_matrix[i, j] = self[j, i]

        return out_matrix

    def __add__(self, other: Matrix) -> Matrix:
        assert self.column_count == other.column_count
        assert self.row_count == other.row_count
        length = len(self.data)

        out_matrix = Matrix.from_matrix(self)
        for i in range(length):
            out_matrix.data[i] += other.data[i]

        return out_matrix

    def __sub__(self, other: Matrix) -> Matrix:
        assert self.column_count == other.column_count
        assert self.row_count == other.row_count
        length = len(self.data)

        out_matrix = Matrix.from_matrix(self)
        for i in range(length):
            out_matrix.data[i] -= other.data[i]

        return out_matrix

    def __mul__(self, other: float | Vector) -> Matrix | Vector:
        if isinstance(other, float):
            out_matrix = Matrix.from_matrix(self)
            for i in range(len(self.data)):
                self.data[i] *= other
            return out_matrix
        elif isinstance(other, Vector):
            return Vector((self @ Matrix.from_column_vector([other])).data)

        return Matrix.from_matrix(self)

    def __matmul__(self, other: Matrix) -> Matrix:
        assert self.column_count == other.row_count

        out_matrix = Matrix.zero(self.row_count, other.column_count)
        for i in range(self.row_count):
            for j in range(other.column_count):
                for k in range(self.column_count):
                    out_matrix[i, j] += self[i, k] * other[k, j]

        return out_matrix

    @staticmethod
    def add_many(matrices: list[Matrix]) -> Matrix:
        return reduce(
            lambda x, y: x + y,
            matrices,
            Matrix.zero(matrices[0].row_count, matrices[0].column_count),
        )

    @staticmethod
    def multiply_many(matrices: list[Matrix]) -> Matrix:
        identity = Matrix.identity(matrices[0].row_count)

        return reduce(lambda x, y: x @ y, matrices, identity)

    def swap_row(self, i: int, j: int) -> None:
        for k in range(self.column_count):
            self[i, k], self[j, k] = self[j, k], self[i, k]

    def multiply_row_by_scalar(self, i: int, scalar: float) -> None:
        for j in range(self.column_count):
            self[i, j] *= scalar

    def add_row_with_scalar(
        self, value_row: int, scalar: float, target_row: int
    ) -> None:
        for i in range(self.column_count):
            self[target_row, i] += scalar * self[value_row, i]

    def first_minor(self, pivot: tuple[int, int]) -> Matrix:
        data: list[float] = []
        for i in range(self.row_count):
            if i == pivot[0]:
                continue
            for j in range(self.column_count):
                if j == pivot[1]:
                    continue
                data.append(self[i, j])

        return Matrix.from_data_with_dimension(
            data, self.row_count - 1, self.column_count - 1
        )

    def adjugate(self) -> Matrix:
        out_matrix = Matrix.zero(self.row_count, self.column_count)

        for i in range(self.row_count):
            for j in range(self.column_count):
                out_matrix[i, j] = sign(j, i) * self.first_minor((j, i)).determinant()

        return out_matrix

    def gauss_jordan_elimination(self) -> tuple[Matrix, Matrix, int, float]:
        size = self.row_count
        error = self.error()
        original_matrix = Matrix.from_matrix(self)
        inverse_matrix = Matrix.identity(size)
        swap_count = 0
        abs_determinant = 1

        for i in range(size):
            max_entry_row = i
            max_entry = abs(original_matrix[i, i])

            # find row with max absolute value in column i
            for j in range(i + 1, original_matrix.row_count):
                value = abs(original_matrix[j, i])
                if value > max_entry:
                    max_entry = value
                    max_entry_row = j

            # skip column if values from [i, i] to [size - 1, i] is zero
            if abs(max_entry) < error:
                continue

            # swap i-th row with row with max value
            if i != max_entry_row:
                swap_count += 1
                original_matrix.swap_row(i, max_entry_row)
                inverse_matrix.swap_row(i, max_entry_row)

            # make [i + 1, i] to [size - 1, i] equal to 0
            for j in range(i + 1, size):
                if abs(original_matrix[j, i]) <= error:
                    continue

                scalar = -original_matrix[j, i] / original_matrix[i, i]
                original_matrix.add_row_with_scalar(i, scalar, j)
                inverse_matrix.add_row_with_scalar(i, scalar, j)

        # convert row echelon form to reduced row echelon form
        for i in range(size - 1, -1, -1):
            if abs(original_matrix[i, i]) < error:
                continue

            # make [i - 1, i] to [0, i] equal to 0
            for j in range(i - 1, -1, -1):
                if abs(original_matrix[j, i]) < error:
                    continue

                scalar = -original_matrix[j, i] / original_matrix[i, i]
                original_matrix.add_row_with_scalar(i, scalar, j)
                inverse_matrix.add_row_with_scalar(i, scalar, j)

        # make the main diagonal equal to 1
        for i in range(size):
            abs_determinant *= original_matrix[i, i]
            if abs(original_matrix[i, i] - 1) < error:
                continue

            if abs(original_matrix[i, i]) < error:
                continue

            inverse_matrix.multiply_row_by_scalar(i, 1 / original_matrix[i, i])
            original_matrix.multiply_row_by_scalar(i, 1 / original_matrix[i, i])

        return original_matrix, inverse_matrix, swap_count, abs_determinant

    def determinant(self) -> float:
        assert self.row_count == self.column_count
        _, _, swap_count, abs_determinant = self.gauss_jordan_elimination()

        return abs_determinant * (1 if swap_count % 2 == 0 else -1)

    def inverse(self) -> Matrix | None:
        assert self.row_count == self.column_count
        size = self.column_count
        if self.rank() != size:
            return None
        return self.gauss_jordan_elimination()[1]

    def rank(self) -> int:
        reduced_row_echelon_matrix = self.gauss_jordan_elimination()[0]
        rank = 0

        for i in range(reduced_row_echelon_matrix.row_count):
            if abs(reduced_row_echelon_matrix[i, i]) > self.error():
                rank += 1

        return rank

    def qr_decomposition(self) -> tuple[Matrix, Matrix]:
        q = Matrix.from_column_vector(
            Vector.gram_schimidt_process(self.to_column_vector())
        )
        r = q.tranpose() @ self
        return q, r

    def schur_form(self) -> Matrix:
        ITERATION_COUNT = 100

        out_matrix = self
        for _ in range(ITERATION_COUNT):
            q, r = out_matrix.qr_decomposition()
            out_matrix = r @ q

        return out_matrix

    def eigenvalues(self) -> list[float]:
        assert self.row_count == self.column_count
        size = self.row_count

        schur_form = self.schur_form()

        eigenvalues = [schur_form[i, i] for i in range(size)]
        eigenvalues.sort(reverse=True)

        return eigenvalues

    def eigenvectors(self) -> list[Vector]:
        ITERATION_COUNT = 100

        assert self.row_count == self.column_count

        eigenvectors_matrix, _ = self.qr_decomposition()

        for i in range(ITERATION_COUNT):
            x = self @ eigenvectors_matrix
            eigenvectors_matrix, _ = x.qr_decomposition()

        return eigenvectors_matrix.to_column_vector()

    def diagonalize(self) -> tuple[Matrix, Matrix, Matrix] | None:
        assert self.row_count == self.column_count
        eigenvalues = self.eigenvalues()
        eigenvectors = self.eigenvectors()

        lambda_matrix = Matrix.from_diagonal(eigenvalues)
        q = Matrix.from_column_vector(eigenvectors)
        q_inverse = q.inverse( )
        if q_inverse is None:
            return None

        return q, lambda_matrix, q_inverse

    def is_symmetric(self) -> bool:
        if self.row_count != self.column_count:
            return False

        size = self.row_count
        for i in range(size):
            for j in range(size):
                if self[i, j] != self[j, i]:
                    return False
        return True

    def is_positive_defined(self) -> bool:
        if not self.is_symmetric():
            return False

        for (
            eigenvalue,
            _,
        ) in self.eigenpairs():
            if eigenvalue <= 0:
                return False

        return True

    def frobenius_norm(self) -> float:
        return sum([x**2 for x in self.data]) ** 0.5
