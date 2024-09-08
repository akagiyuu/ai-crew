from __future__ import annotations
from functools import reduce
from typing import Optional, override
from copy import deepcopy
import operator


class Vector:
    data: list[float]

    def __init__(self, data: list[float]) -> None:
        self.data = data

    def norm(self, p: float) -> float:
        return sum([abs(x) ** p for x in self.data]) ** (1 / p)


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

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        self.data[key[0] * self.column_count + key[1]] = value

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self.data[key[0] * self.column_count + key[1]]

    """
    Clone a matrix
    """
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

    @staticmethod
    def add_many(matrices: list[Matrix]) -> Matrix:
        zero = Matrix.zero(matrices[0].row_count, matrices[0].column_count)

        return reduce(operator.add, matrices, zero)

    def __mul__(self, other: Matrix) -> Matrix:
        assert self.column_count == other.row_count

        out_matrix = Matrix.zero(self.row_count, other.column_count)
        for i in range(self.row_count):
            for j in range(other.column_count):
                for k in range(self.column_count):
                    out_matrix[i, j] += self[i, k] * other[k, j]

        return out_matrix

    @staticmethod
    def multiply_many(matrices: list[Matrix]) -> Matrix:
        identity = Matrix.identity(matrices[0].row_count)

        return reduce(operator.mul, matrices, identity)

    def swap_row(self, i: int, j: int) -> None:
        for k in range(self.column_count):
            self[i, k], self[j, k] = self[j, k], self[i, k]

    def multiply_row_with_scalar(self, i: int, scalar: float) -> None:
        for j in range(self.column_count):
            self[i, j] *= scalar

    def add_row_with_scalar(
        self, value_row: int, scalar: float, target_row: int
    ) -> None:
        for i in range(self.column_count):
            self[target_row, i] += scalar * self[value_row, i]

    def gaussian_elimination(self) -> tuple[Matrix, int]:
        out_matrix = Matrix.from_matrix(self)
        if out_matrix.row_count > out_matrix.column_count:
            out_matrix = out_matrix.transpose()
        swap_count = 0

        for i in range(out_matrix.row_count):
            max_entry_row = i
            max_entry = abs(out_matrix[i, i])

            for j in range(i + 1, out_matrix.row_count):
                value = abs(out_matrix[j, i])
                if value > max_entry:
                    max_entry = value
                    max_entry_row = j
                    
            if max_entry == 0:
                continue

            if i != max_entry_row:
                out_matrix.swap_row(i, max_entry_row)
                swap_count += 1

            for j in range(i + 1, out_matrix.row_count):
                if out_matrix[j, i] == 0:
                    continue

                scalar = -out_matrix[j, i] / out_matrix[i, i]
                out_matrix.add_row_with_scalar(i, scalar, j)

        return out_matrix, swap_count

    def determinant(self) -> float:
        # assert self.row_count == self.column_count
        # size = self.row_count
        #
        # if size == 1:
        #     return abs(self.data[0])
        # if size == 2:
        #     return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        #
        # i = 0
        # return sum(
        #     [
        #         sign(i, j) * self[i, j] * self.minor((i, j)).determinant()
        #         for j in range(size)
        #     ]
        # )
        row_echelon_matrix, swap_count = Matrix.from_matrix(self).gaussian_elimination()

        size = min(row_echelon_matrix.row_count, row_echelon_matrix.column_count)
        return reduce(
            operator.mul,
            [
                row_echelon_matrix[i, i] * (1 if swap_count % 2 == 0 else -1)
                for i in range(size)
            ],
            1,
        )

    def gauss_jordan_elimination(self) -> tuple[Matrix, Matrix]:
        assert self.row_count == self.column_count

        size = self.row_count
        original_matrix = Matrix.from_matrix(self)
        out_matrix = Matrix.identity(size)

        original_matrix = Matrix.from_matrix(self)

        for i in range(size):
            max_entry_row = i
            max_entry = abs(original_matrix[i, i])

            for j in range(i + 1, original_matrix.row_count):
                value = abs(original_matrix[j, i])
                if value > max_entry:
                    max_entry = value
                    max_entry_row = j
                    
            if max_entry == 0:
                continue

            if i != max_entry_row:
                original_matrix.swap_row(i, max_entry_row)
                out_matrix.swap_row(i, max_entry_row)

            for j in range(i + 1, size):
                if original_matrix[j, i] == 0:
                    continue

                scalar = -original_matrix[j, i] / original_matrix[i, i]
                original_matrix.add_row_with_scalar(i, scalar, j)
                out_matrix.add_row_with_scalar(i, scalar, j)

        for i in range(size - 1, -1, -1):
            if original_matrix[i, i] == 0:
                continue
            
            for j in range(i - 1, -1, -1):
                if original_matrix[j, i] == 0:
                    continue

                scalar = -original_matrix[j, i] / original_matrix[i, i]
                original_matrix.add_row_with_scalar(i, scalar, j)
                out_matrix.add_row_with_scalar(i, scalar, j)

            pass

        for i in range(size):
            if original_matrix[i, i] == 0 or original_matrix[i, i] == 1:
                continue
            original_matrix[i, i] = 1
            out_matrix.multiply_row_with_scalar(i, 1 / original_matrix[i, i])

        return original_matrix, out_matrix

    def inverse(self) -> Matrix | None:
        assert self.row_count == self.column_count

        if self.determinant() == 0:
            return None
        return self.gauss_jordan_elimination()[1]

    def rank(self) -> int:
        reduced_row_echelon_matrix = self.gauss_jordan_elimination()[0]
        rank = 0

        for i in range(reduced_row_echelon_matrix.row_count):
            if reduced_row_echelon_matrix[i, i] != 0:
                rank += 1

        return rank

    def eigenpairs(self) -> list[tuple[list[float], float]]:
        pass

    def diagonalize(self) -> tuple[Matrix, Matrix, Matrix]:
        pass

    def is_symmetric(self) -> bool:
        if self.row_count != self.column_count:
            return False
        
        half = self.row_count // 2
        for i in range(half):
            for j in range(half):
                if self[i, j] != self[j, i]:
                    return False
        return True


    def is_positive_defined(self) -> bool:
        if not self.is_symmetric():
            return False
        pass

    def frobenius_norm(self) -> float:
        return sum([x**2 for x in self.data]) ** 0.5


# a = Matrix([[1, 2, 3], [4, 5, 6]])
# b = Matrix([[7, 8], [9, 10], [11, 12]])
# c = Matrix([[13, 14], [15, 16]])
# print(Matrix.multiply_many([a, b, c]))

# a = Vector([1, 2, 3, 4])
# print(Vector.norm(a, 3))

a = Matrix([[1, 2], [2,4]])
b = Matrix([[-3, -1, 2], [3, 1, 4], [0, 3, -1]])
print(b.determinant())
