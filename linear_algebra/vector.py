from __future__ import annotations
from functools import reduce
from typing import override


class Vector:
    data: list[float]

    def __init__(self, data: list[float]) -> None:
        self.data = data

    @staticmethod
    def zero(size: int) -> Vector:
        return Vector([0.0] * size)

    def __setitem__(self, key: int, value: float) -> None:
        self.data[key] = value

    def __getitem__(self, key: int) -> float:
        return self.data[key]

    @override
    def __str__(self) -> str:
        return str(self.data)

    @override
    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: Vector) -> Vector:
        return Vector([self.data[i] + other.data[i] for i in range(len(self.data))])

    def __sub__(self, other: Vector) -> Vector:
        return Vector([self.data[i] - other.data[i] for i in range(len(self.data))])

    def __mul__(self, other: Vector | float) -> Vector | float:
        if isinstance(other, Vector):
            return sum([self.data[i] * other.data[i] for i in range(len(self.data))])
        else:
            return Vector([self.data[i] * other for i in range(len(self.data))])

    def __truediv__(self, other: float) -> Vector:
        return Vector([self.data[i] / other for i in range(len(self.data))])

    def project(self, other: Vector) -> Vector:
        scale = (other * self) / (other * other)
        return other * scale

    def normalize(self) -> Vector:
        return self / self.norm(2)

    def norm(self, p: float) -> float:
        return sum([abs(x) ** p for x in self.data]) ** (1 / p)

    @staticmethod
    def gram_schimidt_process(matrix_rows: list[Vector]) -> list[Vector]:
        othornomalized_columns: list[Vector] = []
        othornomalized_columns.append(matrix_rows[0])

        for i in range(1, len(matrix_rows)):
            othornomalized_columns.append(
                matrix_rows[i]
                - reduce(
                    lambda acc, u: acc + matrix_rows[i].project(u),
                    othornomalized_columns,
                    Vector.zero(len(matrix_rows[0].data)),
                )
            )

        for i in range(len(othornomalized_columns)):
            if othornomalized_columns[i].norm(2) != 0:
                othornomalized_columns[i] = othornomalized_columns[i].normalize()

        return othornomalized_columns
