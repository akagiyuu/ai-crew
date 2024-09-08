from __future__ import annotations
from typing import override


class Vector:
    data: list[float]

    def __init__(self, data: list[float]) -> None:
        self.data = data

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

    def norm(self, p: float) -> float:
        return sum([abs(x) ** p for x in self.data]) ** (1 / p)
