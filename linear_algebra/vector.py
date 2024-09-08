class Vector:
    data: list[float]

    def __init__(self, data: list[float]) -> None:
        self.data = data

    def norm(self, p: float) -> float:
        return sum([abs(x) ** p for x in self.data]) ** (1 / p)


