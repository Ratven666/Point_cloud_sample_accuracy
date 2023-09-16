from abc import ABC


class PointABC(ABC):
    """
    Абстрактный класс точки
    """

    __slots__ = ["id", "X", "Y", "Z", "R", "G", "B"]

    def __init__(self, X, Y, Z, R, G, B, id_=None):
        self.id = id_
        self.X, self.Y, self.Z = X, Y, Z
        self.R, self.G, self.B = R, G, B

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tx: {self.X} y: {self.Y} z: {self.Z},\t" \
               f"RGB: ({self.R},{self.G},{self.B})]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.id}]"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, PointABC):
            raise TypeError("Операнд справа должен иметь тип производный от PointABC")
        if hash(self) == hash(other) or self.id is None or other.id is None:
            return (self.X == other.X) and \
                   (self.Y == other.Y) and \
                   (self.Z == other.Z) and \
                   (self.R == other.R) and \
                   (self.G == other.G) and \
                   (self.B == other.B)
        return False

    def get_dict(self):
        return {"id": self.id,
                "X": self.X, "Y": self.Y, "Z": self.Z,
                "R": self.R, "G": self.G, "B": self.B}
