from abc import ABC, abstractmethod


class TriangulatorABC(ABC):

    def __init__(self, scan):
        self.scan = scan

    @abstractmethod
    def triangulate(self):
        pass
