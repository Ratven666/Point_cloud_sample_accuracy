import logging
from abc import ABC

from app.core.CONFIG import LOGGER


class VoxelABC(ABC):
    """
    Абстрактный класс вокселя
    """

    logger = logging.getLogger(LOGGER)

    def __init__(self, X, Y, Z, step, vxl_mdl_id):
        self.id = None
        self.X = X
        self.Y = Y
        self.Z = Z
        self.step = step
        self.vxl_mdl_id = vxl_mdl_id
        self.vxl_name = self.__name_generator()
        self.scan_id = None
        self.len = 0
        self.R, self.G, self.B = 0, 0, 0

    def get_dict(self):
        return {"id": self.id,
                "X": self.X, "Y": self.Y, "Z": self.Z,
                "step": self.step,
                "vxl_mdl_id": self.vxl_mdl_id,
                "vxl_name": self.vxl_name,
                "scan_id": self.scan_id,
                "len": self.len,
                "R": self.R, "G": self.G, "B": self.B}

    def __name_generator(self):
        """
        Конструктор имени вокселя
        :return: None
        """
        return (f"VXL_VM:{self.vxl_mdl_id}_s{self.step}_"
                f"X:{round(self.X, 5)}_"
                f"Y:{round(self.Y, 5)}_"
                f"Z:{round(self.Z, 5)}"
                )

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[id: {self.id},\tName: {self.vxl_name}\t\t"
                f"X: {round(self.X, 5)}\tY: {round(self.Y, 5)}\tZ: {round(self.Z, 5)}]"
                )

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len
