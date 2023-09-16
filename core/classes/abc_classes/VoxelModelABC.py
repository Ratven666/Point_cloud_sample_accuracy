import logging
from abc import ABC, abstractmethod

from core.CONFIG import LOGGER


class VoxelModelABC(ABC):
    """
    Абстрактный класс воксельной модели
    """
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan, step, dx, dy, dz, is_2d_vxl_mdl=True):
        self.id = None
        self.is_2d_vxl_mdl = is_2d_vxl_mdl
        self.step = float(step)
        self.dx, self.dy, self.dz = self.__dx_dy_formatter(dx, dy, dz)
        self.vm_name: str = self.__name_generator(scan)
        self.len: int = 0
        self.X_count, self.Y_count, self.Z_count = None, None, None
        self.min_X, self.max_X = None, None
        self.min_Y, self.max_Y = None, None
        self.min_Z, self.max_Z = None, None
        self.base_scan_id = None
        self.base_scan = scan

    @staticmethod
    def __dx_dy_formatter(dx, dy, dz):
        return dx % 1, dy % 1, dz % 1

    def __name_generator(self, scan):
        """
        Конструктор имени воксельной модели
        :param scan: базовый скан, по которому создается модель
        :return: None
        """
        vm_type = "2D" if self.is_2d_vxl_mdl else "3D"
        return f"VM_{vm_type}_Sc:{scan.scan_name}_st:{self.step}_dx:{self.dx:.2f}_dy:{self.dz:.2f}_dy:{self.dz:.2f}"

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tName: {self.vm_name}\tLEN: (x:{self.X_count} * y:{self.Y_count} *" \
               f" z:{self.Z_count})={self.len}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len

    @abstractmethod
    def __iter__(self):
        pass
