from core.classes.abc_classes.PointABC import PointABC
from core.classes.abc_classes.ScanABC import ScanABC
from core.utils.scan_utils.Scan_metrics import update_scan_borders


class ScanLite(ScanABC):
    """
    Скан не связанный с базой данных
    Все данные, включая точки при переборе берутся из оперативной памяти
    """

    def __init__(self, scan_name):
        super().__init__(scan_name)
        self._points = []

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def add_point(self, point):
        """
        Добавляет точку в скан
        :param point: объект класса Point
        :return: None
        """
        if isinstance(point, PointABC):
            self._points.append(point)
            self.len += 1
            update_scan_borders(self, point)
        else:
            raise TypeError(f"Можно добавить только объект точки. "
                            f"Переданно - {type(point)}, {point}")

    @classmethod
    def create_from_another_scan(cls, scan, copy_with_points=True):
        """
        Создает скан типа ScanLite и копирует в него данные из другого скана
        :param scan: копируемый скан
        :param copy_with_points: определяет нужно ли копировать скан вместе с точками
        :type copy_with_points: bool
        :return: объект класса ScanLite
        """
        scan_lite = cls(scan.scan_name)
        scan_lite.id = scan.id
        scan_lite.len = 0
        scan_lite.min_X, scan_lite.min_Y, scan_lite.min_Z = scan.min_X, scan.min_Y, scan.min_Z
        scan_lite.max_X, scan_lite.max_Y, scan_lite.max_Z = scan.max_X, scan.max_Y, scan.max_Z
        if copy_with_points:
            scan_lite._points = [point for point in scan]
            scan_lite.len = len(scan_lite._points)
        return scan_lite

    @classmethod
    def create_from_scan_dict(cls, scan_dict):
        scan_lite = cls(scan_dict["scan_name"])
        scan_lite.id = scan_dict["id"]
        scan_lite.len = scan_dict["len"]
        scan_lite.min_X, scan_lite.min_Y, scan_lite.min_Z = scan_dict["min_X"], scan_dict["min_Y"], scan_dict["min_Z"]
        scan_lite.max_X, scan_lite.max_Y, scan_lite.max_Z = scan_dict["max_X"], scan_dict["max_Y"], scan_dict["max_Z"]
        return scan_lite
