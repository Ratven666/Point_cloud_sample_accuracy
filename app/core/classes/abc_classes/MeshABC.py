from abc import abstractmethod

from app.core.classes.ScanDB import ScanDB
from app.core.utils.mesh_utils.mesh_triangulators import ScipyTriangulator


class MeshABC:

    def __init__(self, scan, scan_triangulator=ScipyTriangulator):
        self.scan = scan
        self.scan_triangulator = scan_triangulator
        self.mesh_name = self.__name_generator()
        self.len = 0
        self.r = None
        self.mse = None

    @abstractmethod
    def __iter__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[mesh_name: {self.mesh_name},\tlen: {self.len} r: {self.r} mse: {self.mse}]"

    def __name_generator(self):
        return f"MESH_{self.scan.scan_name}"

    def calk_mesh_mse(self, mesh_segment_model, base_scan=None):
        if base_scan is None:
            base_scan = ScanDB.get_scan_from_id(mesh_segment_model.voxel_model.base_scan_id)
        triangles = {}
        for point in base_scan:
            point.id = None
            cell = mesh_segment_model.get_model_element_for_point(point)
            if cell is None or len(cell.triangles) == 0:
                continue
            for triangle in cell.triangles:
                if triangle.is_point_in_triangle(point):
                    if point not in [triangle.point_0, triangle.point_1, triangle.point_2]:
                        try:
                            triangle.vv += (triangle.get_z_from_xy(point.X, point.Y) - point.Z) ** 2
                            triangle.r += 1
                        except (AttributeError, TypeError):
                            triangle.vv = (triangle.get_z_from_xy(point.X, point.Y) - point.Z) ** 2
                            triangle.r = 1
                    triangles[triangle.id] = triangle
                    break
                else:
                    continue
        for triangle in triangles.values():
            if triangle.r is not None:
                try:
                    triangle.mse = (triangle.vv / triangle.r) ** 0.5
                except AttributeError:
                    triangle.mse = None
                    triangle.r = None
        svv, sr = 0, 0
        for triangle in triangles.values():
            try:
                svv += triangle.r * triangle.mse ** 2
                sr += triangle.r
            except TypeError:
                continue
        try:
            self.mse = (svv / sr) ** 0.5
        except ZeroDivisionError:
            self.mse = None
        self.r = sr
        return triangles.values()
