from app.core.classes.Line import Line
from app.core.classes.MeshCellDB import MeshCellDB
from app.core.classes.Point import Point
from app.core.classes.ScanDB import ScanDB
from app.core.classes.abc_classes.SegmentedModelABC import SegmentedModelABC


class MeshSegmentModelDB(SegmentedModelABC):

    def __init__(self, voxel_model, mesh):
        self.model_type = "MESH"
        self.model_name = f"{self.model_type}_from_{voxel_model.vm_name}"
        self.mse_data = None
        self.cell_type = MeshCellDB
        self.mesh = mesh
        super().__init__(voxel_model, self.cell_type)

    def _calk_segment_model(self):
        """
        Метод определяющий логику создания конкретной модели
        :return: None
        """
        self.logger.info(f"Начат расчет модели {self.model_name}")
        base_scan = ScanDB.get_scan_from_id(self.voxel_model.base_scan_id)
        self._calk_cell_mse(base_scan)

    def _calk_cell_mse(self, base_scan):
        """
        Расчитываает СКП в ячейках сегментированной модели от точек базового скана
        :param base_scan: базовый скан из воксельной модели
        :return: None
        """
        for point in base_scan:
            cell = self.get_model_element_for_point(point)
            if cell is None:
                continue
            cell_z = cell.get_z_from_xy(point.X, point.Y)
            if cell_z is None:
                continue
            try:
                cell.scan_points_in_cell += 1
                cell.vv += (point.Z - cell_z) ** 2
            except AttributeError:
                cell.scan_points_in_cell = 1
                cell.vv = (point.Z - cell_z) ** 2
        for cell in self:
            try:
                cell.r = cell.scan_points_in_cell - cell.count_of_mesh_points
                if cell.r > 0:
                    cell.mse = (cell.vv / cell.r) ** 0.5
            except AttributeError:
                cell.r = 0
                cell.mse = None
        self.logger.info(f"Расчет СКП высот в ячейках модели {self.model_name} завершен")

    def _create_model_structure(self, element_class):
        """
        Создание структуры сегментированной модели
        :param element_class: Класс ячейки конкретной модели
        :return: None
        """
        for voxel in self.voxel_model:
            model_key = f"{voxel.X:.5f}_{voxel.Y:.5f}_{voxel.Z:.5f}"
            self._model_structure[model_key] = element_class(voxel, self)
        self._sort_points_and_triangles_to_cells()

    def _sort_points_and_triangles_to_cells(self):
        for triangle in self.mesh:
            lines = [Line(triangle.point_0, triangle.point_1),
                     Line(triangle.point_1, triangle.point_2),
                     Line(triangle.point_2, triangle.point_0)]
            for line in lines:
                cross_points = line.get_grid_cross_points_list(self.voxel_model.step)
                mid_points = []
                for idx in range(len(cross_points) - 1):
                    mid_x = (cross_points[idx].X + cross_points[idx + 1].X) / 2
                    mid_y = (cross_points[idx].Y + cross_points[idx + 1].Y) / 2
                    mid_points.append(Point(X=mid_x, Y=mid_y, Z=0, R=0, G=0, B=0))
                for point in mid_points:
                    cell = self.get_model_element_for_point(point)
                    if cell is None:
                        continue
                    if triangle not in cell.triangles:
                        cell.triangles.append(triangle)
                        cell.count_of_triangles += 1
            for point in triangle:
                cell = self.get_model_element_for_point(point)
                if cell is not None:
                    if point not in cell.points:
                        cell.points.append(point)
                        cell.count_of_mesh_points += 1
                    if triangle not in cell.triangles:
                        cell.triangles.append(triangle)
                        cell.count_of_triangles += 1
