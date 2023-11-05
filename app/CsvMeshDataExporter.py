import csv
from os import path


class CsvMeshDataExporter:

    def __init__(self, mesh):
        self.mesh = mesh
        self.file_name = f"{self.mesh.mesh_name}.csv"

    def export_mesh_data(self, file_path="."):
        file_path = path.join(file_path, f"{self.file_name}")
        with open(file_path, "w", newline="") as csvfile:
            fieldnames = ["point_0", "point_1", "point_2", "area", "r", "rmse"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for triangle in self.mesh:
                r = 0 if triangle.r is None else triangle.r
                data = {"point_0": self.__point_data_exporter(triangle.point_0),
                            "point_1": self.__point_data_exporter(triangle.point_1),
                            "point_2": self.__point_data_exporter(triangle.point_2),
                            "area": triangle.get_area(),
                            "r": r,
                            "rmse": triangle.mse}
                writer.writerow(data)
        return self.file_name

    @staticmethod
    def __point_data_exporter(point):
        return {"XYZ": [point.X, point.Y, point.Z],
                "RGB": [point.R, point.G, point.B]}
