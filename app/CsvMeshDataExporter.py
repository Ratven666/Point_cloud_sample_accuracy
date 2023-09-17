import csv


class CsvMeshDataExporter:

    def __init__(self, mesh):
        self.mesh = mesh
        self.file_name = f"{self.mesh.mesh_name}.csv"

    def export_mesh_data(self):
        with open(self.file_name, "w", newline="") as csvfile:
            fieldnames = ["point_0", "point_1", "point_2", "area", "r", "rmse"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for triangle in self.mesh:
                data = {"point_0": self.__point_data_exporter(triangle.point_0),
                            "point_1": self.__point_data_exporter(triangle.point_1),
                            "point_2": self.__point_data_exporter(triangle.point_2),
                            "area": triangle.get_area(),
                            "r": triangle.r,
                            "rmse": triangle.mse}
                writer.writerow(data)
        return self.file_name

    @staticmethod
    def __point_data_exporter(point):
        return {"XYZ": [point.X, point.Y, point.Z],
                "RGB": [point.R, point.G, point.B]}
