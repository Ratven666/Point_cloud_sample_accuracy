from os import path

from core.utils.mesh_utils.mesh_exporters.PlyMeshExporter import PlyMeshExporter


class PlyMseMeshExporter(PlyMeshExporter):

    def __init__(self, mesh, min_mse=None, max_mse=None):
        self.min_mse, self.max_mse = self.__get_mse_limits(mesh, min_mse, max_mse)
        super().__init__(mesh)

    @staticmethod
    def __get_mse_limits(mesh, min_mse, max_mse):
        if min_mse is not None and max_mse is not None:
            return min_mse, max_mse
        min_mesh_mse = float("inf")
        max_mesh_mse = 0
        for triangle in mesh:
            mse = triangle.mse
            if mse is None:
                continue
            if mse < min_mesh_mse:
                min_mesh_mse = mse
            if mse > max_mesh_mse:
                max_mesh_mse = mse
        if min_mesh_mse - max_mesh_mse == float("inf"):
            raise ValueError("В поверхности не расчитаны СКП!")
        if min_mse is not None:
            return min_mse, max_mesh_mse
        if max_mse is not None:
            return min_mesh_mse, max_mse
        return min_mesh_mse, max_mesh_mse

    def __get_color_for_mse(self, mse):
        if mse is None or mse == 0:
            return [0, 0, 255]
        if mse > self.max_mse:
            return [255, 0, 0]
        if mse < self.min_mse:
            return [0, 255, 0]
        half_mse_delta = (self.max_mse - self.min_mse) / 2
        mse = mse - half_mse_delta - self.min_mse
        gradient_color = 255 - round((255 * abs(mse)) / half_mse_delta)
        if mse > 0:
            return [255, gradient_color, 0]
        elif mse < 0:
            return [gradient_color, 255, 0]
        else:
            return [255, 255, 0]

    def _init_base_data(self):
        for triangle in self.mesh:
            face_indexes = []
            color_lst = self.__get_color_for_mse(triangle.mse)
            for point in triangle:
                self.vertices.append([point.X, point.Y, point.Z])
                self.vertices_colors.append(color_lst)
                face_indexes.append(len(self.vertices))
            self.faces.append(face_indexes)

    def export(self, file_path="."):
        file_path = path.join(file_path, f"MSE_{self.mesh.mesh_name.replace(':', '=')}"
                                         f"_MseLimits=[{self.min_mse:.3f}-{self.max_mse:.3f}].ply")
        self._save_ply(file_path)
