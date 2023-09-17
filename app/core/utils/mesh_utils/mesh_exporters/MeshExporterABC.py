from abc import ABC, abstractmethod


class MeshExporterABC(ABC):

    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices = []
        self.vertices_colors = []
        self.faces = []
        self._init_base_data()

    def _init_base_data(self):
        points = {}
        triangles = []
        fake_id = -1
        for triangle in self.mesh:
            face_indexes = []
            triangles.append(triangle)
            for point in triangle:
                if point.id is None:
                    point.id = fake_id
                    fake_id -= 1
                if point in points:
                    face_indexes.append(points[point])
                else:
                    new_idx = len(points)
                    points[point] = new_idx
                    face_indexes.append(new_idx)
                    self.vertices.append([point.X, point.Y, point.Z])
                    self.vertices_colors.append([point.R, point.G, point.B])
            self.faces.append(face_indexes)

    @abstractmethod
    def export(self):
        pass
