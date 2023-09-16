from core.classes.Point import Point
from core.classes.Triangle import Triangle
from core.classes.abc_classes.MeshABC import MeshABC
from core.utils.mesh_utils.mesh_triangulators.ScipyTriangulator import ScipyTriangulator


class MeshLite(MeshABC):

    def __init__(self, scan, scan_triangulator=ScipyTriangulator):
        super().__init__(scan, scan_triangulator)
        self.triangles = []
        self.__init_mesh()

    def __iter__(self):
        return iter(self.triangles)

    def calk_mesh_mse(self, mesh_segment_model, base_scan=None, clear_previous_mse=False):
        triangles = super().calk_mesh_mse(mesh_segment_model, base_scan)
        self.triangles = list(triangles)

    def __init_mesh(self):
        triangulation = self.scan_triangulator(self.scan).triangulate()
        self.len = len(triangulation.faces)
        fake_point_id = -1
        fake_triangle_id = -1
        for face in triangulation.faces:
            points = []
            for point_idx in face:
                id_ = triangulation.points_id[point_idx]
                if triangulation.points_id[point_idx] is None:
                    id_ = fake_point_id
                    fake_point_id -= 1
                point = Point(id_=id_,
                              X=triangulation.vertices[point_idx][0],
                              Y=triangulation.vertices[point_idx][1],
                              Z=triangulation.vertices[point_idx][2],
                              R=triangulation.vertices_colors[point_idx][0],
                              G=triangulation.vertices_colors[point_idx][1],
                              B=triangulation.vertices_colors[point_idx][2])
                points.append(point)
            triangle = Triangle(*points)
            triangle.id = fake_triangle_id
            fake_triangle_id -= 1
            self.triangles.append(triangle)
