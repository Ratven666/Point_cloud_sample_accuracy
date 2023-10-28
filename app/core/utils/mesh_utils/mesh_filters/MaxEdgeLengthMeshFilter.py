from app.core.classes.Line import Line
from app.core.utils.mesh_utils.mesh_filters.MeshFilterABC import MeshFilterABC


class MaxEdgeLengthMeshFilter(MeshFilterABC):

    def __init__(self, mesh, max_edge_length):
        super().__init__(mesh)
        self.max_edge_length = max_edge_length

    def get_distance(self, line):
        return ((line.point_0.X - line.point_1.X) ** 2 +
                (line.point_0.Y - line.point_1.Y) ** 2 +
                (line.point_0.Z - line.point_1.Z) ** 2) ** 0.5

    def _filter_logic(self, triangle):
        tr_edges = [Line(triangle.point_0, triangle.point_1),
                    Line(triangle.point_1, triangle.point_2),
                    Line(triangle.point_2, triangle.point_0)]
        for edge in tr_edges:
            if self.get_distance(edge) > self.max_edge_length:
                return False
        return True
