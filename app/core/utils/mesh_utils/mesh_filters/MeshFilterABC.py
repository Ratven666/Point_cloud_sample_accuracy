from abc import ABC, abstractmethod

from app.core.utils.mesh_utils.mesh_filters.BadTriangleDeleterFromMesh import DeleterBadTriangleInMesh


class MeshFilterABC(ABC):

    def __init__(self, mesh):
        self._mesh = mesh
        self._bad_triangles_deleter = DeleterBadTriangleInMesh(self._mesh)
        self._bad_triangles = {}

    @abstractmethod
    def _filter_logic(self, triangle):
        pass

    def filter_mesh(self):
        for triangle in self._mesh:
            if self._filter_logic(triangle) is False:
                self._bad_triangles[triangle] = triangle.id
        self._bad_triangles_deleter.delete_triangles_in_mesh(self._bad_triangles)
