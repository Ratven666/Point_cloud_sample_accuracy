from sqlalchemy import delete, and_, update

from app.core.classes.MeshLite import MeshLite
from app.core.utils.start_db import engine, Tables


class DeleterBadTriangleInMesh:

    def __init__(self, mesh):
        self._mesh = mesh
        self._deleter = self.__chose_deleter()

    def __chose_deleter(self):
        if isinstance(self._mesh, MeshLite):
            return MeshLiteBadTriangleDeleter

    def __recalculate_mesh_metrics(self):
        svv, sr = 0, 0
        len_ = 0
        for triangle in self._mesh:
            len_ += 1
            try:
                svv += triangle.r * triangle.mse ** 2
                sr += triangle.r
            except TypeError:
                continue
        try:
            mse = (svv / sr) ** 0.5
            r = sr
        except ZeroDivisionError:
            mse = None
            r = 0
        return {"len": len_, "mse": mse, "r": r}

    def delete_triangles_in_mesh(self, bad_triangles):
        self._deleter.deleting_logic(self._mesh, bad_triangles)
        metrics_dict = self.__recalculate_mesh_metrics()
        self._deleter.update_mesh_metrics(self._mesh, metrics_dict)


class MeshDBBadTriangleDeleter:
    @staticmethod
    def deleting_logic(mesh, bad_triangles):
        with engine.connect() as db_connection:
            for triangle_id in bad_triangles.values():
                stmt = delete(Tables.triangles_db_table) \
                    .where(and_(Tables.triangles_db_table.c.id == triangle_id,
                                Tables.triangles_db_table.c.mesh_id == mesh.id))
                db_connection.execute(stmt)
            db_connection.commit()

    @staticmethod
    def update_mesh_metrics(mesh, metrics_dict):
        with engine.connect() as db_connection:
            stmt = update(Tables.meshes_db_table) \
                .where(Tables.meshes_db_table.c.id == mesh.id) \
                .values(len=metrics_dict["len"],
                        r=metrics_dict["r"],
                        mse=metrics_dict["mse"])
            db_connection.execute(stmt)
            db_connection.commit()
        mesh.len = metrics_dict["len"]
        mesh.r = metrics_dict["r"]
        mesh.mse = metrics_dict["mse"]


class MeshLiteBadTriangleDeleter:
    @staticmethod
    def deleting_logic(mesh, bad_triangles):
        good_triangles = []
        for triangle in mesh:
            if triangle in bad_triangles:
                continue
            good_triangles.append(triangle)
        mesh.triangles = good_triangles

    @staticmethod
    def update_mesh_metrics(mesh, metrics_dict):
        mesh.len = metrics_dict["len"]
        mesh.r = metrics_dict["r"]
        mesh.mse = metrics_dict["mse"]
