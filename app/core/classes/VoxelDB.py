from sqlalchemy import select, insert, delete

from app.core.classes.ScanDB import ScanDB
from app.core.classes.abc_classes import VoxelABC
from app.core import Tables, engine


class VoxelDB(VoxelABC):
    """
    Воксель связанный с базой данных
    """

    __slots__ = ["id", "X", "Y", "Z", "step", "vxl_mdl_id", "vxl_name", "scan_id", "len", "R", "G", "B", "scan"]

    def __init__(self, X, Y, Z, step, vxl_mdl_id, db_connection=None):
        super().__init__(X, Y, Z, step, vxl_mdl_id)
        self.scan = None
        self.__init_voxel(db_connection)

    @staticmethod
    def delete_voxel(voxel_id, db_connection=None):
        """
        Удаляет запись вокселя из БД
        :param voxel_id: id вокселя который требуется удалить из БД
        :param db_connection: Открытое соединение с БД
        :return: None
        """
        stmt = delete(Tables.voxels_db_table).where(Tables.voxels_db_table.c.id == voxel_id)
        if db_connection is None:
            with engine.connect() as db_connection:
                db_connection.execute(stmt)
                db_connection.commit()
        else:
            db_connection.execute(stmt)
            db_connection.commit()

    def __init_voxel(self, db_connection=None):
        """
        Инициализирует воксель при запуске
        Если воксель с таким именем уже есть в БД - запускает копирование данных из БД в атрибуты скана
        Если такого вокселя нет - создает новую запись в БД
        :param db_connection: Открытое соединение с БД
        :return: None
        """
        def init_logic(db_conn):
            select_ = select(Tables.voxels_db_table).where(Tables.voxels_db_table.c.vxl_name == self.vxl_name)
            db_voxel_data = db_conn.execute(select_).mappings().first()
            if db_voxel_data is not None:
                self.__copy_voxel_data(db_voxel_data)
                self.scan = ScanDB(f"SC_{self.vxl_name}", db_conn)
            else:
                scan = ScanDB(f"SC_{self.vxl_name}", db_conn)
                stmt = insert(Tables.voxels_db_table).values(vxl_name=self.vxl_name,
                                                             X=self.X,
                                                             Y=self.Y,
                                                             Z=self.Z,
                                                             step=self.step,
                                                             scan_id=scan.id,
                                                             vxl_mdl_id=self.vxl_mdl_id,
                                                             )
                db_conn.execute(stmt)
                db_conn.commit()
                self.__init_voxel(db_conn)

        if db_connection is None:
            with engine.connect() as db_connection:
                init_logic(db_connection)
        else:
            init_logic(db_connection)

    def __copy_voxel_data(self, db_voxel_data: dict):
        """
        Копирует данные записи из БД в атрибуты вокселя
        :param db_voxel_data: Результат запроса к БД
        :return: None
        """
        self.id = db_voxel_data["id"]
        self.X = db_voxel_data["X"]
        self.Y = db_voxel_data["Y"]
        self.Z = db_voxel_data["Z"]
        self.step = db_voxel_data["step"]
        self.vxl_mdl_id = db_voxel_data["vxl_mdl_id"]
        self.vxl_name = db_voxel_data["vxl_name"]
        self.scan_id = db_voxel_data["scan_id"]
        self.len = db_voxel_data["len"]
        self.R = db_voxel_data["R"]
        self.G = db_voxel_data["G"]
        self.B = db_voxel_data["B"]
