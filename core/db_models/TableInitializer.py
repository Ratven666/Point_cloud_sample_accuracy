from threading import Lock

from core.db_models.dem_models_table import create_dem_models_db_table
from core.db_models.imported_files_table import create_imported_files_table
from core.db_models.mesh_cell_table import create_mesh_cell_db_table
from core.db_models.points_scans_table import create_points_scans_db_table
from core.db_models.points_table import create_points_db_table
from core.db_models.scans_table import create_scans_db_table
from core.db_models.voxel_models_table import create_voxel_models_db_table
from core.db_models.voxels_table import create_voxels_db_table


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class TableInitializer(metaclass=SingletonMeta):
    """
    Объект инициализирующий и создающий таблицы в БД
    """

    def __init__(self, metadata):
        self.__db_metadata = metadata
        self.points_db_table = create_points_db_table(self.__db_metadata)
        self.scans_db_table = create_scans_db_table(self.__db_metadata)
        self.points_scans_db_table = create_points_scans_db_table(self.__db_metadata)
        self.imported_files_db_table = create_imported_files_table(self.__db_metadata)
        self.voxel_models_db_table = create_voxel_models_db_table(self.__db_metadata)
        self.voxels_db_table = create_voxels_db_table(self.__db_metadata)
        self.dem_models_db_table = create_dem_models_db_table(self.__db_metadata)
        self.mesh_cell_db_table = create_mesh_cell_db_table(self.__db_metadata)
