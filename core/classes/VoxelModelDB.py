from sqlalchemy import select, insert, desc

from core.classes.abc_classes.VoxelModelABC import VoxelModelABC
from core.utils.start_db import Tables, engine
from core.utils.voxel_utils.voxel_model_iterators.VMRawIterator import VMRawIterator
from core.utils.voxel_utils.voxel_model_separators.FastVMSeparator import FastVMSeparator


class VoxelModelDB(VoxelModelABC):
    """
    Воксельная модель связанная с базой данных
    """

    def __init__(self, scan, step, dx=0.0, dy=0.0, dz=0.0, is_2d_vxl_mdl=True,
                 voxel_model_separator=FastVMSeparator()):
        super().__init__(scan, step, dx, dy, dz, is_2d_vxl_mdl)
        self.voxel_model_separator = voxel_model_separator
        self.__init_vxl_mdl(scan)
        self.voxel_structure = None

    def __iter__(self):
        return iter(VMRawIterator(self))

    def __init_vxl_mdl(self, scan):
        """
        Инициализирует воксельную модель при запуске
        Если воксельная модеьл с таким именем уже есть в БД - запускает копирование данных из БД в атрибуты модели
        Если такой воксельной модели нет - создает новую запись в БД и запускает процедуру рабиения скана на воксели
        по логике переданного в конструкторе воксельной модели разделителя voxel_model_separator
        :return: None
        """
        select_ = select(Tables.voxel_models_db_table).where(Tables.voxel_models_db_table.c.vm_name == self.vm_name)

        with engine.connect() as db_connection:
            db_vm_data = db_connection.execute(select_).mappings().first()
            if db_vm_data is not None:
                self.__copy_vm_data(db_vm_data)
            else:
                self.__calc_vxl_md_metric(scan)
                self.base_scan_id = scan.id
                stmt = insert(Tables.voxel_models_db_table).values(vm_name=self.vm_name,
                                                                   step=self.step,
                                                                   dx=self.dx,
                                                                   dy=self.dy,
                                                                   dz=self.dz,
                                                                   len=self.len,
                                                                   X_count=self.X_count,
                                                                   Y_count=self.Y_count,
                                                                   Z_count=self.Z_count,
                                                                   min_X=self.min_X,
                                                                   max_X=self.max_X,
                                                                   min_Y=self.min_Y,
                                                                   max_Y=self.max_Y,
                                                                   min_Z=self.min_Z,
                                                                   max_Z=self.max_Z,
                                                                   base_scan_id=self.base_scan_id
                                                                   )
                db_connection.execute(stmt)
                db_connection.commit()
                stmt = (select(Tables.voxel_models_db_table.c.id).order_by(desc("id")))
                self.id = db_connection.execute(stmt).first()[0]
                self.voxel_model_separator.separate_voxel_model(self, scan)

    def __calc_vxl_md_metric(self, scan):
        """
        Рассчитывает границы воксельной модели и максимальное количество вокселей
        исходя из размера вокселя и границ скана
        :param scan: скан на основе которого рассчитываются границы модели
        :return: None
        """
        if len(scan) == 0:
            return None
        self.min_X = (scan.min_X // self.step * self.step) - ((1 - self.dx) % 1 * self.step)
        self.min_Y = (scan.min_Y // self.step * self.step) - ((1 - self.dy) % 1 * self.step)
        self.min_Z = (scan.min_Z // self.step * self.step) - ((1 - self.dy) % 1 * self.step)

        self.max_X = (scan.max_X // self.step + 1) * self.step + ((self.dx % 1) * self.step)
        self.max_Y = (scan.max_Y // self.step + 1) * self.step + ((self.dy % 1) * self.step)
        self.max_Z = (scan.max_Z // self.step + 1) * self.step + ((self.dy % 1) * self.step)

        self.X_count = round((self.max_X - self.min_X) / self.step)
        self.Y_count = round((self.max_Y - self.min_Y) / self.step)
        if self.is_2d_vxl_mdl:
            self.Z_count = 1
        else:
            self.Z_count = round((self.max_Z - self.min_Z) / self.step)
        self.len = self.X_count * self.Y_count * self.Z_count

    def __copy_vm_data(self, db_vm_data: dict):
        """
        Копирует данные записи из БД в атрибуты вокселбной модели
        :param db_vm_data: Результат запроса к БД
        :return: None
        """
        self.id = db_vm_data["id"]
        self.vm_name = db_vm_data["vm_name"]
        self.step = db_vm_data["step"]
        self.dx = db_vm_data["dx"]
        self.dy = db_vm_data["dy"]
        self.dz = db_vm_data["dz"]
        self.len = db_vm_data["len"]
        self.X_count, self.Y_count, self.Z_count = db_vm_data["X_count"], db_vm_data["Y_count"], db_vm_data["Z_count"]
        self.min_X, self.max_X = db_vm_data["min_X"], db_vm_data["max_X"]
        self.min_Y, self.max_Y = db_vm_data["min_Y"], db_vm_data["max_Y"]
        self.min_Z, self.max_Z = db_vm_data["min_Z"], db_vm_data["max_Z"]
        self.base_scan_id = db_vm_data["base_scan_id"]
        if self.Z_count == 1:
            self.is_2d_vxl_mdl = True
        else:
            self.is_2d_vxl_mdl = False
