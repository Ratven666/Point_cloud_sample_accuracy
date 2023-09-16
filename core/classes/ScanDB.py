from sqlalchemy import select, insert

from core.CONFIG import FILE_NAME, POINTS_CHUNK_COUNT
from core.classes.abc_classes.ScanABC import ScanABC
from core.utils.scan_utils.ScanLoader import ScanLoader
from core.utils.scan_utils.scan_iterators.SqlLiteScanIterator import SqlLiteScanIterator
from core.utils.scan_utils.scan_parsers.ScanTxtParser import ScanTxtParser
from core.utils.start_db import Tables, engine


class ScanDB(ScanABC):
    """
    Скан связанный с базой данных
    Точки при переборе скана берутся напрямую из БД
    """

    def __init__(self, scan_name, db_connection=None):
        super().__init__(scan_name)
        self.__init_scan(db_connection)

    def __iter__(self):
        """
        Иттератор скана берет точки из БД
        """
        return iter(SqlLiteScanIterator(self))

    def load_scan_from_file(self,
                            scan_loader=ScanLoader(scan_parser=ScanTxtParser(chunk_count=POINTS_CHUNK_COUNT)),
                            file_name=FILE_NAME):
        """
        Загружает точки в скан из файла
        Ведется запись в БД
        Обновляются метрики скана в БД
        :param scan_loader: объект определяющий логику работы с БД при загрузке точек (
        принимает в себя парсер определяющий логику работы с конкретным типом файлов)
        :type scan_loader: ScanLoader
        :param file_name: путь до файла из которого будут загружаться данные
        :return: None
        """
        scan_loader.load_data(self, file_name)

    @classmethod
    def get_scan_from_id(cls, scan_id: int):
        """
        Возвращает объект скана по id
        :param scan_id: id скана который требуется загрузить и вернуть из БД
        :return: объект ScanDB с заданным id
        """
        select_ = select(Tables.scans_db_table).where(Tables.scans_db_table.c.id == scan_id)
        with engine.connect() as db_connection:
            db_scan_data = db_connection.execute(select_).mappings().first()
            if db_scan_data is not None:
                return cls(db_scan_data["scan_name"])
            else:
                raise ValueError("Нет скана с таким id!!!")

    def __init_scan(self, db_connection=None):
        """
        Инициализирует скан при запуске
        Если скан с таким именем уже есть в БД - запускает копирование данных из БД в атрибуты скана
        Если такого скана нет - создает новую запись в БД
        :param db_connection: Открытое соединение с БД
        :return: None
        """
        def init_logic(db_conn):
            select_ = select(Tables.scans_db_table).where(Tables.scans_db_table.c.scan_name == self.scan_name)
            db_scan_data = db_conn.execute(select_).mappings().first()
            if db_scan_data is not None:
                self.__copy_scan_data(db_scan_data)
            else:
                stmt = insert(Tables.scans_db_table).values(scan_name=self.scan_name)
                db_conn.execute(stmt)
                db_conn.commit()
                self.__init_scan(db_conn)

        if db_connection is None:
            with engine.connect() as db_connection:
                init_logic(db_connection)
        else:
            init_logic(db_connection)

    def __copy_scan_data(self, db_scan_data: dict):
        """
        Копирует данные записи из БД в атрибуты скана
        :param db_scan_data: Результат запроса к БД
        :return: None
        """
        self.id = db_scan_data["id"]
        self.scan_name = db_scan_data["scan_name"]
        self.len = db_scan_data["len"]
        self.min_X, self.max_X = db_scan_data["min_X"], db_scan_data["max_X"]
        self.min_Y, self.max_Y = db_scan_data["min_Y"], db_scan_data["max_Y"]
        self.min_Z, self.max_Z = db_scan_data["min_Z"], db_scan_data["max_Z"]
