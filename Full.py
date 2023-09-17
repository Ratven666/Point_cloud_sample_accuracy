import logging
import sqlite3
import sys
import warnings
import os
from pathlib import Path
from abc import ABC, abstractmethod
import csv
from threading import Lock

from PyQt6 import QtCore
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QMessageBox, QApplication, QHBoxLayout, QVBoxLayout, \
    QGridLayout, QToolButton, QSpacerItem, QSizePolicy, QFrame, QLabel, QCheckBox, QDoubleSpinBox, QTextEdit, \
    QProgressBar, QTableWidget, QPushButton

import pandas as pd
import seaborn as sns
import numpy as np
from PyQt6.uic.properties import QtWidgets
from scipy.spatial import Delaunay
from sqlalchemy import select, and_, insert, update, create_engine, MetaData, ForeignKey, \
    Column, Integer, Float, Table, String, desc, func

DATABASE_NAME = "TEMP.sqlite"
FILE_NAME = "src/Pit_clean.txt"
POINTS_CHUNK_COUNT = 100_000
LOGGER = "console"
LOGGING_LEVEL = "DEBUG"

logger = logging.getLogger(LOGGER)
path = os.path.join("", DATABASE_NAME)
engine = create_engine(f'sqlite:///{path}')

db_metadata = MetaData()


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


def create_points_db_table(metadata):
    points_db_table = Table("points", metadata,
                            Column("id", Integer, primary_key=True),
                            Column("X", Float, nullable=False),
                            Column("Y", Float, nullable=False),
                            Column("Z", Float, nullable=False),
                            Column("R", Integer, nullable=False),
                            Column("G", Integer, nullable=False),
                            Column("B", Integer, nullable=False),
                            )
    return points_db_table


def create_scans_db_table(metadata):
    scans_db_table = Table("scans", metadata,
                           Column("id", Integer, primary_key=True),
                           Column("scan_name", String, nullable=False, unique=True, index=True),
                           Column("len", Integer, default=0),
                           Column("min_X", Float),
                           Column("max_X", Float),
                           Column("min_Y", Float),
                           Column("max_Y", Float),
                           Column("min_Z", Float),
                           Column("max_Z", Float),
                           )
    return scans_db_table


def create_points_scans_db_table(metadata):
    points_scans_db_table = Table("points_scans", metadata,
                                  Column("point_id", Integer, ForeignKey("points.id"), primary_key=True),
                                  Column("scan_id", Integer, ForeignKey("scans.id"), primary_key=True)
                                  )
    return points_scans_db_table


def create_imported_files_table(metadata):
    imported_files_table = Table("imported_files", metadata,
                                 Column("id", Integer, primary_key=True),
                                 Column("file_name", String, nullable=False),
                                 Column("scan_id", Integer, ForeignKey("scans.id"))
                                 )
    return imported_files_table


def create_voxel_models_db_table(metadata):
    voxel_models_db_table = Table("voxel_models", metadata,
                                  Column("id", Integer, primary_key=True),
                                  Column("vm_name", String, nullable=False, unique=True, index=True),
                                  Column("step", Float, nullable=False),
                                  Column("dx", Float, nullable=False),
                                  Column("dy", Float, nullable=False),
                                  Column("dz", Float, nullable=False),
                                  Column("len", Integer, default=0),
                                  Column("X_count", Integer, default=0),
                                  Column("Y_count", Integer, default=0),
                                  Column("Z_count", Integer, default=0),
                                  Column("min_X", Float),
                                  Column("max_X", Float),
                                  Column("min_Y", Float),
                                  Column("max_Y", Float),
                                  Column("min_Z", Float),
                                  Column("max_Z", Float),
                                  Column("base_scan_id", Integer, ForeignKey("scans.id"))
                                  )
    return voxel_models_db_table


def create_voxels_db_table(metadata):
    voxels_db_table = Table("voxels", metadata,
                            Column("id", Integer, primary_key=True),
                            Column("vxl_name", String, nullable=False, unique=True, index=True),
                            Column("X", Float),
                            Column("Y", Float),
                            Column("Z", Float),
                            Column("step", Float, nullable=False),
                            Column("len", Integer, default=0),
                            Column("R", Integer, default=0),
                            Column("G", Integer, default=0),
                            Column("B", Integer, default=0),
                            Column("scan_id", Integer, ForeignKey("scans.id", ondelete="CASCADE")),
                            Column("vxl_mdl_id", Integer, ForeignKey("voxel_models.id"))
                            )
    return voxels_db_table


def create_dem_models_db_table(metadata):
    dem_models_db_table = Table("dem_models", metadata,
                                Column("id", Integer, primary_key=True),
                                Column("base_voxel_model_id", Integer,
                                       ForeignKey("voxel_models.id")),
                                Column("model_type", String, nullable=False),
                                Column("model_name", String, nullable=False, unique=True),
                                Column("MSE_data", Float, default=None)
                                )
    return dem_models_db_table


def create_mesh_cell_db_table(metadata):
    mesh_cell_db_table = Table("mesh_cells", metadata,
                               Column("voxel_id", Integer,
                                      ForeignKey("voxels.id", ondelete="CASCADE"),
                                      primary_key=True),
                               Column("base_model_id", Integer,
                                      ForeignKey("dem_models.id", ondelete="CASCADE"),
                                      primary_key=True),
                               Column("count_of_mesh_points", Integer),
                               Column("count_of_triangles", Integer),
                               Column("r", Integer),
                               Column("mse", Float, default=None)
                               )
    return mesh_cell_db_table


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


Tables = TableInitializer(db_metadata)


def create_db():
    """
    Создает базу данных при ее отстутсвии
    :return: None
    """
    db_is_created = os.path.exists(path)
    if not db_is_created:
        db_metadata.create_all(engine)
    else:
        logger.info("Такая БД уже есть!")


class PointABC(ABC):
    """
    Абстрактный класс точки
    """

    __slots__ = ["id", "X", "Y", "Z", "R", "G", "B"]

    def __init__(self, X, Y, Z, R, G, B, id_=None):
        self.id = id_
        self.X, self.Y, self.Z = X, Y, Z
        self.R, self.G, self.B = R, G, B

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tx: {self.X} y: {self.Y} z: {self.Z},\t" \
               f"RGB: ({self.R},{self.G},{self.B})]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.id}]"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, PointABC):
            raise TypeError("Операнд справа должен иметь тип производный от PointABC")
        if hash(self) == hash(other) or self.id is None or other.id is None:
            return (self.X == other.X) and \
                   (self.Y == other.Y) and \
                   (self.Z == other.Z) and \
                   (self.R == other.R) and \
                   (self.G == other.G) and \
                   (self.B == other.B)
        return False

    def get_dict(self):
        return {"id": self.id,
                "X": self.X, "Y": self.Y, "Z": self.Z,
                "R": self.R, "G": self.G, "B": self.B}


class Point(PointABC):
    """
    Класс точки
    """

    __slots__ = []

    @classmethod
    def parse_point_from_db_row(cls, row: tuple):
        """
        Метод который создает и возвращает объект Point по данным читаемым из БД
        :param row: кортеж данных читаемых из БД
        :return: объект класса Point
        """
        return cls(id_=row[0], X=row[1], Y=row[2], Z=row[3], R=row[4], G=row[5], B=row[6])


class Line:

    def __init__(self, point_0: Point, point_1: Point, id_=None):
        self.id = id_
        self.point_0 = point_0
        self.point_1 = point_1

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tp0: {self.point_0} p1: {self.point_1}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.id}]"

    def get_distance(self):
        return ((self.point_0.X - self.point_1.X) ** 2 +
                (self.point_0.Y - self.point_1.Y) ** 2 +
                (self.point_0.Z - self.point_1.Z) ** 2) ** 0.5

    def __get_y_by_x(self, x):
        x1, x2 = self.point_0.X, self.point_1.X
        y1, y2 = self.point_0.Y, self.point_1.Y
        y = ((x - x1) * (y2 - y1)) / (x2 - x1) + y1
        return y

    def __get_x_by_y(self, y):
        x1, x2 = self.point_0.X, self.point_1.X
        y1, y2 = self.point_0.Y, self.point_1.Y
        x = ((y - y1) * (x2 - x1)) / (y2 - y1) + x1
        return x

    def get_grid_cross_points_list(self, grid_step):
        points_ = set()
        x1, x2 = self.point_0.X, self.point_1.X
        y1, y2 = self.point_0.Y, self.point_1.Y
        points_.add((x1, y1))
        points_.add((x2, y2))
        x, y = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)
        while True:
            x += grid_step
            grid_x = x // grid_step * grid_step
            if grid_x < x_max:
                grid_y = self.__get_y_by_x(grid_x)
                points_.add((grid_x, grid_y))
            else:
                break
        while True:
            y += grid_step
            grid_y = y // grid_step * grid_step
            if grid_y < y_max:
                grid_x = self.__get_x_by_y(grid_y)
                points_.add((grid_x, grid_y))
            else:
                break
        points_ = sorted(list(points_), key=lambda x_: (x_[0], x_[1]))
        points_ = [Point(X=point[0], Y=point[1], Z=0,
                         R=0, G=0, B=0) for point in points_]
        return points_


class Triangle:
    __slots__ = ["id", "point_0", "point_1", "point_2", "r", "mse", "vv"]

    def __init__(self, point_0: Point, point_1: Point, point_2: Point, r=None, mse=None, id_=None):
        self.id = id_
        self.point_0 = point_0
        self.point_1 = point_1
        self.point_2 = point_2
        self.r = r
        self.mse = mse

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id}\t[[Point_0: [id: {self.point_0.id},\t" \
               f"x: {self.point_0.X} y: {self.point_0.Y} z: {self.point_0.Z}]\t" \
               f"\t\t [Point_1: [id: {self.point_1.id},\t" \
               f"x: {self.point_1.X} y: {self.point_1.Y} z: {self.point_1.Z}]\t" \
               f"\t\t [Point_2: [id: {self.point_2.id},\t" \
               f"x: {self.point_2.X} y: {self.point_2.Y} z: {self.point_2.Z}]\t" \
               f"r: {self.r},\tmse: {self.mse}"

    def __repr__(self):
        return f"{self.__class__.__name__} [id={self.id}, points=[{self.point_0.id}-" \
               f"{self.point_1.id}-{self.point_2.id}]]"

    def __iter__(self):
        point_lst = [self.point_0, self.point_1, self.point_2]
        return iter(point_lst)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            raise TypeError("Операнд справа должен иметь тип Triangle")
        if hash(self) == hash(other) or self.id is None or other.id is None:
            return (self.point_0 == other.point_0) and \
                   (self.point_1 == other.point_1) and \
                   (self.point_2 == other.point_2)
        return False

    def get_z_from_xy(self, x, y):
        """
        Рассчитывает отметку точки (x, y) в плоскости треугольника
        :param x: координата x
        :param y: координата y
        :return: координата z для точки (x, y)
        """
        a = -((self.point_1.Y - self.point_0.Y) * (self.point_2.Z - self.point_0.Z) -
              (self.point_2.Y - self.point_0.Y) * (self.point_1.Z - self.point_0.Z))
        b = ((self.point_1.X - self.point_0.X) * (self.point_2.Z - self.point_0.Z) -
             (self.point_2.X - self.point_0.X) * (self.point_1.Z - self.point_0.Z))
        c = -((self.point_1.X - self.point_0.X) * (self.point_2.Y - self.point_0.Y) -
              (self.point_2.X - self.point_0.X) * (self.point_1.Y - self.point_0.Y))
        d = -(self.point_0.X * a + self.point_0.Y * b + self.point_0.Z * c)
        z = (a * x + b * y + d) / -c
        return z

    def get_area(self):
        a = ((self.point_1.X - self.point_0.X) ** 2 + (self.point_1.Y - self.point_0.Y) ** 2) ** 0.5
        b = ((self.point_2.X - self.point_1.X) ** 2 + (self.point_2.Y - self.point_1.Y) ** 2) ** 0.5
        c = ((self.point_0.X - self.point_2.X) ** 2 + (self.point_0.Y - self.point_2.Y) ** 2) ** 0.5
        p = (a + b + c) / 2
        s = (p * (p - a) * (p - b) * (p - c)) ** 0.5
        return s

    def is_point_in_triangle(self, point: Point):
        s_abc = self.get_area()
        if s_abc == 0:
            return False
        s_ab_p = Triangle(self.point_0, self.point_1, point).get_area()
        s_bc_p = Triangle(self.point_1, self.point_2, point).get_area()
        s_ca_p = Triangle(self.point_2, self.point_0, point).get_area()
        delta_s = abs(s_abc - (s_ab_p + s_bc_p + s_ca_p))
        if delta_s < 1e-6:
            return True
        return False

    @classmethod
    def parse_triangle_from_db_row(cls, row: tuple):
        """
        Метод который создает и возвращает объект Triangle по данным читаемым из БД
        :param row: кортеж данных читаемых из БД
        :return: объект класса Triangle
        """
        id_ = row[0]
        r = row[1]
        mse = row[2]
        point_0 = Point.parse_point_from_db_row(row[3:10])
        point_1 = Point.parse_point_from_db_row(row[10:17])
        point_2 = Point.parse_point_from_db_row(row[17:])
        return cls(id_=id_, r=r, mse=mse, point_0=point_0, point_1=point_1, point_2=point_2)

    def get_dict(self):
        return {"id": self.id,
                "r": self.r,
                "mse": self.mse,
                "point_0": self.point_0.get_dict(),
                "point_1": self.point_1.get_dict(),
                "point_2": self.point_2.get_dict()}


class ScanABC(ABC):
    """
    Абстрактный класс скана
    """
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan_name):
        self.id = None
        self.scan_name: str = scan_name
        self.len: int = 0
        self.min_X, self.max_X = None, None
        self.min_Y, self.max_Y = None, None
        self.min_Z, self.max_Z = None, None

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tName: {self.scan_name}\tLEN: {self.len}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len

    @abstractmethod
    def __iter__(self):
        pass


def update_scan_borders(scan, point):
    """
    Проверяет положение в точки в существующих границах скана
    и меняет их при выходе точки за их пределы
    :param scan: скан
    :param point: точка
    :return: None
    """
    if scan.min_X is None:
        scan.min_X, scan.max_X = point.X, point.X
        scan.min_Y, scan.max_Y = point.Y, point.Y
        scan.min_Z, scan.max_Z = point.Z, point.Z
    if point.X < scan.min_X:
        scan.min_X = point.X
    if point.X > scan.max_X:
        scan.max_X = point.X
    if point.Y < scan.min_Y:
        scan.min_Y = point.Y
    if point.Y > scan.max_Y:
        scan.max_Y = point.Y
    if point.Z < scan.min_Z:
        scan.min_Z = point.Z
    if point.Z > scan.max_Z:
        scan.max_Z = point.Z


class ScanLite(ScanABC):
    """
    Скан не связанный с базой данных
    Все данные, включая точки при переборе берутся из оперативной памяти
    """

    def __init__(self, scan_name):
        super().__init__(scan_name)
        self._points = []

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def add_point(self, point):
        """
        Добавляет точку в скан
        :param point: объект класса Point
        :return: None
        """
        if isinstance(point, PointABC):
            self._points.append(point)
            self.len += 1
            update_scan_borders(self, point)
        else:
            raise TypeError(f"Можно добавить только объект точки. "
                            f"Переданно - {type(point)}, {point}")

    @classmethod
    def create_from_another_scan(cls, scan, copy_with_points=True):
        """
        Создает скан типа ScanLite и копирует в него данные из другого скана
        :param scan: копируемый скан
        :param copy_with_points: определяет нужно ли копировать скан вместе с точками
        :type copy_with_points: bool
        :return: объект класса ScanLite
        """
        scan_lite = cls(scan.scan_name)
        scan_lite.id = scan.id
        scan_lite.len = 0
        scan_lite.min_X, scan_lite.min_Y, scan_lite.min_Z = scan.min_X, scan.min_Y, scan.min_Z
        scan_lite.max_X, scan_lite.max_Y, scan_lite.max_Z = scan.max_X, scan.max_Y, scan.max_Z
        if copy_with_points:
            scan_lite._points = [point for point in scan]
            scan_lite.len = len(scan_lite._points)
        return scan_lite

    @classmethod
    def create_from_scan_dict(cls, scan_dict):
        scan_lite = cls(scan_dict["scan_name"])
        scan_lite.id = scan_dict["id"]
        scan_lite.len = scan_dict["len"]
        scan_lite.min_X, scan_lite.min_Y, scan_lite.min_Z = scan_dict["min_X"], scan_dict["min_Y"], scan_dict["min_Z"]
        scan_lite.max_X, scan_lite.max_Y, scan_lite.max_Z = scan_dict["max_X"], scan_dict["max_Y"], scan_dict["max_Z"]
        return scan_lite


class ScanParserABC(ABC):
    """
    Абстрактный класс парсера данных для скана
    """
    logger = logging.getLogger(LOGGER)

    def __str__(self):
        return f"Парсер типа: {self.__class__.__name__}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _check_file_extension(file_name, __supported_file_extensions__):
        """
        Проверяет соответствует ли расширение файла допустимому для парсера
        :param file_name: имя и путь до файла, который будет загружаться
        :param __supported_file_extensions__: список допустимых расширений для выбранного парсера
        :return: None
        """
        file_extension = f".{file_name.split('.')[-1]}"
        if file_extension not in __supported_file_extensions__:
            raise TypeError(f"Неправильный для парсера тип файла. "
                            f"Ожидаются файлы типа: {__supported_file_extensions__}")

    @staticmethod
    def _get_last_point_id():
        """
        Возвращает последний id для точки в таблице БД points
        :return: последний id для точки в таблице БД points
        """
        with engine.connect() as db_connection:
            stmt = (select(Tables.points_db_table.c.id).order_by(desc("id")))
            last_point_id = db_connection.execute(stmt).first()
            if last_point_id:
                return last_point_id[0]
            else:
                return 0

    @abstractmethod
    def parse(self, file_name: str):
        """
        Запускает процедуру парсинга
        :param file_name: имя и путь до файла, который будет загружаться
        :return:
        """
        pass


class ScanTxtParser(ScanParserABC):
    """
    Парсер точек из текстового txt формата
    Формат данных:
        4.2517 -14.2273 33.4113 208 195 182 -0.023815 -0.216309 0.976035
          X        Y       Z     R   G   B      nX nY nZ (не обязательны и пока игнорируются)
    """
    __supported_file_extension__ = [".txt"]

    def __init__(self, chunk_count=POINTS_CHUNK_COUNT):
        self.__chunk_count = chunk_count
        self.__last_point_id = None

    def parse(self, file_name=FILE_NAME):
        """
        Запускает процедуру парсинга файла и возвращает списки словарей с данными для загрузки в БД
        размером не превышающим POINTS_CHUNK_COUNT
        При запуске выполняется процедурка проверки расширения файла
        :param file_name: путь до файла из которго будут загружаться данные
        :return: список точек готовый к загрузке в БД
        """
        self._check_file_extension(file_name, self.__supported_file_extension__)
        self.__last_point_id = self._get_last_point_id()

        with open(file_name, "rt", encoding="utf-8") as file:
            points = []
            for line in file:
                line = line.strip().split()
                self.__last_point_id += 1
                try:
                    point = {"id": self.__last_point_id,
                             "X": line[0], "Y": line[1], "Z": line[2],
                             "R": line[3], "G": line[4], "B": line[5]
                             }
                except IndexError:
                    self.logger.critical(f"Структура \"{file_name}\" некорректна - \"{line}\"")
                    return
                points.append(point)
                if len(points) == self.__chunk_count:
                    yield points
                    points = []
            yield points


def update_scan_in_db_from_scan(updated_scan, db_connection=None):
    """
    Обновляет значения метрик скана в БД
    :param updated_scan: Объект скана для которого обновляются метрики
    :param db_connection: Открытое соединение с БД
    :return: None
    """
    stmt = update(Tables.scans_db_table) \
        .where(Tables.scans_db_table.c.id == updated_scan.id) \
        .values(scan_name=updated_scan.scan_name,
                len=updated_scan.len,
                min_X=updated_scan.min_X,
                max_X=updated_scan.max_X,
                min_Y=updated_scan.min_Y,
                max_Y=updated_scan.max_Y,
                min_Z=updated_scan.min_Z,
                max_Z=updated_scan.max_Z)
    if db_connection is None:
        with engine.connect() as db_connection:
            db_connection.execute(stmt)
            db_connection.commit()
    else:
        db_connection.execute(stmt)
        db_connection.commit()


def calk_scan_metrics(scan_id):
    """
    Рассчитывает метрики скана средствами SQL
    :param scan_id: id скана для которого будет выполняться расчет метрик
    :return: словарь с метриками скана
    """
    with engine.connect() as db_connection:
        stmt = select(func.count(Tables.points_db_table.c.id).label("len"),
                      func.min(Tables.points_db_table.c.X).label("min_X"),
                      func.max(Tables.points_db_table.c.X).label("max_X"),
                      func.min(Tables.points_db_table.c.Y).label("min_Y"),
                      func.max(Tables.points_db_table.c.Y).label("max_Y"),
                      func.min(Tables.points_db_table.c.Z).label("min_Z"),
                      func.max(Tables.points_db_table.c.Z).label("max_Z")).where(and_(
            Tables.points_scans_db_table.c.point_id == Tables.points_db_table.c.id,
            Tables.points_scans_db_table.c.scan_id == Tables.scans_db_table.c.id,
            Tables.scans_db_table.c.id == scan_id
        ))
        scan_metrics = dict(db_connection.execute(stmt).mappings().first())
        scan_metrics["id"] = scan_id
        return scan_metrics


def update_scan_metrics(scan):
    """
    Рассчитывает значения метрик скана по точкам загруженным в БД
    средствами SQL и обновляет их в самом скане
    :param scan: скан для которого рассчитываются и в котором обновляются метрики
    :return: скан с обновленными  метриками
    """
    scan_metrics = calk_scan_metrics(scan_id=scan.id)
    scan.len = scan_metrics["len"]
    scan.min_X, scan.max_X = scan_metrics["min_X"], scan_metrics["max_X"]
    scan.min_Y, scan.max_Y = scan_metrics["min_Y"], scan_metrics["max_Y"]
    scan.min_Z, scan.max_Z = scan_metrics["min_Z"], scan_metrics["max_Z"]
    return scan


class ImportedFileDB:
    """
    Класс определяющий логику контроля повторной загрузки файла с данными
    """

    def __init__(self, file_name):
        self.__file_name = file_name
        self.__hash = None

    def is_file_already_imported_into_scan(self, scan):
        """
        Проверяет был ли этот файл уже загружен в скан

        :param scan: скан в который загружаются данные из файла
        :type scan: ScanDB
        :return: True / False
        """
        select_ = select(Tables.imported_files_db_table).where(
            and_(Tables.imported_files_db_table.c.file_name == self.__file_name,
                 Tables.imported_files_db_table.c.scan_id == scan.id))
        with engine.connect() as db_connection:
            imp_file = db_connection.execute(select_).first()
        if imp_file is None:
            return False
        return True

    def insert_in_db(self, scan):
        """
        Добавляет в таблицу БД imported_files данные о файле и скане в который он был загружен
        :param scan: скан в который загружаются данные из файла
        :type scan: ScanDB
        :return: None
        """
        with engine.connect() as db_connection:
            stmt = insert(Tables.imported_files_db_table).values(file_name=self.__file_name,
                                                                 scan_id=scan.id)
            db_connection.execute(stmt)
            db_connection.commit()


class ScanLoader:
    """
    Класс, определяющий логику загрузки точек в БД
    """
    __logger = logging.getLogger(LOGGER)

    def __init__(self, scan_parser=ScanTxtParser()):
        self.__scan_parser = scan_parser

    def load_data(self, scan, file_name: str):
        """
        Загрузка данных из файла в базу данных

        :param scan: скан в который загружаются данные из файла
        :type scan: ScanDB
        :param file_name: путь до файла с данными
        :type file_name: str
        :return: None

        При выполнении проверяется был ли ранее произведен импорт в этот скан из этого файла.
        Если файл ранее не импортировался - происходит загрузка.
        Полсле загрузки данных рассчитываются новые метрики скана, которые обновляют его свойства в БД
        Файл с данными записывается в таблицу imported_files
        """
        imp_file = ImportedFileDB(file_name)

        if imp_file.is_file_already_imported_into_scan(scan):
            self.__logger.info(f"Файл \"{file_name}\" уже загружен в скан \"{scan.scan_name}\"")
            return

        with engine.connect() as db_connection:
            for points in self.__scan_parser.parse(file_name):
                points_scans = self.__get_points_scans_list(scan, points)
                self.__insert_to_db(points, points_scans, db_connection)
                self.__logger.info(f"Пакет точек загружен в БД")
            db_connection.commit()
        scan = update_scan_metrics(scan)
        update_scan_in_db_from_scan(scan)
        imp_file.insert_in_db(scan)
        self.__logger.info(f"Точки из файла \"{file_name}\" успешно"
                           f" загружены в скан \"{scan.scan_name}\"")

    @staticmethod
    def __get_points_scans_list(scan, points):
        """
        Собирает список словарей для пакетной загрузки в таблицу points_scans_db_table

        :param scan: скан в который загружаются данные из файла
        :type scan: ScanDB
        :param points: список точек полученный из парсера
        :type points: list
        :return: список словарей для пакетной загрузки в таблицу points_scans_db_table
        """
        points_scans = []
        for point in points:
            points_scans.append({"point_id": point["id"], "scan_id": scan.id})
        return points_scans

    @staticmethod
    def __insert_to_db(points, points_scans, db_engine_connection):
        """
        Загружает данные о точках и их связях со сканами в БД
        :param points: список словарей для пакетной загрузки в таблицу points_db_table
        :param points_scans: список словарей для пакетной загрузки в таблицу points_scans_db_table
        :param db_engine_connection: открытое соединение с БД
        :return: None
        """
        db_engine_connection.execute(Tables.points_db_table.insert(), points)
        db_engine_connection.execute(Tables.points_scans_db_table.insert(), points_scans)

    @property
    def scan_parser(self):
        return self.__scan_parser

    @scan_parser.setter
    def scan_parser(self, parser: ScanParserABC):
        if isinstance(parser, ScanParserABC):
            self.__scan_parser = parser
        else:
            raise TypeError(f"Нужно передать объект парсера! "
                            f"Переданно - {type(parser)}, {parser}")


class SqlLiteScanIterator:
    """
    Иттератор скана из БД SQLite
    Реализован через стандартную библиотеку sqlite3
    """

    def __init__(self, scan):
        self.__path = os.path.join("", DATABASE_NAME)
        self.scan_id = scan.id
        self.cursor = None
        self.generator = None

    def __iter__(self):
        connection = sqlite3.connect(self.__path)
        self.cursor = connection.cursor()
        stmt = """SELECT p.id, p.X, p.Y, p.Z,
                         p.R, p.G, p.B
                  FROM points p
                  JOIN (SELECT ps.point_id
                        FROM points_scans ps
                        WHERE ps.scan_id = (?)) ps
                  ON ps.point_id = p.id
                          """
        self.generator = (Point.parse_point_from_db_row(data) for data in
                          self.cursor.execute(stmt, (self.scan_id,)))
        return self.generator

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration:
            self.cursor.close()
            raise StopIteration
        finally:
            self.cursor.close()


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


class VoxelModelABC(ABC):
    """
    Абстрактный класс воксельной модели
    """
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan, step, dx, dy, dz, is_2d_vxl_mdl=True):
        self.id = None
        self.is_2d_vxl_mdl = is_2d_vxl_mdl
        self.step = float(step)
        self.dx, self.dy, self.dz = self.__dx_dy_formatter(dx, dy, dz)
        self.vm_name: str = self.__name_generator(scan)
        self.len: int = 0
        self.X_count, self.Y_count, self.Z_count = None, None, None
        self.min_X, self.max_X = None, None
        self.min_Y, self.max_Y = None, None
        self.min_Z, self.max_Z = None, None
        self.base_scan_id = None
        self.base_scan = scan

    @staticmethod
    def __dx_dy_formatter(dx, dy, dz):
        return dx % 1, dy % 1, dz % 1

    def __name_generator(self, scan):
        """
        Конструктор имени воксельной модели
        :param scan: базовый скан, по которому создается модель
        :return: None
        """
        vm_type = "2D" if self.is_2d_vxl_mdl else "3D"
        return f"VM_{vm_type}_Sc:{scan.scan_name}_st:{self.step}_dx:{self.dx:.2f}_dy:{self.dz:.2f}_dy:{self.dz:.2f}"

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tName: {self.vm_name}\tLEN: (x:{self.X_count} * y:{self.Y_count} *" \
               f" z:{self.Z_count})={self.len}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len

    @abstractmethod
    def __iter__(self):
        pass


class VoxelABC(ABC):
    """
    Абстрактный класс вокселя
    """

    logger = logging.getLogger(LOGGER)

    def __init__(self, X, Y, Z, step, vxl_mdl_id):
        self.id = None
        self.X = X
        self.Y = Y
        self.Z = Z
        self.step = step
        self.vxl_mdl_id = vxl_mdl_id
        self.vxl_name = self.__name_generator()
        self.scan_id = None
        self.len = 0
        self.R, self.G, self.B = 0, 0, 0

    def get_dict(self):
        return {"id": self.id,
                "X": self.X, "Y": self.Y, "Z": self.Z,
                "step": self.step,
                "vxl_mdl_id": self.vxl_mdl_id,
                "vxl_name": self.vxl_name,
                "scan_id": self.scan_id,
                "len": self.len,
                "R": self.R, "G": self.G, "B": self.B}

    def __name_generator(self):
        """
        Конструктор имени вокселя
        :return: None
        """
        return (f"VXL_VM:{self.vxl_mdl_id}_s{self.step}_"
                f"X:{round(self.X, 5)}_"
                f"Y:{round(self.Y, 5)}_"
                f"Z:{round(self.Z, 5)}"
                )

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[id: {self.id},\tName: {self.vxl_name}\t\t"
                f"X: {round(self.X, 5)}\tY: {round(self.Y, 5)}\tZ: {round(self.Z, 5)}]"
                )

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len


class VoxelLite(VoxelABC):
    """
    Воксель не связанный с базой данных
    """
    __slots__ = ["id", "X", "Y", "Z", "step", "vxl_mdl_id", "vxl_name", "scan_id", "len", "R", "G", "B"]

    def __init__(self, X, Y, Z, step, vxl_mdl_id):
        super().__init__(X, Y, Z, step, vxl_mdl_id)
        self.scan = ScanLite(f"SC_{self.vxl_name}")

    @classmethod
    def parse_voxel_from_data_row(cls, row):
        voxel = cls(X=row[1], Y=row[2], Z=row[3], step=row[4], vxl_mdl_id=row[5])
        voxel.id = row[0]
        voxel.vxl_name = row[6]
        voxel.scan_id = row[7]
        voxel.len = row[8]
        voxel.R = row[9]
        voxel.G = row[10]
        voxel.B = row[11]
        return voxel


class VMRawIterator:
    """
    Универсальный иттератор вокселльной модели из БД
    Реализован средствами sqlalchemy
    """

    def __init__(self, vxl_model):
        self.__vxl_model = vxl_model
        self.__engine = engine.connect()
        self.__select = select(Tables.voxels_db_table).where(self.__vxl_model.id == Tables.voxels_db_table.c.vxl_mdl_id)
        self.__query = self.__engine.execute(self.__select).mappings()
        self.__iterator = None

    def __iter__(self):
        self.__iterator = iter(self.__query)
        return self

    def __next__(self):
        try:
            row = next(self.__iterator)
            voxel = VoxelLite(X=row["X"], Y=row["Y"], Z=row["Z"],
                              step=row["step"],
                              vxl_mdl_id=row["vxl_mdl_id"])
            voxel.id = row["id"]
            voxel.R, voxel.G, voxel.B = row["R"], row["G"], row["B"]
            voxel.len = row["len"]
            voxel.scan_id = row["scan_id"]
            voxel.vxl_name = row["vxl_name"]
            return voxel
        except StopIteration:
            self.__engine.close()
            raise StopIteration
        finally:
            self.__engine.close()


class VMFullBaseIterator:
    """
    Иттератор полной воксельной модели
    """

    def __init__(self, vxl_mdl):
        self.vxl_mdl = vxl_mdl
        self.x = 0
        self.y = 0
        self.z = 0
        self.X_count, self.Y_count, self.Z_count = vxl_mdl.X_count, vxl_mdl.Y_count, vxl_mdl.Z_count

    def __iter__(self):
        return self

    def __next__(self):
        for vxl_z in range(self.z, self.Z_count):
            for vxl_y in range(self.y, self.Y_count):
                for vxl_x in range(self.x, self.X_count):
                    self.x += 1
                    return self.vxl_mdl.voxel_structure[vxl_z][vxl_y][vxl_x]
                self.y += 1
                self.x = 0
            self.z += 1
            self.y = 0
        raise StopIteration


class VMSeparatorABC(ABC):
    """
    Абстрактный сепоратор вокселььной модели
    """

    @abstractmethod
    def separate_voxel_model(self, voxel_model, scan):
        pass


def update_voxel_model_in_db_from_voxel_model(updated_voxel_model, db_connection=None):
    """
    Обновляет значения метрик воксельной модели в БД
    :param updated_voxel_model: Объект воксельной модели для которой обновляются метрики
    :param db_connection: Открытое соединение с БД
    :return: None
    """
    stmt = update(Tables.voxel_models_db_table) \
        .where(Tables.voxel_models_db_table.c.id == updated_voxel_model.id) \
        .values(id=updated_voxel_model.id,
                vm_name=updated_voxel_model.vm_name,
                step=updated_voxel_model.step,
                len=updated_voxel_model.len,
                X_count=updated_voxel_model.X_count,
                Y_count=updated_voxel_model.Y_count,
                Z_count=updated_voxel_model.Z_count,
                min_X=updated_voxel_model.min_X,
                max_X=updated_voxel_model.max_X,
                min_Y=updated_voxel_model.min_Y,
                max_Y=updated_voxel_model.max_Y,
                min_Z=updated_voxel_model.min_Z,
                max_Z=updated_voxel_model.max_Z,
                base_scan_id=updated_voxel_model.base_scan_id)
    if db_connection is None:
        with engine.connect() as db_connection:
            db_connection.execute(stmt)
            db_connection.commit()
    else:
        db_connection.execute(stmt)
        db_connection.commit()


class FastVMSeparator(VMSeparatorABC):
    """
    Быстрый сепоратор воксельной модели через создание
    полной воксельной структуры в оперативной памяти
    """

    def __init__(self):
        self.voxel_model = None
        self.voxel_structure = None

    def separate_voxel_model(self, voxel_model, scan):
        """
        Общая логика разбиения воксельной модели
        :param voxel_model: воксельная модель
        :param scan: скан
        :return: None
        1. Создается полная воксельная структура
        2. Скан разбивается на отдельные воксели
        3. Загружает метрики сканов и вокселей игнорируя пустые
        """
        voxel_model.logger.info(f"Начато создание структуры {voxel_model.vm_name}")
        self.__create_full_vxl_struct(voxel_model)
        voxel_model.logger.info(f"Структура {voxel_model.vm_name} создана")
        voxel_model.logger.info(f"Начат расчет метрик сканов и вокселей")
        self.__update_scan_and_voxel_data(scan)
        voxel_model.logger.info(f"Расчет метрик сканов и вокселей завершен")
        voxel_model.logger.info(f"Начата загрузка метрик сканов и вокселей в БД")
        self.__load_scan_and_voxel_data_in_db()
        voxel_model.logger.info(f"Загрузка метрик сканов и вокселей в БД завершена")

    def __create_full_vxl_struct(self, voxel_model):
        """
        Создается полная воксельная структура
        :param voxel_model: воксельная модель
        :return: None
        """
        self.voxel_model = voxel_model
        self.voxel_structure = [[[VoxelLite(voxel_model.min_X + x * voxel_model.step,
                                            voxel_model.min_Y + y * voxel_model.step,
                                            voxel_model.min_Z + z * voxel_model.step,
                                            voxel_model.step, voxel_model.id)
                                  for x in range(voxel_model.X_count)]
                                 for y in range(voxel_model.Y_count)]
                                for z in range(voxel_model.Z_count)]
        self.voxel_model.voxel_structure = self.voxel_structure

    def __update_scan_and_voxel_data(self, scan):
        """
        Пересчитывает метрики сканов и вокселей по базовому скану scan
        :param scan: скан по которому разбивается воксельная модель
        :return: None
        """
        for point in scan:
            vxl_md_X = int((point.X - self.voxel_model.min_X) // self.voxel_model.step)
            vxl_md_Y = int((point.Y - self.voxel_model.min_Y) // self.voxel_model.step)
            if self.voxel_model.is_2d_vxl_mdl:
                vxl_md_Z = 0
            else:
                vxl_md_Z = int((point.Z - self.voxel_model.min_Z) // self.voxel_model.step)
            self.__update_scan_data(self.voxel_structure[vxl_md_Z][vxl_md_Y][vxl_md_X].scan,
                                    point)
            self.__update_voxel_data(self.voxel_structure[vxl_md_Z][vxl_md_Y][vxl_md_X], point)
        self.__init_scans_and_voxels_id()

    @staticmethod
    def __update_scan_data(scan, point):
        """
        Обновляет значения метрик скана (количество точек и границы)
        :param scan: обновляемый скан
        :param point: добавляемая в скан точка
        :return: None
        """
        scan.len += 1
        update_scan_borders(scan, point)

    @staticmethod
    def __update_voxel_data(voxel, point):
        """
        Обновляет значения метрик вокселя (цвет и количество точек)
        :param voxel: обновляемый воксель
        :param point: точка, попавшая в воксель
        :return: None
        """
        voxel.R = (voxel.R * voxel.len + point.R) / (voxel.len + 1)
        voxel.G = (voxel.G * voxel.len + point.G) / (voxel.len + 1)
        voxel.B = (voxel.B * voxel.len + point.B) / (voxel.len + 1)
        voxel.len += 1

    def __init_scans_and_voxels_id(self):
        """
        Иничиирует в сканы и воксели модели id
        :return: None
        """
        last_scan_id_stmt = (select(Tables.scans_db_table.c.id).order_by(desc("id")))
        last_voxels_id_stmt = (select(Tables.voxels_db_table.c.id).order_by(desc("id")))
        with engine.connect() as db_connection:
            last_scan_id = db_connection.execute(last_scan_id_stmt).first()
            last_voxel_id = db_connection.execute(last_voxels_id_stmt).first()
        last_scan_id = last_scan_id[0] if last_scan_id else 0
        last_voxel_id = last_voxel_id[0] if last_voxel_id else 0
        for voxel in iter(VMFullBaseIterator(self.voxel_model)):
            last_scan_id += 1
            last_voxel_id += 1
            voxel.id = last_voxel_id
            voxel.scan.id = last_scan_id

    def __load_scan_and_voxel_data_in_db(self):
        """
        Загружает значения метрик сканов и вокселей в БД
        игнорируя пустые воксели
        :return: None
        """
        voxels = []
        scans = []
        voxel_counter = 0
        for voxel in iter(VMFullBaseIterator(self.voxel_model)):
            if len(voxel) == 0:
                continue
            scan = voxel.scan
            scans.append({"id": scan.id,
                          "scan_name": scan.scan_name,
                          "len": scan.len,
                          "min_X": scan.min_X,
                          "max_X": scan.max_X,
                          "min_Y": scan.min_Y,
                          "max_Y": scan.max_Y,
                          "min_Z": scan.min_Z,
                          "max_Z": scan.max_Z
                          })
            voxels.append({"id": voxel.id,
                           "vxl_name": voxel.vxl_name,
                           "X": voxel.X,
                           "Y": voxel.Y,
                           "Z": voxel.Z,
                           "step": voxel.step,
                           "len": voxel.len,
                           "R": round(voxel.R),
                           "G": round(voxel.G),
                           "B": round(voxel.B),
                           "scan_id": scan.id,
                           "vxl_mdl_id": voxel.vxl_mdl_id
                           })
            voxel_counter += 1
        with engine.connect() as db_connection:
            db_connection.execute(Tables.scans_db_table.insert(), scans)
            db_connection.execute(Tables.voxels_db_table.insert(), voxels)
            db_connection.commit()
        self.voxel_model.len = voxel_counter
        update_voxel_model_in_db_from_voxel_model(self.voxel_model)


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


class TriangulatorABC(ABC):

    def __init__(self, scan):
        self.scan = scan

    @abstractmethod
    def triangulate(self):
        pass


class ScipyTriangulator(TriangulatorABC):

    def __init__(self, scan):
        super().__init__(scan=scan)
        self.points_id = None
        self.vertices = None
        self.vertices_colors = None
        self.faces = None
        self.face_colors = None

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[Name: {self.scan.scan_name}\t\t"
                f"Count_of_triangles: {len(self.faces)}]"
                )

    def __get_data_dict(self):
        """
        Возвращает словарь с данными для построения триангуляции
        :return: словарь с данными для построения триангуляции
        """
        point_id_lst, x_lst, y_lst, z_lst, c_lst = [], [], [], [], []
        for point in self.scan:
            point_id_lst.append(point.id)
            x_lst.append(point.X)
            y_lst.append(point.Y)
            z_lst.append(point.Z)
            c_lst.append([point.R, point.G, point.B])
        return {"id": point_id_lst, "x": x_lst, "y": y_lst, "z": z_lst, "color": c_lst}

    def __calk_delone_triangulation(self):
        """
        Рассчитываает треугольники между точками
        :return: славарь с указанием вершин треугольников
        """
        points2D = self.vertices[:, :2]
        tri = Delaunay(points2D)
        i_lst, j_lst, k_lst = ([triplet[c] for triplet in tri.simplices] for c in range(3))
        return {"i_lst": i_lst, "j_lst": j_lst, "k_lst": k_lst, "ijk": tri.simplices}

    @staticmethod
    def __calk_faces_colors(ijk_dict, scan_data):
        """
        Рассчитывает цвета треугольников на основании усреднения цветов точек, образующих
        треугольник
        :param ijk_dict: словарь с вершинами треугольников
        :param scan_data: словарь с данными о точках скана
        :return: список цветов треугольников в формате [r, g, b] от 0 до 255
        """
        c_lst = []
        for idx in range(len(ijk_dict["i_lst"])):
            c_i = scan_data["color"][ijk_dict["i_lst"][idx]]
            c_j = scan_data["color"][ijk_dict["j_lst"][idx]]
            c_k = scan_data["color"][ijk_dict["k_lst"][idx]]
            r = round((c_i[0] + c_j[0] + c_k[0]) / 3)
            g = round((c_i[1] + c_j[1] + c_k[1]) / 3)
            b = round((c_i[2] + c_j[2] + c_k[2]) / 3)
            c_lst.append([r, g, b])
        return c_lst

    def triangulate(self):
        scan_data = self.__get_data_dict()
        self.vertices = np.vstack([scan_data["x"], scan_data["y"], scan_data["z"]]).T
        self.vertices_colors = scan_data["color"]
        self.points_id = scan_data["id"]
        tri_data_dict = self.__calk_delone_triangulation()
        self.faces = tri_data_dict["ijk"]
        self.face_colors = self.__calk_faces_colors(tri_data_dict, scan_data)
        return self


class MeshABC:

    def __init__(self, scan, scan_triangulator=ScipyTriangulator):
        self.scan = scan
        self.scan_triangulator = scan_triangulator
        self.mesh_name = self.__name_generator()
        self.len = 0
        self.r = None
        self.mse = None

    @abstractmethod
    def __iter__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[mesh_name: {self.mesh_name},\tlen: {self.len} r: {self.r} mse: {self.mse}]"

    def __name_generator(self):
        return f"MESH_{self.scan.scan_name}"

    def calk_mesh_mse(self, mesh_segment_model, base_scan=None):
        if base_scan is None:
            base_scan = ScanDB.get_scan_from_id(mesh_segment_model.voxel_model.base_scan_id)
        triangles = {}
        for point in base_scan:
            point.id = None
            cell = mesh_segment_model.get_model_element_for_point(point)
            if cell is None or len(cell.triangles) == 0:
                continue
            for triangle in cell.triangles:
                if triangle.is_point_in_triangle(point):
                    if point not in [triangle.point_0, triangle.point_1, triangle.point_2]:
                        try:
                            triangle.vv += (triangle.get_z_from_xy(point.X, point.Y) - point.Z) ** 2
                            triangle.r += 1
                        except (AttributeError, TypeError):
                            triangle.vv = (triangle.get_z_from_xy(point.X, point.Y) - point.Z) ** 2
                            triangle.r = 1
                    triangles[triangle.id] = triangle
                    break
                else:
                    continue
        for triangle in triangles.values():
            if triangle.r is not None:
                try:
                    triangle.mse = (triangle.vv / triangle.r) ** 0.5
                except AttributeError:
                    triangle.mse = None
                    triangle.r = None
        svv, sr = 0, 0
        for triangle in triangles.values():
            try:
                svv += triangle.r * triangle.mse ** 2
                sr += triangle.r
            except TypeError:
                continue
        self.mse = (svv / sr) ** 0.5
        self.r = sr
        return triangles.values()


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


class CellABC(ABC):
    """
    Абстрактный класс ячейки сегментированной модели
    """

    def __str__(self):
        return f"{self.__class__.__name__} [id: {self.voxel_id}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.voxel_id}]"

    @abstractmethod
    def get_z_from_xy(self, x, y):
        """
        Рассчитывает отметку точки (x, y) в ячейке
        :param x: координата x
        :param y: координата y
        :return: координата z для точки (x, y)
        """
        pass

    @abstractmethod
    def get_mse_z_from_xy(self, x, y):
        """
        Рассчитывает СКП отметки точки (x, y) в ячейке
        :param x: координата x
        :param y: координата y
        :return: СКП координаты z для точки (x, y)
        """
        pass

    @abstractmethod
    def get_db_raw_data(self):
        pass

    @abstractmethod
    def _save_cell_data_in_db(self, db_connection):
        """
        Сохраняет данные ячейки из модели в БД
        :param db_connection: открытое соединение с БД
        :return: None
        """
        pass

    def _load_cell_data_from_db(self, db_connection):
        """
        Загружает данные ячейки из БД в модель
        :param db_connection: открытое соединение с БД
        :return: None
        """
        select_ = select(self.db_table) \
            .where(and_(self.db_table.c.voxel_id == self.voxel.id,
                        self.db_table.c.base_model_id == self.dem_model.id))
        db_cell_data = db_connection.execute(select_).mappings().first()
        if db_cell_data is not None:
            self._copy_cell_data(db_cell_data)

    @abstractmethod
    def _copy_cell_data(self, db_cell_data):
        """
        Копирует данные из записи БД в атрибуты ячейки
        :param db_cell_data: загруженные из БД данные
        :return: None
        """
        pass


class MeshCellDB(CellABC):
    db_table = Tables.mesh_cell_db_table

    def __init__(self, voxel, dem_model):
        self.voxel = voxel
        self.dem_model = dem_model
        self.voxel_id = None
        self.count_of_mesh_points = 0
        self.count_of_triangles = 0
        self.r = None
        self.mse = None
        self.points = []
        self.triangles = []

    def get_z_from_xy(self, x, y):
        """
        Рассчитывает отметку точки (x, y) в ячейке
        :param x: координата x
        :param y: координата y
        :return: координата z для точки (x, y)
        """
        point = Point(x, y, 0, 0, 0, 0)
        for triangle in self.triangles:
            if triangle.is_point_in_triangle(point):
                return triangle.get_z_from_xy(x, y)
        return None

    def get_mse_z_from_xy(self, x, y):
        """
        Рассчитывает СКП отметки точки (x, y) в ячейке
        :param x: координата x
        :param y: координата y
        :return: СКП координаты z для точки (x, y)
        """
        point = Point(x, y, 0, 0, 0, 0)
        for triangle in self.triangles:
            if triangle.is_point_in_triangle(point):
                return triangle.mse

    def get_db_raw_data(self):
        return {"voxel_id": self.voxel.id,
                "base_model_id": self.dem_model.id,
                "count_of_mesh_points": self.count_of_mesh_points,
                "count_of_triangles": self.count_of_triangles,
                "r": self.r,
                "mse": self.mse}

    def _save_cell_data_in_db(self, db_connection):
        """
        Сохраняет данные ячейки из модели в БД
        :param db_connection: открытое соединение с БД
        :return: None
        """
        stmt = insert(self.db_table).values(voxel_id=self.voxel.id,
                                            base_model_id=self.dem_model.id,
                                            count_of_mesh_points=self.count_of_mesh_points,
                                            count_of_triangles=self.count_of_triangles,
                                            r=self.r,
                                            mse=self.mse,
                                            )
        db_connection.execute(stmt)

    def _copy_cell_data(self, db_cell_data):
        """
        Копирует данные из записи БД в атрибуты ячейки
        :param db_cell_data: загруженные из БД данные
        :return: None
        """
        self.voxel_id = db_cell_data["voxel_id"]
        self.base_model_id = db_cell_data["base_model_id"]
        self.count_of_mesh_points = db_cell_data["count_of_mesh_points"]
        self.count_of_triangles = db_cell_data["count_of_triangles"]
        self.r = db_cell_data["r"]
        self.mse = db_cell_data["mse"]


class SegmentedModelABC(ABC):
    """
    Абстрактный класс сегментированной модели
    """

    logger = logging.getLogger(LOGGER)
    db_table = Tables.dem_models_db_table

    def __init__(self, voxel_model, element_class):
        self.base_voxel_model_id = voxel_model.id
        self.voxel_model = voxel_model
        self._model_structure = {}
        self._create_model_structure(element_class)
        self.__init_model()

    def __iter__(self):
        return iter(self._model_structure.values())

    def __str__(self):
        return f"{self.__class__.__name__} [ID: {self.id},\tmodel_name: {self.model_name}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def get_z_from_point(self, point):
        cell = self.get_model_element_for_point(point)
        try:
            z = cell.get_z_from_xy(point.X, point.Y)
        except AttributeError:
            z = None
        return z

    @abstractmethod
    def _calk_segment_model(self):
        """
        Метод определяющий логику создания конкретной модели
        :return: None
        """
        pass

    def _create_model_structure(self, element_class):
        """
        Создание структуры сегментированной модели
        :param element_class: Класс ячейки конкретной модели
        :return: None
        """
        for voxel in self.voxel_model:
            model_key = f"{voxel.X:.5f}_{voxel.Y:.5f}_{voxel.Z:.5f}"
            self._model_structure[model_key] = element_class(voxel, self)

    def get_model_element_for_point(self, point):
        """
        Возвращает ячейку содержащую точку point
        :param point: точка для которой нужна соответствующая ячейка
        :return: объект ячейки модели, содержащая точку point
        """
        X = point.X // self.voxel_model.step * self.voxel_model.step
        Y = point.Y // self.voxel_model.step * self.voxel_model.step
        if self.voxel_model.is_2d_vxl_mdl is False:
            Z = point.Z // self.voxel_model.step * self.voxel_model.step
        else:
            Z = self.voxel_model.min_Z
        model_key = f"{X:.5f}_{Y:.5f}_{Z:.5f}"
        return self._model_structure.get(model_key, None)

    def _calk_model_mse(self, db_connection):
        """
        Расчитывает СКП всей модели по СКП отдельных ячеек
        :param db_connection: открытое соединение с БД
        :return: None
        """
        vv = 0
        sum_of_r = 0
        for cell in self:
            if cell.r > 0 and cell.mse is not None:
                vv += (cell.mse ** 2) * cell.r
                sum_of_r += cell.r
        try:
            self.mse_data = (vv / sum_of_r) ** 0.5
        except ZeroDivisionError:
            self.mse_data = None
        stmt = update(self.db_table).values(MSE_data=self.mse_data).where(self.db_table.c.id == self.id)
        db_connection.execute(stmt)
        db_connection.commit()
        self.logger.info(f"Расчет СКП модели {self.model_name} завершен и загружен в БД")

    def _load_cell_data_from_db(self, db_connection):
        """
        Загружает данные всех ячеек модели из БД
        :param db_connection: открытое соединение с БД
        :return: None
        """
        for cell in self._model_structure.values():
            cell._load_cell_data_from_db(db_connection)

    def _save_cell_data_in_db(self, db_connection):
        """
        Сохраняет данные из всех ячеек модели в БД
        :param db_connection: открытое соединение с БД
        :return: None
        """
        for cell in self._model_structure.values():
            cell._save_cell_data_in_db(db_connection)

    def _get_last_model_id(self):
        """
        Возвращает последний id для сегментированной модели в таблице БД dem_models
        :return: последний id для сегментированной модели в таблице БД dem_models
        """
        with engine.connect() as db_connection:
            stmt = (select(self.db_table.c.id).order_by(desc("id")))
            last_model_id = db_connection.execute(stmt).first()
            if last_model_id:
                return last_model_id[0]
            else:
                return 0

    def _copy_model_data(self, db_model_data: dict):
        """
        Копирует данные из записи БД в атрибуты сегментированной модели
        :param db_model_data: Данные записи из БД
        :return: None
        """
        self.id = db_model_data["id"]
        self.base_voxel_model_id = db_model_data["base_voxel_model_id"]
        self.model_type = db_model_data["model_type"]
        self.model_name = db_model_data["model_name"]
        self.mse_data = db_model_data["MSE_data"]

    def __init_model(self):
        """
        Инициализирует сегментированную модель при запуске
        Если модель для воксельной модели нужного типа уже есть в БД - запускает
        копирование данных из БД в атрибуты модели
        Если такой модели нет - создает новую модели и запись в БД
        :return: None
        """
        select_ = select(self.db_table) \
            .where(and_(self.db_table.c.base_voxel_model_id == self.voxel_model.id,
                        self.db_table.c.model_type == self.model_type))

        with engine.connect() as db_connection:
            db_model_data = db_connection.execute(select_).mappings().first()
            if db_model_data is not None:
                self._copy_model_data(db_model_data)
                self._load_cell_data_from_db(db_connection)
                self.logger.info(f"Загрузка {self.model_name} модели завершена")
            else:
                stmt = insert(self.db_table).values(base_voxel_model_id=self.voxel_model.id,
                                                    model_type=self.model_type,
                                                    model_name=self.model_name,
                                                    MSE_data=self.mse_data
                                                    )
                db_connection.execute(stmt)
                db_connection.commit()
                self.id = self._get_last_model_id()
                self._calk_segment_model()
                self._calk_model_mse(db_connection)
                self._save_cell_data_in_db(db_connection)
                db_connection.commit()
                self.logger.info(f"Расчет модели {self.model_name} завершен и загружен в БД\n")

    def _calk_cell_mse(self, base_scan):
        """
        Расчитываает СКП в ячейках сегментированной модели от точек базового скана
        :param base_scan: базовый скан из воксельной модели
        :return: None
        """
        for point in base_scan:
            try:
                cell = self.get_model_element_for_point(point)
                cell_z = cell.get_z_from_xy(point.X, point.Y)
                if cell_z is None:
                    continue
            except AttributeError:
                continue
            try:
                cell.vv += (point.Z - cell_z) ** 2
            except AttributeError:
                cell.vv = (point.Z - cell_z) ** 2

        for cell in self:
            if cell.r > 0:
                try:
                    cell.mse = (cell.vv / cell.r) ** 0.5
                except AttributeError:
                    cell.mse = None
        self.logger.info(f"Расчет СКП высот в ячейках модели {self.model_name} завершен")


class MeshSegmentModelDB(SegmentedModelABC):

    def __init__(self, voxel_model, mesh):
        self.model_type = "MESH"
        self.model_name = f"{self.model_type}_from_{voxel_model.vm_name}"
        self.mse_data = None
        self.cell_type = MeshCellDB
        self.mesh = mesh
        super().__init__(voxel_model, self.cell_type)

    def _calk_segment_model(self):
        """
        Метод определяющий логику создания конкретной модели
        :return: None
        """
        self.logger.info(f"Начат расчет модели {self.model_name}")
        base_scan = ScanDB.get_scan_from_id(self.voxel_model.base_scan_id)
        self._calk_cell_mse(base_scan)

    def _calk_cell_mse(self, base_scan):
        """
        Расчитываает СКП в ячейках сегментированной модели от точек базового скана
        :param base_scan: базовый скан из воксельной модели
        :return: None
        """
        for point in base_scan:
            cell = self.get_model_element_for_point(point)
            if cell is None:
                continue
            cell_z = cell.get_z_from_xy(point.X, point.Y)
            if cell_z is None:
                continue
            try:
                cell.scan_points_in_cell += 1
                cell.vv += (point.Z - cell_z) ** 2
            except AttributeError:
                cell.scan_points_in_cell = 1
                cell.vv = (point.Z - cell_z) ** 2
        for cell in self:
            try:
                cell.r = cell.scan_points_in_cell - cell.count_of_mesh_points
                if cell.r > 0:
                    cell.mse = (cell.vv / cell.r) ** 0.5
            except AttributeError:
                cell.r = 0
                cell.mse = None
        self.logger.info(f"Расчет СКП высот в ячейках модели {self.model_name} завершен")

    def _create_model_structure(self, element_class):
        """
        Создание структуры сегментированной модели
        :param element_class: Класс ячейки конкретной модели
        :return: None
        """
        for voxel in self.voxel_model:
            model_key = f"{voxel.X:.5f}_{voxel.Y:.5f}_{voxel.Z:.5f}"
            self._model_structure[model_key] = element_class(voxel, self)
        self._sort_points_and_triangles_to_cells()

    def _sort_points_and_triangles_to_cells(self):
        for triangle in self.mesh:
            lines = [Line(triangle.point_0, triangle.point_1),
                     Line(triangle.point_1, triangle.point_2),
                     Line(triangle.point_2, triangle.point_0)]
            for line in lines:
                cross_points = line.get_grid_cross_points_list(self.voxel_model.step)
                mid_points = []
                for idx in range(len(cross_points) - 1):
                    mid_x = (cross_points[idx].X + cross_points[idx + 1].X) / 2
                    mid_y = (cross_points[idx].Y + cross_points[idx + 1].Y) / 2
                    mid_points.append(Point(X=mid_x, Y=mid_y, Z=0, R=0, G=0, B=0))
                for point in mid_points:
                    cell = self.get_model_element_for_point(point)
                    if cell is None:
                        continue
                    if triangle not in cell.triangles:
                        cell.triangles.append(triangle)
                        cell.count_of_triangles += 1
            for point in triangle:
                cell = self.get_model_element_for_point(point)
                if cell is not None:
                    if point not in cell.points:
                        cell.points.append(point)
                        cell.count_of_mesh_points += 1
                    if triangle not in cell.triangles:
                        cell.triangles.append(triangle)
                        cell.count_of_triangles += 1


class CsvMeshDataExporter:

    def __init__(self, mesh):
        self.mesh = mesh
        self.file_name = f"{self.mesh.mesh_name}.csv"

    def export_mesh_data(self):
        with open(self.file_name, "w", newline="") as csvfile:
            fieldnames = ["point_0", "point_1", "point_2", "area", "r", "rmse"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for triangle in self.mesh:
                data = {"point_0": self.__point_data_exporter(triangle.point_0),
                        "point_1": self.__point_data_exporter(triangle.point_1),
                        "point_2": self.__point_data_exporter(triangle.point_2),
                        "area": triangle.get_area(),
                        "r": triangle.r,
                        "rmse": triangle.mse}
                writer.writerow(data)
        return self.file_name

    @staticmethod
    def __point_data_exporter(point):
        return {"XYZ": [point.X, point.Y, point.Z],
                "RGB": [point.R, point.G, point.B]}


class MeshStatisticCalculator:

    def __init__(self, mesh):
        sns.set_style("darkgrid")
        self.mesh = mesh
        try:
            self.df = pd.read_csv(f"{self.mesh.mesh_name}.csv", delimiter=",")[["area", "r", "rmse"]]
        except FileNotFoundError:
            file_name = CsvMeshDataExporter(mesh).export_mesh_data()
            self.df = pd.read_csv(file_name, delimiter=",")[["area", "r", "rmse"]]
            os.remove(file_name)

    def save_statistic(self):
        statistic = self.df.describe()
        total_area = self.df["area"].sum()
        statistic.loc["TOTAL_MESH"] = [total_area, self.mesh.r, self.mesh.mse]
        statistic.to_csv(f"{self.mesh.mesh_name}_statistics.csv")
        return {"Total_area": total_area,
                "Count_of_r": self.mesh.r,
                "Cloud_MSE": self.mesh.mse,
                "Min_MSE": statistic["rmse"]["min"],
                "Max_MSE": statistic["rmse"]["max"],
                "Median_MSE": statistic["rmse"]["50%"]}

    def save_area_distributions_histograms(self):
        sns_plot = sns.displot(self.df, x="area", kde=True)
        sns_plot.savefig(f"{self.mesh.mesh_name}_area_distribution.png")

    def save_r_distributions_histograms(self):
        sns_plot = sns.displot(self.df, x="r", kde=True)
        sns_plot.savefig(f"{self.mesh.mesh_name}_r_distribution.png")

    def save_rmse_distributions_histograms(self):
        sns_plot = sns.displot(self.df, x="rmse", kde=True)
        sns_plot.savefig(f"{self.mesh.mesh_name}_rmse_distribution.png")

    def save_pair_plot_distributions_histograms(self):
        pair_grid = sns.PairGrid(self.df)
        pair_grid.map_upper(sns.histplot)
        pair_grid.map_lower(sns.kdeplot, fill=True)
        pair_grid.map_diag(sns.histplot, kde=True)
        pair_grid.savefig(f"{self.mesh.mesh_name}_pair_grid.png")


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


class PlyMeshExporter(MeshExporterABC):

    def __init__(self, mesh):
        super().__init__(mesh)

    def __create_header(self):
        return f"ply\n" \
               f"format ascii 1.0\n" \
               f"comment author: Mikhail Vystrchil\n" \
               f"comment object: {self.mesh.mesh_name}\n" \
               f"element vertex {len(self.vertices)}\n" \
               f"property float x\n" \
               f"property float y\n" \
               f"property float z\n" \
               f"property uchar red\n" \
               f"property uchar green\n" \
               f"property uchar blue\n" \
               f"element face {len(self.faces)}\n" \
               f"property list uchar int vertex_index\n" \
               f"end_header\n"

    def __create_vertices_str(self):
        vertices_str = ""
        for idx in range(len(self.vertices)):
            vertices_str += f"{self.vertices[idx][0]} {self.vertices[idx][1]} {self.vertices[idx][2]} " \
                            f"{self.vertices_colors[idx][0]} " \
                            f"{self.vertices_colors[idx][1]} " \
                            f"{self.vertices_colors[idx][2]}\n"
        return vertices_str

    def __create_faces_str(self):
        faces_str = ""
        for face in self.faces:
            faces_str += f"3 {face[0]} {face[1]} {face[2]}\n"
        return faces_str

    def _save_ply(self, file_path):
        with open(file_path, "wb") as file:
            file.write(self.__create_header().encode("ascii"))
            file.write(self.__create_vertices_str().encode("ascii"))
            file.write(self.__create_faces_str().encode("ascii"))

    def export(self, file_path="."):
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name.replace(':', '=')}.ply")
        self._save_ply(file_path)


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
        file_path = os.path.join(file_path, f"MSE_{self.mesh.mesh_name.replace(':', '=')}"
                                         f"_MseLimits=[{self.min_mse:.3f}-{self.max_mse:.3f}].ply")
        self._save_ply(file_path)


class UiPointCloudAnalizer(QWidget):

    def __init__(self):
        super().__init__()
        self.setupUi()
        self.base_filepath = None
        self.sample_filepath = None
        self.base_file_path_button.clicked.connect(self.open_file_dialog_base_filepath)
        self.base_file_path_text.textChanged.connect(self.base_filepath_from_text_line)
        self.sample_file_path_button.clicked.connect(self.open_file_dialog_sample_filepath)
        self.sample_file_path_text.textChanged.connect(self.sample_filepath_from_text_line)
        self.cb_main_graf_distr.toggled.connect(self.cb_mse_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(self.cb_r_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(self.cb_area_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(self.cb_pair_plot_graf.setEnabled)
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_mse_graf.setChecked(False))
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_r_graf.setChecked(False))
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_area_graf.setChecked(False))
        self.cb_main_graf_distr.toggled.connect(lambda checked: not checked and
                                                                self.cb_pair_plot_graf.setChecked(False))
        self.cb_main_tin_surface.toggled.connect(self.sb_min_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(self.label_min_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(self.sb_max_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(self.label_max_mse_tin_surf.setEnabled)
        self.cb_main_tin_surface.toggled.connect(lambda checked: not checked and
                                                                 self.sb_min_mse_tin_surf.setValue(0.00))
        self.cb_main_tin_surface.toggled.connect(lambda checked: not checked and
                                                                 self.sb_max_mse_tin_surf.setValue(99.99))
        self.progress = 0
        self.start_button.clicked.connect(self.start_calculation)

    def start_calculation(self):
        self.start_button.setEnabled(False)
        self.result_table.setEnabled(False)
        self.progress = 0
        self.progressBar.setProperty("value", 0)
        create_db()
        base_file_name = os.path.basename(self.base_filepath).split(".")[0]
        sample_file_name = os.path.basename(self.sample_filepath).split(".")[0]
        self.progressBar.setProperty("value", 1)
        base_scan = ScanDB(f"Base_scan_{base_file_name}")
        base_scan.load_scan_from_file(file_name=self.base_filepath)
        self.progressBar.setProperty("value", 15)
        sampled_scan = ScanDB(sample_file_name)
        sampled_scan.load_scan_from_file(file_name=self.sample_filepath)
        self.progressBar.setProperty("value", 20)
        vm = VoxelModelDB(base_scan,
                          self.__calk_voxel_size(base_scan, sampled_scan),
                          dx=0, dy=0, dz=0, is_2d_vxl_mdl=True)
        self.progressBar.setProperty("value", 40)
        mesh = MeshLite(sampled_scan)
        self.progressBar.setProperty("value", 50)
        mesh_sm = MeshSegmentModelDB(vm, mesh)
        self.progressBar.setProperty("value", 70)
        mesh.calk_mesh_mse(mesh_sm)
        self.progressBar.setProperty("value", 80)
        ####################################################
        csv_exp = CsvMeshDataExporter(mesh)
        csv_exp.export_mesh_data()
        stat_calculator = MeshStatisticCalculator(mesh)
        stat_dict = stat_calculator.save_statistic()
        self.progressBar.setProperty("value", 85)
        ####################################################
        if self.cb_mse_graf.isChecked():
            stat_calculator.save_rmse_distributions_histograms()
        if self.cb_r_graf.isChecked():
            stat_calculator.save_r_distributions_histograms()
        if self.cb_area_graf.isChecked():
            stat_calculator.save_area_distributions_histograms()
        if self.cb_pair_plot_graf.isChecked():
            stat_calculator.save_pair_plot_distributions_histograms()
        self.progressBar.setProperty("value", 95)
        ####################################################
        if self.cb_main_tin_surface.isChecked():
            min_mse = None if self.sb_min_mse_tin_surf.value() == 0 else self.sb_min_mse_tin_surf.value()
            max_mse = None if self.sb_max_mse_tin_surf.value() == 99.99 else self.sb_max_mse_tin_surf.value()
            PlyMseMeshExporter(mesh, min_mse=min_mse, max_mse=max_mse).export()
        self.progressBar.setProperty("value", 100)
        ###################################################
        if self.cb_save_full_tin_csv_log.isChecked() is False:
            os.remove(csv_exp.file_name)
        if self.cb_save_db.isChecked() is False:
            engine.dispose()
            os.remove(os.path.join(".", DATABASE_NAME))
        ###################################################
        self.result_table.setEnabled(True)
        self.result_table.setItem(0, 0, QTableWidgetItem(str(round(stat_dict["Total_area"], 4))))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(stat_dict["Count_of_r"])))
        self.result_table.setItem(0, 2, QTableWidgetItem(str(round(stat_dict["Cloud_MSE"], 4))))
        self.result_table.setItem(0, 3, QTableWidgetItem(str(round(stat_dict["Min_MSE"], 4))))
        self.result_table.setItem(0, 4, QTableWidgetItem(str(round(stat_dict["Max_MSE"], 4))))
        self.result_table.setItem(0, 5, QTableWidgetItem(str(round(stat_dict["Median_MSE"], 4))))
        ###################################################
        self.start_button.setEnabled(True)
        dig = QMessageBox(self)
        dig.setWindowTitle("Result")
        dig.setText("Расчет завершен!")
        dig.setIcon(QMessageBox.Icon.Information)
        dig.exec()

    @staticmethod
    def __calk_voxel_size(base_scan, sampled_scan):
        area = (base_scan.max_X - base_scan.min_X) * (base_scan.max_Y - base_scan.min_Y)
        voxel_size = area / sampled_scan.len
        voxel_size = round((voxel_size // 0.05 + 1) * 0.05, 2)
        return voxel_size

    def base_filepath_from_text_line(self):
        self.base_filepath = self.base_file_path_text.toPlainText()
        if self.base_filepath and self.sample_filepath:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def sample_filepath_from_text_line(self):
        self.sample_filepath = self.sample_file_path_text.toPlainText()
        if self.base_filepath and self.sample_filepath:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def open_file_dialog_base_filepath(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            ".",
            "PointCloud (*.txt *.ascii)"
        )
        if filename:
            path_ = Path(filename)
            self.base_file_path_text.setText(str(filename))
            self.base_filepath = str(path_)

    def open_file_dialog_sample_filepath(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            ".",
            "PointCloud (*.txt *.ascii)"
        )
        if filename:
            path_ = Path(filename)
            self.sample_file_path_text.setText(str(filename))
            self.sample_filepath = str(path_)

    def setupUi(self):
        self.setWindowIcon(QIcon("icon.ico"))
        self.setObjectName("PointCloudAnalizer")
        self.setEnabled(True)
        self.resize(925, 466)
        self.setMinimumSize(QtCore.QSize(925, 0))
        self.setMaximumSize(QtCore.QSize(925, 466))
        self.horizontalLayout_5 = QHBoxLayout(self)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.base_file_path_button = QToolButton(parent=self)
        self.base_file_path_button.setObjectName("base_file_path_button")
        self.gridLayout_4.addWidget(self.base_file_path_button, 0, 2, 1, 1)
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.line = QFrame(parent=self)
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 1, 0, 1, 1)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                           QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.label_2 = QLabel(parent=self)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        self.label_8 = QLabel(parent=self)
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 16))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.cb_main_graf_distr = QCheckBox(parent=self)
        self.cb_main_graf_distr.setMaximumSize(QtCore.QSize(16777215, 16))
        self.cb_main_graf_distr.setObjectName("cb_main_graf_distr")
        self.gridLayout_3.addWidget(self.cb_main_graf_distr, 1, 1, 1, 1)
        spacerItem2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 0, 0, 1, 1)
        self.label = QLabel(parent=self)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 1, 1, 1)
        spacerItem3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 0, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_3, 0, 2, 1, 1)
        self.line_3 = QFrame(parent=self)
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_2.addWidget(self.line_3, 1, 3, 1, 1)
        self.cb_r_graf = QCheckBox(parent=self)
        self.cb_r_graf.setEnabled(False)
        self.cb_r_graf.setObjectName("cb_r_graf")
        self.gridLayout_2.addWidget(self.cb_r_graf, 3, 2, 1, 1)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_max_mse_tin_surf = QLabel(parent=self)
        self.label_max_mse_tin_surf.setEnabled(False)
        self.label_max_mse_tin_surf.setObjectName("label_max_mse_tin_surf")
        self.horizontalLayout_4.addWidget(self.label_max_mse_tin_surf)
        self.sb_max_mse_tin_surf = QDoubleSpinBox(parent=self)
        self.sb_max_mse_tin_surf.setEnabled(False)
        self.sb_max_mse_tin_surf.setMaximumSize(QtCore.QSize(48, 16777215))
        self.sb_max_mse_tin_surf.setSingleStep(0.1)
        self.sb_max_mse_tin_surf.setProperty("value", 99.99)
        self.sb_max_mse_tin_surf.setObjectName("sb_max_mse_tin_surf")
        self.horizontalLayout_4.addWidget(self.sb_max_mse_tin_surf)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 3, 3, 1, 1)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_min_mse_tin_surf = QLabel(parent=self)
        self.label_min_mse_tin_surf.setEnabled(False)
        self.label_min_mse_tin_surf.setObjectName("label_min_mse_tin_surf")
        self.horizontalLayout.addWidget(self.label_min_mse_tin_surf)
        self.sb_min_mse_tin_surf = QDoubleSpinBox(parent=self)
        self.sb_min_mse_tin_surf.setEnabled(False)
        self.sb_min_mse_tin_surf.setMaximumSize(QtCore.QSize(48, 16777215))
        self.sb_min_mse_tin_surf.setSingleStep(0.1)
        self.sb_min_mse_tin_surf.setObjectName("sb_min_mse_tin_surf")
        self.horizontalLayout.addWidget(self.sb_min_mse_tin_surf)
        self.gridLayout_2.addLayout(self.horizontalLayout, 2, 3, 1, 1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.cb_main_tin_surface = QCheckBox(parent=self)
        self.cb_main_tin_surface.setObjectName("cb_main_tin_surface")
        self.gridLayout_5.addWidget(self.cb_main_tin_surface, 1, 1, 1, 1)
        self.label_3 = QLabel(parent=self)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 0, 1, 1, 1)
        spacerItem4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Minimum)
        self.gridLayout_5.addItem(spacerItem4, 0, 0, 1, 1)
        spacerItem5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Minimum)
        self.gridLayout_5.addItem(spacerItem5, 0, 2, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_5)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 3, 1, 1)
        self.line_2 = QFrame(parent=self)
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 1, 2, 1, 1)
        self.cb_area_graf = QCheckBox(parent=self)
        self.cb_area_graf.setEnabled(False)
        self.cb_area_graf.setObjectName("cb_area_graf")
        self.gridLayout_2.addWidget(self.cb_area_graf, 4, 2, 1, 1)
        self.cb_save_db = QCheckBox(parent=self)
        self.cb_save_db.setEnabled(True)
        self.cb_save_db.setObjectName("cb_save_db")
        self.gridLayout_2.addWidget(self.cb_save_db, 3, 0, 1, 1)
        self.cb_mse_graf = QCheckBox(parent=self)
        self.cb_mse_graf.setEnabled(False)
        self.cb_mse_graf.setObjectName("cb_mse_graf")
        self.gridLayout_2.addWidget(self.cb_mse_graf, 2, 2, 1, 1)
        self.cb_save_full_tin_csv_log = QCheckBox(parent=self)
        self.cb_save_full_tin_csv_log.setEnabled(True)
        self.cb_save_full_tin_csv_log.setObjectName("cb_save_full_tin_csv_log")
        self.gridLayout_2.addWidget(self.cb_save_full_tin_csv_log, 2, 0, 1, 1)
        self.cb_pair_plot_graf = QCheckBox(parent=self)
        self.cb_pair_plot_graf.setEnabled(False)
        self.cb_pair_plot_graf.setObjectName("cb_pair_plot_graf")
        self.gridLayout_2.addWidget(self.cb_pair_plot_graf, 5, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 3, 1, 1, 1)
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem6 = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum,
                                            QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem6)
        self.base_file_path_text = QTextEdit(parent=self)
        self.base_file_path_text.setMaximumSize(QtCore.QSize(16777215, 25))
        self.base_file_path_text.setObjectName("base_file_path_text")
        self.verticalLayout_5.addWidget(self.base_file_path_text)
        spacerItem7 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum,
                                            QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem7)
        self.gridLayout_4.addLayout(self.verticalLayout_5, 0, 1, 1, 1)
        self.label_10 = QLabel(parent=self)
        self.label_10.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                   QtCore.Qt.AlignmentFlag.AlignTrailing |
                                   QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 1, 0, 1, 1)
        self.label_6 = QLabel(parent=self)
        self.label_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                  QtCore.Qt.AlignmentFlag.AlignTrailing |
                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 3, 0, 1, 1)
        self.label_7 = QLabel(parent=self)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 100))
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                  QtCore.Qt.AlignmentFlag.AlignTrailing |
                                  QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_4.addLayout(self.verticalLayout_2, 3, 2, 1, 1)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.sample_file_path_button = QToolButton(parent=self)
        self.sample_file_path_button.setObjectName("sample_file_path_button")
        self.verticalLayout_3.addWidget(self.sample_file_path_button)
        self.gridLayout_4.addLayout(self.verticalLayout_3, 1, 2, 1, 1)
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum,
                                            QSizePolicy.Policy.Expanding)
        self.verticalLayout_6.addItem(spacerItem8)
        self.sample_file_path_text = QTextEdit(parent=self)
        self.sample_file_path_text.setMaximumSize(QtCore.QSize(16777215, 25))
        self.sample_file_path_text.setObjectName("sample_file_path_text")
        self.verticalLayout_6.addWidget(self.sample_file_path_text)
        spacerItem9 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum,
                                            QSizePolicy.Policy.Expanding)
        self.verticalLayout_6.addItem(spacerItem9)
        self.gridLayout_4.addLayout(self.verticalLayout_6, 1, 1, 1, 1)
        self.line_4 = QFrame(parent=self)
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_4.addWidget(self.line_4, 2, 1, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_4)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.progressBar = QProgressBar(parent=self)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_3.addWidget(self.progressBar)
        self.start_button = QPushButton(parent=self)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("background-color: rgb(170, 255, 127);")
        self.start_button.setFlat(False)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout_3.addWidget(self.start_button)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.result_table = QTableWidget(parent=self)
        self.result_table.setEnabled(False)
        self.result_table.setObjectName("result_table")
        self.result_table.setColumnCount(6)
        self.result_table.setRowCount(1)
        item = QTableWidgetItem()
        self.result_table.setVerticalHeaderItem(0, item)
        item = QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(0, item)
        item = QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(1, item)
        item = QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(2, item)
        item = QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(3, item)
        item = QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(4, item)
        item = QTableWidgetItem()
        self.result_table.setHorizontalHeaderItem(5, item)
        self.verticalLayout_4.addWidget(self.result_table)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.result_table.setColumnWidth(0, 150)
        self.result_table.setColumnWidth(1, 150)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("PointCloudAnalizer", "PointCloudAnalizer"))
        self.base_file_path_button.setText(_translate("PointCloudAnalizer", "..."))
        self.label_2.setText(_translate("PointCloudAnalizer", "Сохраняемая информация"))
        self.cb_main_graf_distr.setText(_translate("PointCloudAnalizer", "Построить графики распределения"))
        self.label.setText(_translate("PointCloudAnalizer", "Графики распределения параметров\nTIN поверхности"))
        self.cb_r_graf.setText(_translate("PointCloudAnalizer", "Распределение количества\nизбыточных данных"))
        self.label_max_mse_tin_surf.setText(_translate("PointCloudAnalizer", "Максимальное значение шкалы СКП"))
        self.label_min_mse_tin_surf.setText(_translate("PointCloudAnalizer", "Минимальное значение шкалы СКП"))
        self.cb_main_tin_surface.setText(_translate("PointCloudAnalizer", "Создать"))
        self.label_3.setText(_translate("PointCloudAnalizer", "Создать TIN поверхность\nраспределения СКП "))
        self.cb_area_graf.setText(_translate("PointCloudAnalizer",
                                             "Распределение площади\nтреугольников в поверхности"))
        self.cb_save_db.setText(_translate("PointCloudAnalizer", "Сохранить служебную базу данных"))
        self.cb_mse_graf.setText(_translate("PointCloudAnalizer", "Распределение СКП\nв треугольниках поверхности"))
        self.cb_save_full_tin_csv_log.setText(
            _translate("PointCloudAnalizer", "Сохранить полное описание анализируемой\nповерхности в CSV файле"))
        self.cb_pair_plot_graf.setText(_translate("PointCloudAnalizer", "Совмещенный график\nраспределения"))
        self.label_10.setText(_translate("PointCloudAnalizer", "Разреженное\nоблако:"))
        self.label_6.setText(_translate("PointCloudAnalizer", "Настройки\nвыводимой\nинформации"))
        self.label_7.setText(_translate("PointCloudAnalizer", "Исходное\nоблако:"))
        self.sample_file_path_button.setText(_translate("PointCloudAnalizer", "..."))
        self.start_button.setText(_translate("PointCloudAnalizer", "Запуск анализа"))
        item = self.result_table.verticalHeaderItem(0)
        item.setText(_translate("PointCloudAnalizer", "Рассчитанные значения"))
        item = self.result_table.horizontalHeaderItem(0)
        item.setText(_translate("PointCloudAnalizer", "Общая площадь, м2"))
        item = self.result_table.horizontalHeaderItem(1)
        item.setText(_translate("PointCloudAnalizer", "Кол-во изб. данных"))
        item = self.result_table.horizontalHeaderItem(2)
        item.setText(_translate("PointCloudAnalizer", "СКП облака, м"))
        item = self.result_table.horizontalHeaderItem(3)
        item.setText(_translate("PointCloudAnalizer", "Мин. СКП, м"))
        item = self.result_table.horizontalHeaderItem(4)
        item.setText(_translate("PointCloudAnalizer", "Макс. СКП, м"))
        item = self.result_table.horizontalHeaderItem(5)
        item.setText(_translate("PointCloudAnalizer", "Медиана СКП, м"))


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    app = QApplication(sys.argv)
    ui = UiPointCloudAnalizer()
    ui.show()
    sys.exit(app.exec())
