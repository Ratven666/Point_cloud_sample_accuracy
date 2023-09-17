import os
import sqlite3

from app.core.CONFIG import DATABASE_NAME
from app.core.classes.Point import Point


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
