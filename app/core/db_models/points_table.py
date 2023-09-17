from sqlalchemy import Column, Integer, Table, Float


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
