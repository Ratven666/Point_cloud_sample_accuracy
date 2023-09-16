from sqlalchemy import Column, Table, Integer, ForeignKey


def create_points_scans_db_table(metadata):
    points_scans_db_table = Table("points_scans", metadata,
                                  Column("point_id", Integer, ForeignKey("points.id"), primary_key=True),
                                  Column("scan_id", Integer, ForeignKey("scans.id"), primary_key=True)
                                  )
    return points_scans_db_table
