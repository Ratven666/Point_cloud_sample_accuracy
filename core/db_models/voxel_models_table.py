from sqlalchemy import Table, Column, Integer, String, Float, ForeignKey


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
