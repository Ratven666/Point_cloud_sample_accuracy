from sqlalchemy import Table, Column, Integer, String, Float, ForeignKey


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
