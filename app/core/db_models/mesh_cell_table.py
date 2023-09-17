from sqlalchemy import Table, Column, Integer, Float, ForeignKey


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
