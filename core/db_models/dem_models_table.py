from sqlalchemy import Table, Column, Integer, String, ForeignKey, Float


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
