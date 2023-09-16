from sqlalchemy import update

from core.utils.start_db import engine, Tables


def update_voxel_in_db_from_voxel(updated_voxel, db_connection=None):
    """
    Обновляет значения метрик вокселя в БД
    :param updated_voxel: Объект вокселя для которого обновляются метрики
    :param db_connection: Открытое соединение с БД
    :return: None
    """
    stmt = update(Tables.voxels_db_table) \
        .where(Tables.voxels_db_table.c.id == updated_voxel.id) \
        .values(id=updated_voxel.id,
                vxl_name=updated_voxel.vxl_name,
                X=updated_voxel.X,
                Y=updated_voxel.Y,
                Z=updated_voxel.Z,
                step=updated_voxel.step,
                len=updated_voxel.len,
                R=round(updated_voxel.R),
                G=round(updated_voxel.G),
                B=round(updated_voxel.B),
                scan_id=updated_voxel.scan_id,
                vxl_mdl_id=updated_voxel.vxl_mdl_id)
    if db_connection is None:
        with engine.connect() as db_connection:
            db_connection.execute(stmt)
            db_connection.commit()
    else:
        db_connection.execute(stmt)
        db_connection.commit()
