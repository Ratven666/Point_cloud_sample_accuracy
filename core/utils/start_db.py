import logging
import os

from sqlalchemy import create_engine, MetaData

from core import db_models
from core.CONFIG import DATABASE_NAME

path = os.path.join(".", DATABASE_NAME)

engine = create_engine(f'sqlite:///{path}')

db_metadata = MetaData()

Tables = db_models.TableInitializer(db_metadata)

logger = logging.getLogger("console")


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
