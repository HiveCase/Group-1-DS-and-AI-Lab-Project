from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.schema import CreateColumn

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
if settings.database_url.startswith("sqlite:///"):
    db_path = settings.database_url.replace("sqlite:///", "", 1)
    if db_path != ":memory:":
        from pathlib import Path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    sync_sqlite_schema()


def sync_sqlite_schema():
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            existing_columns = {
                row[1]
                for row in connection.exec_driver_sql(f'PRAGMA table_info("{table.name}")')
            }
            for column in table.columns:
                if column.name in existing_columns or column.primary_key:
                    continue
                if not column.nullable and column.default is None and column.server_default is None:
                    continue
                column_sql = str(CreateColumn(column).compile(dialect=engine.dialect))
                connection.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN {column_sql}')
