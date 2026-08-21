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
    from app.db.models import Policy, Claim, ClaimPhoto, AnalysisResult, DecisionRecord, InvestigationCase, PolicyClause, User  # noqa: F401

    _rename_legacy_policy_clauses_table()
    Base.metadata.create_all(bind=engine)
    _restore_legacy_policy_clauses_rows()
    sync_sqlite_schema()


def _rename_legacy_policy_clauses_table() -> None:
    """policy_clauses originally had a UNIQUE constraint on clause_id alone
    (one row per clause, app-wide). Tracking clauses per claim needs
    uniqueness scoped to (claim_id, clause_id) instead, since the same
    clause is legitimately cited by many claims -- and SQLite can't ALTER
    a constraint in place, so the old table is renamed out of the way and
    create_all() (called right after this) creates a fresh one matching the
    current model. _restore_legacy_policy_clauses_rows() then copies the
    old rows back in with claim_id left NULL, since pre-migration rows
    never recorded which claim (if any) they came from."""
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as connection:
        existing_columns = {
            row[1] for row in connection.exec_driver_sql('PRAGMA table_info("policy_clauses")')
        }
        if not existing_columns or "claim_id" in existing_columns:
            return  # table doesn't exist yet, or already migrated
        connection.exec_driver_sql('DROP TABLE IF EXISTS "policy_clauses_legacy"')
        connection.exec_driver_sql('ALTER TABLE "policy_clauses" RENAME TO "policy_clauses_legacy"')


def _restore_legacy_policy_clauses_rows() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as connection:
        tables = {
            row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "policy_clauses_legacy" not in tables:
            return
        connection.exec_driver_sql(
            'INSERT INTO "policy_clauses" (policy_id, clause_id, text, clause_metadata, embedding_id) '
            'SELECT policy_id, clause_id, text, clause_metadata, embedding_id FROM "policy_clauses_legacy"'
        )
        connection.exec_driver_sql('DROP TABLE "policy_clauses_legacy"')


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
