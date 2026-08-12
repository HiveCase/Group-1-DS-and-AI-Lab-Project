import os
import shutil
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_TEST_DB_PATH = _BACKEND_DIR / "data" / "ci_claims.db"
_TEST_UPLOAD_DIR = _BACKEND_DIR / "uploads-ci"

_TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
if _TEST_DB_PATH.exists():
    _TEST_DB_PATH.unlink()
if _TEST_UPLOAD_DIR.exists():
    shutil.rmtree(_TEST_UPLOAD_DIR)

# Must run before anything imports app.core.config / app.db.database --
# those bind a module-level SQLAlchemy engine to DATABASE_URL at import
# time. Without this override the suite writes directly into the real
# dev-mode backend/data/claims.db (which is exactly what happened once
# already: a full pytest run reset the live claims table because tests
# had no database of their own).
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH.as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TEST_UPLOAD_DIR)
