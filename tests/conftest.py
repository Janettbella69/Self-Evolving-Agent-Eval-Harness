"""Shared test setup.

backend/ is not a package: its modules import each other as top-level names
(`from storage import queries`), so it goes on sys.path the same way cli.py does.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


@pytest.fixture(autouse=True)
def eval_db(tmp_path, monkeypatch):
    """Point the storage layer at a throwaway SQLite file for every test.

    This patches the module attribute rather than the DB_PATH env var: config.py
    loads backend/.env with override=True, so a developer's own DB_PATH would win
    over anything set in the environment and tests would write into the real eval.db.
    """
    from storage import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "eval.db")
    monkeypatch.setattr(database, "_db", None)
    return tmp_path / "eval.db"
