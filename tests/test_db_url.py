from app.config import Settings
from app.db import build_engine_url


def test_sqlite_is_the_default(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url="",
        sqlite_path=str(tmp_path / "calls.db"),
    )
    url, connect_args = build_engine_url(settings)
    assert url.drivername == "sqlite+aiosqlite"
    assert str(tmp_path / "calls.db") in url.database
    assert connect_args == {}


def test_railway_postgres_url_enables_ssl_and_drops_sslmode(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url="postgres://user:secret@db.example.com:5432/railway?sslmode=require",
        sqlite_path=str(tmp_path / "unused.db"),
    )
    url, connect_args = build_engine_url(settings)
    assert url.drivername == "postgresql+asyncpg"
    assert url.username == "user"
    assert url.database == "railway"
    assert "sslmode" not in url.query
    assert connect_args["ssl"] is True
