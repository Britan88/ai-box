import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = {
    "cf": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "account_id": "TEXT",
        "email": "TEXT",
        "password": "TEXT",
        "api_key": "TEXT",
        "expire": "TEXT",
        "ai_quota": "TEXT",
        "global_api_key": "TEXT",
        "otp_secret": "TEXT",
        "recovery": "TEXT",
        "problems": "TEXT",
    },
    "aiio": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "email": "TEXT",
        "password": "TEXT",
        "api_key": "TEXT",
        "expire": "TEXT",
        "ai_quota": "TEXT",
        "global_api_key": "TEXT",
        "otp_secret": "TEXT",
        "recovery": "TEXT",
        "problems": "TEXT",
    },
    "models": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "provider": "TEXT",
        "type": "TEXT",
        "model": "TEXT",
        "status": "TEXT",
        "avtime": "INTEGER",
        "error": "TEXT",
        "result": "TEXT",
    },
}


def get_existing_columns(cursor: sqlite3.Cursor, table: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    logger.debug(f"[{table}] existing columns: {columns}")
    return columns


def get_existing_tables(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    logger.debug(f"existing tables: {tables}")
    return tables


def create_table(cursor: sqlite3.Cursor, table: str, columns: dict):
    columns_sql = ", ".join(f"{col} {typedef}" for col, typedef in columns.items())
    sql = f"CREATE TABLE {table} ({columns_sql})"
    logger.info(f"[{table}] creating table | sql: {sql}")
    cursor.execute(sql)


def add_missing_columns(
    cursor: sqlite3.Cursor, table: str, required: dict, existing: set[str]
):
    missing = {col: typedef for col, typedef in required.items() if col not in existing}
    logger.debug(f"[{table}] missing columns: {list(missing.keys()) or 'none'}")
    for col, typedef in missing.items():
        sql = f"ALTER TABLE {table} ADD COLUMN {col} {typedef}"
        logger.info(f"[{table}] adding column | sql: {sql}")
        cursor.execute(sql)


def sync_schema(db_path: str):
    logger.info(f"connecting to db | path: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing_tables = get_existing_tables(cursor)

    for table, columns in SCHEMA.items():
        if table not in existing_tables:
            create_table(cursor, table, columns)
        else:
            logger.info(f"[{table}] table exists, checking columns")
            existing_columns = get_existing_columns(cursor, table)
            add_missing_columns(cursor, table, columns, existing_columns)

    conn.commit()
    conn.close()
    logger.info("schema sync complete")
