import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.core.config import get_config
from src.core.logging import setup_logging


logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = f"""
    CREATE TABLE IF NOT EXISTS {get_config().sqlite_data_table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crim REAL,
        zn REAL,
        indus REAL ,
        chas INTEGER,
        nox REAL,
        rm REAL,
        age REAL,
        dis REAL,
        rad INTEGER,
        tax REAL,
        ptratio REAL,
        b REAL,
        lstat REAL,
        medv REAL
    );
    """


def check_csv_file(path: Path) -> bool:

    if path.exists():
        logger.info("CSV existente en %s", path)
        return True

    raise ValueError(f"CSV no existente en: {path}")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Cargado %d filas y %d columns", len(df), len(df.columns))
    logger.debug("Columnas: %s", list(df.columns))
    return df


def migrate(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    sqllite_table_name = get_config().sqlite_data_table_name

    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {sqllite_table_name}")
        conn.executescript(CREATE_TABLE_SQL)
        df.to_sql(sqllite_table_name, conn, if_exists="append", index=False)
        count = conn.execute(f"SELECT COUNT(*) FROM {sqllite_table_name}").fetchone()[0]

    logger.info("Insertando %d elementos en %s::%s", count, db_path, sqllite_table_name)


def main() -> None:
    setup_logging()

    logger.info("Iniciando migración de DATA a SQLITE.")

    csv_path = Path(get_config().csv_path)
    sqlite_path = Path(get_config().sqlite_path)

    if check_csv_file(csv_path):
        df = load_csv(csv_path)
        migrate(df, sqlite_path)

    logger.info("Migración completa. SQLite en: %s", sqlite_path)


if __name__ == "__main__":
    main()
