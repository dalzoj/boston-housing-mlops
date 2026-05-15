import argparse
import logging
import sqlite3
import warnings
from pathlib import Path

import pandas as pd
from copulas.multivariate import GaussianMultivariate

from src.core.config import get_config
from src.core.logging import setup_logging
from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)
setup_logging()
config = get_config()


CREATE_TABLE_SQL = f"""
    CREATE TABLE IF NOT EXISTS {config.sqlite_data_table_name} (
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
    logger.info("Cargado %d filas y %d columnas", len(df), len(df.columns))
    logger.debug("Columnas: %s", list(df.columns))
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Eliminadas %d filas con NaN", before - len(df))
    return df


def migrate(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {config.sqlite_data_table_name}")
        conn.executescript(CREATE_TABLE_SQL)
        df.to_sql(config.sqlite_data_table_name, conn, if_exists="append", index=False)
        count = conn.execute(f"SELECT COUNT(*) FROM {config.sqlite_data_table_name}").fetchone()[0]

    logger.info(
        "Insertando %d elementos en %s::%s",
        count,
        db_path,
        config.sqlite_data_table_name,
    )


def get_augmented_data(
    df: pd.DataFrame,
    news_items: int,
) -> pd.DataFrame:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = GaussianMultivariate()
        model.fit(df)
        synthetic = model.sample(news_items)

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        synthetic[col] = synthetic[col].clip(lower=df[col].min(), upper=df[col].max())

    logger.info("Data Augmentation: %d filas sintéticas generadas.", news_items)
    return synthetic


def main() -> None:

    parser = argparse.ArgumentParser(description="Migra CSV a SQLite.")
    parser.add_argument(
        "--new_elements",
        type=int,
        default=0,
        help="Generar elementos sintéticos",
    )
    args = parser.parse_args()

    logger.info("Iniciando migración de DATA a SQLITE.")

    csv_path = Path(config.csv_path)
    sqlite_path = Path(config.sqlite_path)

    if check_csv_file(csv_path):
        df = load_csv(csv_path)
        df = drop_nulls(df)
        df.columns = df.columns.str.lower()

        if args.new_elements > 0:
            synthetic_df = get_augmented_data(
                df=df,
                news_items=args.new_elements,
            )
            df = pd.concat([df, synthetic_df], ignore_index=True)

            logger.info("Se han generado %d filas", len(df))

        migrate(df, sqlite_path)

    logger.info("Migración completa. SQLite en: %s", sqlite_path)


if __name__ == "__main__":
    main()
