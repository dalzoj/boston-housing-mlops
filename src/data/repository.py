import sqlite3

import pandas as pd


def open_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def close_connection(conn: sqlite3.Connection) -> None:
    conn.close()


def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    conn = open_connection(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        close_connection(conn)
