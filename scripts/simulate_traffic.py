import argparse
import logging
import random
import time

import pandas as pd
import requests

from src.core.config import get_config
from src.data.repository import load_data
from src.data.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)
config = get_config()

API_URL = "http://localhost:8000/predict"


def build_payload(row: pd.Series, drift: bool = False) -> dict:
    payload = {col: float(row[col]) for col in FEATURE_COLUMNS if not pd.isna(row[col])}

    if drift:
        if "crim" in payload:
            payload["crim"] *= 5.0
        if "rm" in payload:
            payload["rm"] -= 2.0
        if "lstat" in payload:
            payload["lstat"] += random.uniform(5, 15)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula tráfico hacia la API de predicción")
    parser.add_argument("--requests", type=int, default=100, help="Cantidad de requests a enviar")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay entre requests (segundos)")
    parser.add_argument("--drift", action="store_true", help="Induce drift artificial en los payloads")
    args = parser.parse_args()

    logger.info(
        "Enviando %d requests a %s | delay=%.2fs | drift=%s",
        args.requests, API_URL, args.delay, args.drift,
    )

    df = load_data(str(config.sqlite_path), config.sqlite_data_table_name)

    success = 0
    failed = 0

    for i in range(args.requests):
        row = df.sample(n=1).iloc[0]
        payload = build_payload(row, drift=args.drift)

        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()
            success += 1
            if (i + 1) % 10 == 0:
                logger.info(
                "[%d/%d] predicción=%.2f | modelo=v%s",
                i + 1, args.requests, data["prediction"], data["model_version"],
            )
        except Exception as e:
            failed += 1
            logger.warning("Request %d falló: %s", i + 1, e)

        time.sleep(args.delay)

    logger.info("Finalizado. éxitos=%d | fallos=%d", success, failed)

if __name__ == "__main__":
    main()
