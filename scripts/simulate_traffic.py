import argparse
import logging
import random
import time

import pandas as pd
import requests

from src.core.config import get_config
from src.core.logging import setup_logging
from src.data.repository import load_data
from src.data.schema import FEATURE_COLUMNS

setup_logging()
logger = logging.getLogger(__name__)
config = get_config()

API_URL = "http://localhost:8000/predict"

PHASES = [
    {"name": "normal", "delay": (0.5, 2.0), "error_prob": 0.05},
    {"name": "burst", "delay": (0.05, 0.2), "error_prob": 0.05},
    {"name": "quiet", "delay": (2.0, 5.0), "error_prob": 0.05},
    {"name": "errors", "delay": (0.5, 2.0), "error_prob": 0.30},
]

PHASE_DURATION_RANGE = (15, 45)
DRIFT_PROB = 0.20


def build_payload(row: pd.Series, drift: bool = False) -> dict:
    payload = {col: float(row[col]) for col in FEATURE_COLUMNS}
    if drift:
        payload["crim"] *= 5.0
        payload["rm"] -= 2.0
        payload["lstat"] += random.uniform(5, 15)
    return payload


def build_invalid_payload(row: pd.Series) -> dict:
    payload = build_payload(row)
    payload.pop(random.choice(FEATURE_COLUMNS))
    return payload


def send_one(df: pd.DataFrame, phase: dict, drift_enabled: bool) -> tuple[str, int | None]:
    row = df.sample(1).iloc[0]
    roll = random.random()

    if roll < phase["error_prob"]:
        payload, kind = build_invalid_payload(row), "ERROR"
    elif drift_enabled and roll < phase["error_prob"] + DRIFT_PROB:
        payload, kind = build_payload(row, drift=True), "DRIFT"
    else:
        payload, kind = build_payload(row), "OK"

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        return kind, response.status_code
    except requests.RequestException:
        return kind, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador continuo de tráfico.")
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Activa drift en requests",
    )
    args = parser.parse_args()
    df = load_data(str(config.sqlite_path), config.sqlite_data_table_name)
    logger.info("Dataset base: %d filas", len(df))

    total = success = client_err = server_err = drift_count = 0

    try:
        while True:
            phase = random.choice(PHASES)
            phase_duration = random.uniform(*PHASE_DURATION_RANGE)
            phase_end = time.time() + phase_duration
            logger.info(
                " Fase '%s' por %.0fs (delay=%s, error=%.0f%%)",
                phase["name"].upper(),
                phase_duration,
                phase["delay"],
                phase["error_prob"] * 100,
            )

            while time.time() < phase_end:
                kind, status = send_one(df, phase, args.drift)
                total += 1
                if kind == "DRIFT":
                    drift_count += 1

                if status == 200:
                    success += 1

                elif 400 <= status < 500:
                    client_err += 1

                else:
                    server_err += 1

                if total % 10 == 0:
                    logger.info(
                        "Total=%d OK=%d 4XX=%d 5XX=%d Drift=%d (Fase=%s)",
                        total,
                        success,
                        client_err,
                        server_err,
                        drift_count,
                        phase["name"],
                    )

                time.sleep(random.uniform(*phase["delay"]))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
