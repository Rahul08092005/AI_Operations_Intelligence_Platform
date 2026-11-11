"""Load anomaly episodes into database anomalies table (Task T12, optional).

Usage:
    python scripts/load_anomalies_to_db.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.anomaly_detection.config import PROCESSED_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("load_anomalies_to_db")


def main() -> int:
    episodes_path = PROCESSED_DIR / "anomaly_episodes.parquet"
    if not episodes_path.exists():
        logger.error("Episodes file %s missing.", episodes_path)
        return 1

    df_episodes = pd.read_parquet(episodes_path)
    logger.info("Found %d anomaly episodes to load.", len(df_episodes))

    # Optional DB load check per Section 3
    try:
        import os

        import psycopg

        db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_ops")
        logger.info("Attempting connection to %s...", db_url)
        with psycopg.connect(db_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM anomalies;")
            for _, ep in df_episodes.iterrows():
                cur.execute(
                    """
                    INSERT INTO anomalies
                    (machine_id, timestamp, anomaly_score, anomaly_type, severity)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (
                        ep["machine_id"],
                        ep["peak_time"],
                        float(ep["peak_score"]),
                        ep["anomaly_type"],
                        ep["severity"],
                    ),
                )
            conn.commit()
            logger.info(
                "Successfully loaded %d episodes into database anomalies table.",
                len(df_episodes),
            )
    except Exception as exc:
        logger.warning("Database unavailable (%s); skipping optional DB load per Section 3.", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
