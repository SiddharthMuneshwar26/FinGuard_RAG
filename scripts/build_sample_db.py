"""Build the internal SQLite database used by the agent's read-only SQL tool.

  python scripts/build_sample_db.py                     # synthetic sample table
  python scripts/build_sample_db.py --csv path\\to\\freight.csv --table freight_rates

The sample rows are SYNTHETIC (generated with a fixed seed) and exist only to
demonstrate the tool. Point --csv at the real freight dataset to load that instead.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "finguard_internal.db"

LANES = [
    ("Shanghai", "Rotterdam", "ocean"),
    ("Shanghai", "Los Angeles", "ocean"),
    ("Mumbai", "Felixstowe", "ocean"),
    ("Singapore", "Hamburg", "ocean"),
    ("Frankfurt", "Chicago", "air"),
    ("Hong Kong", "London", "air"),
    ("Delhi", "Dubai", "air"),
    ("Rotterdam", "Milan", "road"),
]
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def build_sample(connection: sqlite3.Connection) -> int:
    rng = random.Random(26)
    connection.execute("DROP TABLE IF EXISTS freight_rates")
    connection.execute(
        """CREATE TABLE freight_rates (
               id INTEGER PRIMARY KEY,
               origin TEXT NOT NULL,
               destination TEXT NOT NULL,
               mode TEXT NOT NULL,
               month TEXT NOT NULL,
               rate_usd REAL NOT NULL,
               transit_days INTEGER NOT NULL,
               is_synthetic INTEGER NOT NULL DEFAULT 1
           )"""
    )
    base = {"ocean": 1800.0, "air": 4.2, "road": 1.6}
    rows = []
    for origin, destination, mode in LANES:
        for index, month in enumerate(MONTHS):
            seasonal = 1 + 0.12 * (1 if index in (8, 9, 10) else 0)
            rate = base[mode] * seasonal * rng.uniform(0.85, 1.2)
            transit = {"ocean": rng.randint(24, 38), "air": rng.randint(1, 3), "road": rng.randint(2, 4)}[mode]
            rows.append((origin, destination, mode, month, round(rate, 2), transit))
    connection.executemany(
        "INSERT INTO freight_rates (origin, destination, mode, month, rate_usd, transit_days) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def load_csv(connection: sqlite3.Connection, csv_path: Path, table: str) -> int:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise SystemExit("Table name must be a plain identifier")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = [re.sub(r"\W+", "_", name.strip()).strip("_").lower() or f"col{i}" for i, name in enumerate(header)]
        connection.execute(f'DROP TABLE IF EXISTS "{table}"')
        column_sql = ", ".join(f'"{column}"' for column in columns)
        connection.execute(f'CREATE TABLE "{table}" ({column_sql})')
        placeholders = ", ".join("?" for _ in columns)
        count = 0
        for row in reader:
            connection.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', row[: len(columns)] + [None] * (len(columns) - len(row)))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--csv", type=Path, help="Load this CSV instead of the synthetic sample")
    parser.add_argument("--table", default="freight_rates")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as connection:
        count = load_csv(connection, args.csv, args.table) if args.csv else build_sample(connection)
    print(f"Wrote {count} rows to {args.db}")


if __name__ == "__main__":
    main()
