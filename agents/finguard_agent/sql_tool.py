"""Read-only, parameterised SQL over an internal SQLite database.

Defence in depth, any one of which blocks writes:
  1. validate_select(): a single SELECT (or WITH ... SELECT) statement only, no
     write/DDL/PRAGMA/ATTACH keywords, no stacked statements.
  2. The database file is opened with SQLite's `mode=ro` URI flag.
  3. `PRAGMA query_only = ON` plus an authorizer that denies anything that
     isn't a read.
Values are always bound through `?` placeholders, never string formatting.

Build the database with `python scripts/build_sample_db.py` (synthetic sample)
or `python scripts/build_sample_db.py --csv <your freight csv>`.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext

from .turn_state import register_citations

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "finguard_internal.db"
MAX_ROWS = 50

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|replace|merge|upsert|drop|alter|create|truncate|attach|detach|"
    r"pragma|vacuum|reindex|analyze|grant|revoke|begin|commit|rollback|savepoint|release|load_extension)\b",
    re.IGNORECASE,
)
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def db_path() -> Path:
    return Path(os.getenv("FINGUARD_SQL_DB", str(DEFAULT_DB_PATH)))


def _strip_comments_and_strings(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_select(sql: str) -> Optional[str]:
    """Return an error message if `sql` is not a single read-only SELECT, else None."""
    body = _strip_comments_and_strings(sql).strip().rstrip(";").strip()
    if not body:
        return "The SQL query is empty."
    if ";" in body:
        return "Only a single SQL statement is allowed."
    first = body.split(None, 1)[0].lower()
    if first not in {"select", "with"}:
        return "Only SELECT queries are allowed."
    if first == "with" and not re.search(r"\bselect\b", body, re.IGNORECASE):
        return "Only SELECT queries are allowed."
    keyword = _FORBIDDEN.search(body)
    if keyword:
        return f"Keyword '{keyword.group(1).upper()}' is not allowed; the database is read-only."
    return None


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(
            f"Internal database not found at {path}. Run: python scripts/build_sample_db.py"
        )
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(_authorizer)
    return connection


def describe_internal_database(tool_context: ToolContext) -> dict[str, Any]:
    """List the tables and columns of the internal read-only database.

    Call this before writing a query so you use real table and column names.

    Returns:
        A dict with "status" and "tables": {table_name: [column, ...]}.
    """
    try:
        connection = _connect_read_only(db_path())
    except FileNotFoundError as exc:
        return {"status": "error", "error_message": str(exc)}
    try:
        connection.set_authorizer(None)  # schema introspection reads sqlite_master
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        schema = {
            table: [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            for table in tables
        }
    finally:
        connection.close()
    return {"status": "success", "tables": schema}


def query_internal_database(sql: str, tool_context: ToolContext, parameters: Optional[list[str]] = None) -> dict[str, Any]:
    """Run one read-only, parameterised SELECT query on the internal SQLite database.

    Put every user-supplied value in `parameters` and reference it with a ? placeholder,
    for example: sql="SELECT lane, AVG(rate_usd) FROM freight_rates WHERE mode = ? GROUP BY lane",
    parameters=["ocean"]. Only SELECT is allowed; at most 50 rows are returned.

    Args:
        sql: A single SELECT statement using ? placeholders.
        parameters: Values bound to the ? placeholders, in order.

    Returns:
        A dict with "status", "query_id" (cite it as [sql:<query_id>]), "columns" and "rows".
    """
    problem = validate_select(sql)
    if problem:
        return {"status": "error", "error_message": problem}
    params = list(parameters or [])
    try:
        connection = _connect_read_only(db_path())
    except FileNotFoundError as exc:
        return {"status": "error", "error_message": str(exc)}
    try:
        cursor = connection.execute(sql, params)
        columns = [column[0] for column in cursor.description or []]
        rows = [list(row) for row in cursor.fetchmany(MAX_ROWS + 1)]
    except sqlite3.Error as exc:
        return {"status": "error", "error_message": f"SQL error: {exc}"}
    finally:
        connection.close()

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    query_id = hashlib.sha1(f"{sql}|{params}".encode()).hexdigest()[:8]
    register_citations(tool_context, [f"sql:{query_id}"])
    return {
        "status": "success",
        "query_id": query_id,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
