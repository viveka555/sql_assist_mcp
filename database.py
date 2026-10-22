
"""
PostgreSQL database access for the AI SQL Assistant.

Provides schema inspection and safe, read-only SQL execution.
Database credentials are loaded from application settings.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import sqlglot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

from config import get_settings


# -------------------------------------------------------------------
# Database engine
# -------------------------------------------------------------------

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """
    Create the asynchronous database engine once and reuse it.

    The engine manages a connection pool instead of opening a new
    PostgreSQL connection for every request.
    """

    global _engine

    if _engine is None:
        settings = get_settings()

        database_url = settings.database_url.get_secret_value()

        _engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )

    return _engine


async def close_database() -> None:
    """Dispose of the database connection pool during shutdown."""

    global _engine

    if _engine is not None:
        await _engine.dispose()
        _engine = None


# -------------------------------------------------------------------
# SQL validation
# -------------------------------------------------------------------

def validate_readonly_sql(query: str) -> str:
    """
    Validate that the input contains exactly one read-only query.

    Only SELECT queries, including SELECT queries with CTEs, are
    accepted. PostgreSQL permissions provide an additional security
    boundary; this validation is not the only security control.
    """

    if not query or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    # Parse as PostgreSQL SQL.
    try:
        statements = sqlglot.parse(
            query,
            read="postgres",
        )
    except sqlglot.errors.ParseError as exc:
        raise ValueError("Invalid SQL syntax.") from exc

    # Reject multiple statements.
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only one SQL statement is allowed.")

    statement = statements[0]

    # Allow SELECT and SELECT-based CTEs.
    if not isinstance(statement, (sqlglot.exp.Select, sqlglot.exp.Union)):
        raise ValueError(
            "Only read-only SELECT queries are allowed."
        )

    # SELECT INTO creates a table and is not permitted.
    if statement.find(sqlglot.exp.Into) is not None:
        raise ValueError("SELECT INTO is not allowed.")

    return query.strip().rstrip(";")


# -------------------------------------------------------------------
# Schema inspection
# -------------------------------------------------------------------

async def get_database_schema() -> dict[str, list[dict[str, str]]]:
    """
    Return table and column information from the public schema.

    The schema will later be supplied to the LLM to help it generate
    queries against the correct tables and columns.
    """

    engine = get_engine()

    schema_query = text(
        """
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """
    )

    async with engine.connect() as connection:
        result = await connection.execute(schema_query)

        schema: dict[str, list[dict[str, str]]] = {}

        for row in result.mappings():
            table_name = row["table_name"]

            schema.setdefault(table_name, []).append(
                {
                    "column": row["column_name"],
                    "data_type": row["data_type"],
                }
            )

    return schema


# -------------------------------------------------------------------
# Safe SQL execution
# -------------------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """
    Convert common PostgreSQL values into JSON-friendly values.

    Decimal is converted to a string to preserve numeric precision.
    """

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


async def execute_readonly_query(query: str) -> dict[str, Any]:
    """
    Validate and execute one read-only SQL query.

    Returns:
        A dictionary containing the executed SQL, column names,
        result rows, and the number of rows returned.
    """

    settings = get_settings()

    # Reject unsafe query structures before sending SQL to PostgreSQL.
    validated_query = validate_readonly_sql(query)

    engine = get_engine()

    # Use a transaction so PostgreSQL can enforce read-only mode and
    # apply a transaction-local statement timeout.
    async with engine.connect() as connection:
        async with connection.begin():
            await connection.execute(
                text("SET TRANSACTION READ ONLY")
            )

            # PostgreSQL expects the timeout value in milliseconds.
            timeout_ms = settings.sql_timeout_seconds * 1000

            await connection.execute(
                text(
                    """
                    SELECT set_config(
                        'statement_timeout',
                        :timeout_value,
                        true
                    )
                    """
                ),
                {"timeout_value": str(timeout_ms)},
            )

            # Wrap the query to enforce a server-side row limit.
            limited_query = text(
                """
                SELECT *
                FROM (
                    {query}
                ) AS assistant_result
                LIMIT :row_limit
                """.format(query=validated_query)
            )

            result = await connection.execute(
                limited_query,
                {"row_limit": settings.sql_max_rows},
            )

            columns = list(result.keys())

            rows = [
                {
                    column: _serialize_value(value)
                    for column, value in row.items()
                }
                for row in result.mappings()
            ]

    return {
        "sql": validated_query,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }
