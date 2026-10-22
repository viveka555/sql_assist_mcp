
"""
FastAPI application for the AI SQL Assistant.

Provides authenticated endpoints for schema inspection, direct SQL queries,
and natural-language questions translated into SQL by Groq.
"""

import logging
import secrets
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from client import call_mcp_tool
from config import get_settings
from sql_generator import generate_sql


logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI SQL Assistant API",
    description=(
        "Authenticated API for inspecting a database, running read-only SQL, "
        "and asking questions in natural language."
    ),
    version="0.2.0",
)

bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """Validate the bearer token configured in the project's .env file."""

    settings = get_settings()

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Pydantic SecretStr must be unwrapped before comparing strings.
    expected_token = settings.api_token.get_secret_value()

    if not secrets.compare_digest(
        credentials.credentials,
        expected_token,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class SQLQueryRequest(BaseModel):
    """Request body for directly executing a SQL query."""

    query: str = Field(
        min_length=1,
        max_length=10_000,
        description="A single read-only SELECT statement.",
    )


class AskRequest(BaseModel):
    """Request body for a natural-language database question."""

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="A question about the data in the connected database.",
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return API health without accessing the database."""

    return {"status": "ok"}


@app.get(
    "/schema",
    dependencies=[Depends(verify_api_token)],
)
async def inspect_schema() -> Any:
    """Return the database schema using the MCP schema tool."""

    try:
        return await call_mcp_tool(
            "inspect_database_schema",
            arguments={},
        )
    except Exception as exc:
        logger.exception("MCP schema request failed")
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve database schema through MCP.",
        ) from exc


@app.post(
    "/query",
    dependencies=[Depends(verify_api_token)],
)
async def execute_query(request: SQLQueryRequest) -> Any:
    """Execute a direct SQL query through the read-only MCP tool."""

    try:
        return await call_mcp_tool(
            "run_readonly_sql",
            arguments={"query": request.query},
        )
    except RuntimeError as exc:
        logger.exception("MCP rejected or failed the SQL query")
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("MCP SQL execution failed")
        raise HTTPException(
            status_code=502,
            detail="Could not execute the query through MCP.",
        ) from exc


@app.post(
    "/ask",
    dependencies=[Depends(verify_api_token)],
)
async def ask_database(request: AskRequest) -> dict[str, Any]:
    """
    Translate a natural-language question into SQL and execute it safely.

    The LLM only generates SQL. The MCP/database layer remains responsible
    for validating and executing the query under its read-only safeguards.
    """

    try:
        # Step 1: Fetch the actual schema so the LLM can use real table
        # and column names instead of guessing.
        schema = await call_mcp_tool(
            "inspect_database_schema",
            arguments={},
        )

        # Step 2: Ask Groq to generate a SQL query grounded in that schema.
        generated_sql = await generate_sql(
            question=request.question,
            schema=schema,
        )

        # Step 3: Submit the generated SQL to the existing guarded tool.
        # The MCP/database layer validates it, applies limits and timeout,
        # and executes it in a read-only transaction.
        query_result = await call_mcp_tool(
            "run_readonly_sql",
            arguments={"query": generated_sql},
        )

        # Return both the generated SQL and the data it retrieved.
        return {
            "question": request.question,
            "generated_sql": generated_sql,
            "result": query_result,
        }

    except RuntimeError as exc:
        # For example, the SQL validator may reject unsafe generated SQL.
        logger.exception("Generated SQL was rejected or execution failed")
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        # For example, SQL generation returned an empty response.
        logger.exception("Could not generate SQL for the question")
        raise HTTPException(
            status_code=502,
            detail="The SQL generator could not produce a usable query.",
        ) from exc
    except Exception as exc:
        logger.exception("Natural-language database request failed")
        raise HTTPException(
            status_code=502,
            detail="Could not process the question through the SQL assistant.",
        ) from exc
