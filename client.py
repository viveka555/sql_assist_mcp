
"""
Reusable MCP client for the AI SQL Assistant.

Connects to the local MCP server over stdio, invokes a tool,
and converts its text response into Python data when possible.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """
    Call a tool exposed by the local MCP server.

    Args:
        tool_name: Name of the MCP tool to invoke.
        arguments: Arguments required by that tool.

    Returns:
        Parsed JSON data when the tool returns JSON text;
        otherwise, returns the text response.

    Raises:
        RuntimeError: If the MCP tool reports an error.
    """

    # Locate server.py beside this client.py file.
    server_path = Path(__file__).with_name("server.py")

    # Start the MCP server using this project's Python interpreter.
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    # Establish an MCP stdio connection and initialize the session.
    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Invoke the requested server tool.
            result = await session.call_tool(
                tool_name,
                arguments=arguments,
            )

            # Collect text blocks returned by the MCP tool.
            text_blocks = [
                content.text
                for content in result.content
                if hasattr(content, "text")
            ]

            # Surface tool errors to the calling API.
            if result.isError:
                error_message = "\n".join(text_blocks) or "MCP tool failed."
                raise RuntimeError(error_message)

            response_text = "\n".join(text_blocks)

            # Tool responses are JSON text in our current MCP server.
            # Parse it so FastAPI can return structured JSON.
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return response_text


async def main() -> None:
    """Run a small MCP client smoke test."""

    print("Inspecting database schema...")
    schema = await call_mcp_tool(
        "inspect_database_schema",
        arguments={},
    )
    print(json.dumps(schema, indent=2))

    print("\nRunning a read-only test query...")
    query_result = await call_mcp_tool(
        "run_readonly_sql",
        arguments={"query": "SELECT * FROM customers LIMIT 5"},
    )
    print(json.dumps(query_result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
