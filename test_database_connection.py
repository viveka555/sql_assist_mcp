
"""
Temporary smoke test for the SQL Assistant database module.
"""

import asyncio

from database import (
    close_database,
    execute_readonly_query,
    get_database_schema,
)


async def main() -> None:
    """Verify schema inspection and safe query execution."""

    try:
        schema = await get_database_schema()

        print("Database schema:")
        print(schema)

        result = await execute_readonly_query(
            """
            SELECT
                customer_name,
                city
            FROM customers
            ORDER BY customer_name
            """
        )

        print("\nQuery result:")
        print(result)

    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
