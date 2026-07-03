# INSTRUCTOR ONLY - reset the shared NAMS workspace between workshop runs.
#
# Clears every conversation in the workspace, previewing the list and asking
# for confirmation first. The hosted backend has no delete API for entities,
# facts, preferences, or traces - clear those from the NAMS dashboard at
# https://memory.neo4jlabs.com.

import asyncio
import os

from dotenv import load_dotenv
from neo4j_agent_memory import MemoryClient

load_dotenv()

if not os.getenv("MEMORY_API_KEY"):
    raise SystemExit("MEMORY_API_KEY is not set - add the workspace key to your .env first.")


async def main():
    async with MemoryClient() as client:
        sessions = await client.short_term.list_sessions()
        if not sessions:
            print("No conversations in the workspace.")
            return

        print(f"{len(sessions)} conversation(s) in the workspace:")
        for s in sessions:
            print(f"  - {s.session_id}")

        answer = input("\nClear ALL of these? Type 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            print("Aborted - nothing cleared.")
            return

        cleared, failed = 0, []
        for s in sessions:
            try:
                await client.short_term.clear_session(s.session_id)
                cleared += 1
            except Exception as error:
                failed.append((s.session_id, str(error)))

        print(f"\nCleared {cleared} conversation(s).")
        for session_id, error in failed:
            print(f"  failed: {session_id} - {error}")
        print("Entities, facts, preferences, and traces: clear via the NAMS dashboard.")


if __name__ == "__main__":
    asyncio.run(main())
