# The five-minute quickstart (module 1, lesson 2).
# Store all three memory layers in the hosted Neo4j Agent Memory Service
# (NAMS) and read them back. There is no database to run - MEMORY_API_KEY
# is the whole setup.

import asyncio
import os

from dotenv import load_dotenv
from neo4j_agent_memory import MemoryClient

load_dotenv()

# The workspace is shared with the whole workshop, so your session id keeps
# your quickstart data apart from everyone else's.
SESSION_ID = os.environ["MVP_SESSION_ID"] + "-quickstart"


async def main():
    # No settings: with MEMORY_API_KEY set, the client picks the hosted backend.
    async with MemoryClient() as client:
        conversation = await client.short_term.create_conversation(SESSION_ID)
        conversation_id = str(conversation.id)

        # Short-term memory
        await client.short_term.add_message(conversation_id, "user", "Hi, I'm Alice.")
        await client.short_term.add_message(conversation_id, "user", "I love Italian food.")

        # Long-term memory
        entity = await client.long_term.add_entity("Alice", "PERSON", description="The user")
        if isinstance(entity, tuple):
            entity = entity[0]

        # Reasoning memory
        trace = await client.reasoning.start_trace(conversation_id, "Recommend dinner")
        step = await client.reasoning.add_step(trace.id, thought="Alice likes Italian")
        await client.reasoning.record_tool_call(
            step.id,
            tool_name="search",
            arguments={"cuisine": "Italian"},
            result=["Da Mario"],
        )
        await client.reasoning.complete_trace(trace.id, outcome="Suggested Da Mario", success=True)

        # Read everything back
        conv = await client.short_term.get_conversation(conversation_id)
        traces = await client.reasoning.get_session_traces(conversation_id)
        print(f"Messages: {len(conv.messages)}")
        print(f"Entity stored: {entity.name}")
        print(f"Traces: {[t.task for t in traces]}")

        # Read-only Cypher against the hosted graph
        rows = await client.query.cypher(
            "MATCH (e:Entity {name: $name}) RETURN e.name AS name, e.type AS type",
            {"name": "Alice"},
        )
        for row in rows:
            print(row)


if __name__ == "__main__":
    asyncio.run(main())
