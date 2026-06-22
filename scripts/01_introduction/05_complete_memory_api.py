"""Module 1, Lesson 5 — The complete memory API.

A single end-to-end walkthrough of every public method `neo4j-agent-memory`
exposes, in the order you'd call them in a real application: session management,
short-term memory, long-term memory, combined context retrieval, and reasoning
traces.

Authored from the course lesson's documented API. The package is pre-1.0, so
run this against a live Neo4j and reconcile any signature drift before relying
on it. See the module/lesson MDX for the narrative this script backs.

Run:  python scripts/01_introduction/05_complete_memory_api.py
"""

import asyncio
import sys
from pathlib import Path

# Make the repo-root config.py importable when run as a file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import build_settings  # noqa: E402

from neo4j_agent_memory import MemoryClient  # noqa: E402

SESSION_ID = "user_123"


async def main() -> None:
    settings = build_settings()

    async with MemoryClient(settings) as memory:
        # ── Short-term memory ────────────────────────────────────────────
        # The session/Conversation node is created automatically on first use.
        await memory.short_term.add_message(
            session_id=SESSION_ID,
            role="user",
            content="Review Jessica Norris account for a credit limit increase",
        )
        await memory.short_term.add_message(
            session_id=SESSION_ID,
            role="assistant",
            content="I will retrieve Jessica's full profile now.",
        )

        conversation = await memory.short_term.get_conversation(SESSION_ID)
        print("Conversation:", conversation)

        results = await memory.short_term.search_messages(
            query="credit limit increase",
            session_id=SESSION_ID,
            limit=10,
        )
        print("Message search:", results)

        summary = await memory.short_term.get_conversation_summary(SESSION_ID)
        print("Summary:", summary)

        # ── Long-term memory (POLE+O entity graph) ───────────────────────
        await memory.long_term.add_entity(
            name="Jessica Norris",
            entity_type="PERSON",
            subtype="CUSTOMER",
            description="High-value customer, flagged for compliance review April 2025",
            properties={"risk_score": 0.415},
        )
        await memory.long_term.add_fact(
            subject="Jessica Norris",
            predicate="manages",
            object="Acme Corp account",
            valid_from="2024-01-01",
            valid_until="2025-03-31",
        )
        await memory.long_term.add_preference(
            category="communication",
            preference="Prefers concise responses",
            context="Confirmed during onboarding",
        )

        entities = await memory.long_term.search_entities(
            query="Jessica Norris accounts", limit=10
        )
        print("Entity search:", entities)

        prefs = await memory.long_term.search_preferences(query="communication")
        print("Preferences:", prefs)

        # entity_id is the slugified entity name (see the lesson); confirm
        # against the value returned by add_entity / search_entities.
        subgraph = await memory.long_term.get_entity_graph(
            entity_id="jessica-norris", depth=2
        )
        print("Entity subgraph:", subgraph)

        # ── Combined context (all three layers, formatted for a prompt) ───
        context = await memory.get_context(
            query="What do I know about Jessica Norris?",
            session_id=SESSION_ID,
            include_short_term=True,
            include_long_term=True,
            include_reasoning=True,
            max_items=10,
        )
        print("Combined context:\n", context)

        # ── Reasoning memory (start → step → tool call → complete) ───────
        trace = await memory.reasoning.start_trace(
            task="Evaluate credit limit for Jessica Norris",
            session_id=SESSION_ID,
        )
        step = await memory.reasoning.add_step(
            trace_id=trace.id,
            thought="Retrieving customer entity from long-term memory",
            action="search_entities",
        )
        await memory.reasoning.record_tool_call(
            step_id=step.id,
            tool_name="search_entities",
            arguments={"query": "Jessica Norris", "limit": 5},
            result={"entities": ["Jessica Norris (EntityPerson)"]},
            status="success",
        )
        await memory.reasoning.complete_trace(
            trace.id,
            outcome="Approved — risk score within threshold",
            success=True,
        )

        # ── Querying and analysing traces ────────────────────────────────
        similar = await memory.reasoning.get_similar_traces(
            task="What do you know about me?", limit=3
        )
        print("Similar traces:", similar)

        traces = await memory.reasoning.list_traces()
        print("All traces:", traces)

        stats = await memory.reasoning.get_tool_stats()
        for tool_name, count in stats.items():
            print(f"{tool_name}: {count} calls")

        provenance = await memory.reasoning.get_trace_provenance(trace.id)
        print("Trace provenance:", provenance)


if __name__ == "__main__":
    asyncio.run(main())
