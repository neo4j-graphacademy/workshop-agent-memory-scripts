# The finished memory agent: agent_no_memory.py plus neo4j-agent-memory.
# Its memory lives in the shared workshop instance (MVP_NEO4J_*), and its
# session and user ids come from the environment.

import asyncio
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic_ai import Agent, RunContext

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever, VectorCypherRetriever

from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.config import (
    Neo4jConfig,
    EmbeddingConfig,
    ExtractionConfig,
    ExtractorType,
)

load_dotenv()

if not os.getenv("MVP_NEO4J_URI"):
    raise SystemExit(
        "MVP_NEO4J_URI is not set - the MVP's memory lives in the shared "
        "workshop instance. Add the MVP_NEO4J_* values from module 1 to your .env."
    )

# Silence the driver's deprecation notices for the vector-index queries.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

MODEL = "openai-chat:gpt-5.2"
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
SESSION_ID = os.getenv("MVP_SESSION_ID", "learner")
USER_ID = os.getenv("MVP_USER_ID", SESSION_ID)

# --- The agent's knowledge-graph tools (the genai workshop's, each
#     reporting into the trace) ---------------------------------------------

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm = OpenAILLM(model_name="gpt-5.2")


def run_query(cypher, **params):
    """Run a Cypher query and return the rows as plain dicts."""
    records, _, _ = driver.execute_query(cypher, parameters_=params, database_=DATABASE)
    return [r.data() for r in records]


GRAPHRAG_RETRIEVAL = """
MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
RETURN
    node.text as text, score,
    lesson.url as lesson_url,
    collect {
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            [l IN labels(entity)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            entity.name,
            type(r),
            [l IN labels(other)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            other.name
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""
vector_retriever = VectorCypherRetriever(
    driver,
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=GRAPHRAG_RETRIEVAL,
    neo4j_database=DATABASE,
)

examples = [
    "USER INPUT: 'Find a node with the name $name?' QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower($name) RETURN node.name AS name, labels(node) AS labels",
]
text2cypher_retriever = Text2CypherRetriever(
    driver=driver, neo4j_database=DATABASE, llm=llm, examples=examples,
)


async def get_schema(ctx: RunContext[AgentDeps]) -> list:
    """Get the schema of the graph database - its node labels and relationship
    types. Use this first if you are unsure how the graph is structured."""
    await report_step(ctx, "get_schema")
    return run_query("CALL db.schema.visualization()")


async def search_lesson_content(ctx: RunContext[AgentDeps], query: str) -> list:
    """GraphRAG search over lesson content: semantically match passages and
    return each one together with the graph entities connected to it. Use for
    open-ended 'what does the material say about X' questions."""
    await report_step(ctx, "search_lesson_content", query=query)
    result = vector_retriever.search(query_text=query, top_k=5)
    return [item.content for item in result.items]


async def query_database(ctx: RunContext[AgentDeps], query: str) -> list:
    """Answer a question by converting it to a Cypher query and running it. Use
    for specific, structured questions like 'how many lessons are there'."""
    await report_step(ctx, "query_database", query=query)
    result = text2cypher_retriever.search(query_text=query)
    return [item.content for item in result.items]


# --- Memory ------------------------------------------------------------------
# The shared workshop instance: everyone's memory in one graph.

memory_settings = MemorySettings(
    neo4j=Neo4jConfig(
        uri=os.environ["MVP_NEO4J_URI"],
        username=os.environ["MVP_NEO4J_USERNAME"],
        password=os.environ["MVP_NEO4J_PASSWORD"],
    ),
    embedding=EmbeddingConfig(api_key=os.environ["OPENAI_API_KEY"]),
    extraction=ExtractionConfig(
        extractor_type=ExtractorType.LLM,
        entity_types=[
            "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "OBJECT",
            "ACTIVITY",
        ],
    ),
)


# The agent's dependencies: everything a tool or prompt needs at run time.
@dataclass
class AgentDeps:
    memory_client: MemoryClient
    user_id: str
    session_id: str
    current_query: str | None = None
    current_trace_id: str | None = None


SYSTEM_PROMPT = (
    "You are an assistant for a Neo4j knowledge graph of course material. "
    "Answer questions about the material with get_schema, search_lesson_content, and query_database, and draw on what you remember to help. "
    "When the learner asks who to connect with, use find_similar_attendees. "
    "When they are just introducing themselves or chatting, only acknowledge what they said - do not go looking for people. "
    "When you conclude something durable about the learner or their work, record it with save_fact as a subject, predicate, and object. "
    "When the learner states a connection between two things - wrote, works at, uses - record it with record_connection rather than save_fact. "
    "Save anything worth keeping."
)

# Built once. The static prompt sets policy; memory arrives through deps.
agent = Agent(
    MODEL,
    deps_type=AgentDeps,
    tools=[get_schema, search_lesson_content, query_database],
    system_prompt=SYSTEM_PROMPT,
)


@agent.system_prompt
async def what_you_remember(ctx: RunContext[AgentDeps]) -> str:
    """Read what the agent remembers about this learner into every turn."""
    if ctx.deps.current_query is None:
        return ""
    context = await ctx.deps.memory_client.get_context(
        ctx.deps.current_query, session_id=ctx.deps.session_id,
    )
    return f"What you remember:\n{context}"


async def report_step(ctx: RunContext[AgentDeps], tool_name: str, **arguments) -> None:
    """Report a tool's work into the turn's open trace."""
    if ctx.deps.current_trace_id is None:
        return
    step = await ctx.deps.memory_client.reasoning.add_step(
        trace_id=ctx.deps.current_trace_id,
        thought=f"Calling {tool_name}", action=tool_name,
    )
    await ctx.deps.memory_client.reasoning.record_tool_call(
        step_id=step.id, tool_name=tool_name, arguments=arguments,
    )


@agent.tool
async def search_messages(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search past conversation messages for relevant context."""
    await report_step(ctx, "search_messages", query=query)
    messages = await ctx.deps.memory_client.short_term.search_messages(
        query, session_id=ctx.deps.session_id, limit=5,
    )
    return "\n".join(f"[{m.role.value}] {m.content[:200]}" for m in messages) or "No matching messages."


@agent.tool
async def search_entities(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the entities the agent knows about - the people, places, and
    things it has learned - by meaning."""
    await report_step(ctx, "search_entities", query=query)
    entities = await ctx.deps.memory_client.long_term.search_entities(query, limit=10)
    return "\n".join(f"{e.name} ({e.type})" for e in entities) or "No matching entities."


@agent.tool
async def save_preference(ctx: RunContext[AgentDeps], category: str, preference: str) -> str:
    """Save how the user likes to work, filed under a category."""
    await report_step(ctx, "save_preference", category=category, preference=preference)
    await ctx.deps.memory_client.long_term.add_preference(
        category=category, preference=preference, user_identifier=ctx.deps.user_id,
    )
    return f"Saved: {category} - {preference}"


@agent.tool
async def recall_preferences(ctx: RunContext[AgentDeps], topic: str) -> str:
    """Read back the user's preferences related to a topic."""
    await report_step(ctx, "recall_preferences", topic=topic)
    prefs = await ctx.deps.memory_client.long_term.search_preferences(topic, limit=10)
    return "\n".join(f"[{p.category}] {p.preference}" for p in prefs) or "No preferences on that yet."


@agent.tool
async def find_similar_attendees(ctx: RunContext[AgentDeps], interests: str) -> str:
    """Find other workshop attendees for the learner to connect with, by
    shared interests."""
    await report_step(ctx, "find_similar_attendees", interests=interests)
    # Over-fetch, then keep only messages from other sessions - otherwise the
    # learner's own words outrank everyone else's.
    hits = await ctx.deps.memory_client.short_term.search_messages(interests, limit=25)
    if not hits:
        return "No attendees found yet."
    rows = await ctx.deps.memory_client.query.cypher(
        """
        MATCH (c:Conversation)-[:HAS_MESSAGE]->(m:Message)
        WHERE m.id IN $ids AND c.session_id <> $session_id
        RETURN m.id AS id
        """,
        {"ids": [str(m.id) for m in hits], "session_id": ctx.deps.session_id},
    )
    other_ids = {row["id"] for row in rows}
    others = [m for m in hits if str(m.id) in other_ids][:5]
    return "\n".join(f"[{m.role.value}] {m.content}" for m in others) or "No attendees found yet."


@agent.tool
async def save_fact(ctx: RunContext[AgentDeps], subject: str, predicate: str, obj: str) -> str:
    """Record a durable fact about the learner or their work, as a
    subject, predicate, object triple."""
    await report_step(ctx, "save_fact", subject=subject, predicate=predicate, obj=obj)
    await ctx.deps.memory_client.long_term.add_fact(
        subject=subject, predicate=predicate, obj=obj,
        metadata={"user_id": ctx.deps.user_id, "session_id": ctx.deps.session_id},
    )
    return f"Recorded: {subject} {predicate} {obj}."


@agent.tool
async def how_did_i_handle(ctx: RunContext[AgentDeps], task: str) -> str:
    """Read back how similar past tasks were handled."""
    await report_step(ctx, "how_did_i_handle", task=task)
    traces = await ctx.deps.memory_client.reasoning.get_similar_traces(task=task, limit=3)
    return "\n".join(f"{t.task} -> {t.outcome} (success: {t.success})" for t in traces) or "No similar past tasks yet."


@agent.tool
async def record_connection(ctx: RunContext[AgentDeps], source: str, target: str, connection: str) -> str:
    """Record how two things the conversation has mentioned are connected,
    such as WROTE or WORKS_AT. Reports back if either is not in memory yet."""
    await report_step(ctx, "record_connection", source=source, target=target, connection=connection)
    a = await ctx.deps.memory_client.long_term.get_entity_by_name(source)
    b = await ctx.deps.memory_client.long_term.get_entity_by_name(target)
    if a is None or b is None:
        return "One of those entities is not in memory yet."
    await ctx.deps.memory_client.long_term.add_relationship(a, b, connection)
    return f"Recorded: {source} {connection} {target}."


async def main():
    async with MemoryClient(memory_settings) as memory:
        print("Agent ready - now with memory. Ask about the course, or tell me about yourself.")
        print("Type 'exit' (or Ctrl-D) to quit.\n")

        try:
            while True:
                try:
                    user_input = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break

                # Store the learner's message first, so even a failed turn
                # leaves a complete record.
                await memory.short_term.add_message(
                    SESSION_ID, "user", user_input, user_identifier=USER_ID,
                )

                # Open the trace before the attempt - either ending closes it.
                trace = await memory.reasoning.start_trace(
                    session_id=SESSION_ID, task=user_input[:200], user_identifier=USER_ID,
                )

                # Fresh dependencies each turn - the current query for the
                # dynamic prompt, the open trace for tools to report into.
                deps = AgentDeps(
                    memory_client=memory,
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    current_query=user_input,
                    current_trace_id=str(trace.id),
                )

                try:
                    result = await agent.run(user_input, deps=deps)
                    print(f"\nagent> {result.output}\n")

                    await memory.short_term.add_message(
                        SESSION_ID, "assistant", str(result.output), user_identifier=USER_ID,
                    )
                    await memory.reasoning.complete_trace(trace.id, outcome="success", success=True)
                except Exception as error:
                    await memory.reasoning.complete_trace(trace.id, outcome=str(error), success=False)
                    print("\nagent> Something went wrong - that attempt is on the record.\n")
        finally:
            driver.close()

        print("\nGoodbye. Everything you told me is saved - run me again and I will remember.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass