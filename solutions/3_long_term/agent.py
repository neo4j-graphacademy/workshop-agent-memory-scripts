# Solution - module 3: the memory tools and save_fact added.

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

# Silence the driver's deprecation notices for the vector-index queries.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

MODEL = "openai-chat:gpt-5.2"
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
SESSION_ID = "learner"
USER_ID = "learner"

# --- The agent's knowledge-graph tools (identical to agent_no_memory.py) ------

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


def get_schema() -> list:
    """Get the schema of the graph database - its node labels and relationship
    types. Use this first if you are unsure how the graph is structured."""
    return run_query("CALL db.schema.visualization()")


def search_lesson_content(query: str) -> list:
    """GraphRAG search over lesson content: semantically match passages and
    return each one together with the graph entities connected to it. Use for
    open-ended 'what does the material say about X' questions."""
    result = vector_retriever.search(query_text=query, top_k=5)
    return [item.content for item in result.items]


def query_database(query: str) -> list:
    """Answer a question by converting it to a Cypher query and running it. Use
    for specific, structured questions like 'how many lessons are there'."""
    result = text2cypher_retriever.search(query_text=query)
    return [item.content for item in result.items]


# --- Memory ------------------------------------------------------------------

memory_settings = MemorySettings(
    neo4j=Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    ),
    embedding=EmbeddingConfig(api_key=os.environ["OPENAI_API_KEY"]),
    extraction=ExtractionConfig(extractor_type=ExtractorType.LLM),
)


# The agent's dependencies: everything a tool or prompt needs at run time.
@dataclass
class AgentDeps:
    memory_client: MemoryClient
    user_id: str
    session_id: str
    current_query: str | None = None


SYSTEM_PROMPT = (
    "You are an assistant for a Neo4j knowledge graph of course material. "
    "Answer questions about the material with get_schema, search_lesson_content, and query_database, and draw on what you remember to help. "
    "When you conclude something durable about the learner or their work, record it with save_fact as a subject, predicate, and object. "
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




@agent.tool
async def search_messages(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search past conversation messages for relevant context."""
    messages = await ctx.deps.memory_client.short_term.search_messages(
        query, session_id=ctx.deps.session_id, limit=5,
    )
    return "\n".join(f"[{m.role.value}] {m.content[:200]}" for m in messages) or "No matching messages."


@agent.tool
async def search_entities(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the entities the agent knows about - the people, places, and
    things it has learned - by meaning."""
    entities = await ctx.deps.memory_client.long_term.search_entities(query, limit=10)
    return "\n".join(f"{e.name} ({e.type})" for e in entities) or "No matching entities."


@agent.tool
async def save_preference(ctx: RunContext[AgentDeps], category: str, preference: str) -> str:
    """Save how the user likes to work, filed under a category."""
    await ctx.deps.memory_client.long_term.add_preference(
        category=category, preference=preference, user_identifier=ctx.deps.user_id,
    )
    return f"Saved: {category} - {preference}"


@agent.tool
async def recall_preferences(ctx: RunContext[AgentDeps], topic: str) -> str:
    """Read back the user's preferences related to a topic."""
    prefs = await ctx.deps.memory_client.long_term.search_preferences(topic, limit=10)
    return "\n".join(f"[{p.category}] {p.preference}" for p in prefs) or "No preferences on that yet."


@agent.tool
async def save_fact(ctx: RunContext[AgentDeps], subject: str, predicate: str, obj: str) -> str:
    """Record a durable fact about the learner or their work, as a
    subject, predicate, object triple."""
    await ctx.deps.memory_client.long_term.add_fact(
        subject=subject, predicate=predicate, obj=obj,
        metadata={"user_id": ctx.deps.user_id, "session_id": ctx.deps.session_id},
    )
    return f"Recorded: {subject} {predicate} {obj}."


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

                # Fresh dependencies each turn, carrying the current query
                # for the dynamic system prompt.
                deps = AgentDeps(
                    memory_client=memory,
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    current_query=user_input,
                )

                result = await agent.run(user_input, deps=deps)
                print(f"\nagent> {result.output}\n")

                await memory.short_term.add_message(
                    SESSION_ID, "assistant", str(result.output), user_identifier=USER_ID,
                )
        finally:
            driver.close()

        print("\nGoodbye. Everything you told me is saved - run me again and I will remember.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass