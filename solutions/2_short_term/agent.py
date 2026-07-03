# Solution - module 2: short-term memory added.
# The agent with its deps architecture in place: a dynamic system prompt
# reads get_context on every turn, and both sides of each turn are stored
# with add_message, carrying the user's identity.

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


SYSTEM_PROMPT = (
    "You are an assistant for a Neo4j knowledge graph of course material. "
    "Answer questions about the material with get_schema, search_lesson_content, and query_database, and draw on what you remember to help."
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




async def main():
    async with MemoryClient(memory_settings) as memory:
        print("Agent ready - now with memory. Ask about the course, or tell me about yourself.")
        print("Type 'exit' (or Ctrl-D) to quit.\n")

        try:
            while True:
                try:
                    user_input = input("you> ").strip()
                except EOFError:
                    break
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break

                # Fresh dependencies each turn, carrying the current query
                # for the dynamic system prompt.
                deps = AgentDeps(
                    memory_client=memory,
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    current_query=user_input,
                )

                # Store the learner's message first, so even a failed turn
                # leaves a complete record.
                await memory.short_term.add_message(
                    SESSION_ID, "user", user_input, user_identifier=USER_ID,
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
    asyncio.run(main())