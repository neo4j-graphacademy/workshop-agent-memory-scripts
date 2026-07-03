# The starting agent - the GenAI workshop's three-tool GraphRAG agent, rebuilt
# with PydanticAI. No persistent memory: quit it, and it forgets everything.

import asyncio
import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic_ai import Agent

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever, VectorCypherRetriever

load_dotenv()

# Silence the driver's deprecation notices for the vector-index queries.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

MODEL = "openai-chat:gpt-5.2"
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

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


# GraphRAG retrieval (the genai workshop's canonical query): vector-match a
# passage, return its lesson, and traverse to the entities connected to it.
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

# Natural language -> Cypher, with an example to steer query generation.
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


agent = Agent(
    MODEL,
    tools=[get_schema, search_lesson_content, query_database],
    system_prompt=(
        "You are an assistant for a Neo4j knowledge graph of course material. "
        "Use get_schema to learn the graph's shape, search_lesson_content for "
        "semantic search over the lesson text, and query_database for specific "
        "structured questions. Check the schema first if you are unsure."
    ),
)


async def main():
    history = []
    print("Agent ready. No memory yet - ask about the course material.")
    print("Type 'exit' (or Ctrl-D) to quit.\n")
    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in {"exit", "quit"}:
                break
            result = await agent.run(user_input, message_history=history)
            print(f"\nagent> {result.output}\n")
            history = result.all_messages()  # within this run only - nothing is saved
    finally:
        driver.close()
    print("\nGoodbye.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass