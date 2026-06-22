"""
Starting point — the agent you finished the GenAI workshop with.

This is carried over verbatim from the first workshop (workshop-genai,
solutions/agent_full.py). Learners begin this workshop with this complete,
working agent; we do not rebuild it. Every later script layers
neo4j-agent-memory on top of this baseline.

----

Final challenge — an agent with a custom set of tools over the lesson knowledge graph.

Tools span the three approaches from the workshop:
  - Cypher query tools (run fixed/parameterised Cypher)
  - a Vector + Cypher retriever (semantic search)
  - a Text2Cypher tool (natural-language questions -> Cypher)

Needs the lesson graph (Lesson, Document, Chunk, entities) and the
`chunkEmbedding` vector index, as built by kg_structured_builder.py.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM

model = init_chat_model("gpt-5.2", model_provider="openai")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
DB = os.getenv("NEO4J_DATABASE")

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm = OpenAILLM(model_name="gpt-5.2")


def run_query(cypher, **params):
    """Run a Cypher query and return the rows as plain dicts."""
    records, _, _ = driver.execute_query(cypher, parameters_=params, database_=DB)
    return [r.data() for r in records]


# Semantic search returns the matching chunk text, the lesson it came from, and
# the entities connected to it. OPTIONAL MATCH so chunks still return if the
# Lesson layer is absent.
retrieval_query = """
OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
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
    neo4j_database=DB,
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

examples = [
    "USER INPUT: 'Find a node with the name $name?' QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower($name) RETURN node.name AS name, labels(node) AS labels",
]
text2cypher_retriever = Text2CypherRetriever(
    driver=driver, neo4j_database=DB, llm=llm, examples=examples,
)


# --- Graph overview -------------------------------------------------------

@tool("Get-graph-database-schema")
def get_schema():
    """Get the node labels and relationships in the graph. Use first if unsure
    how the graph is shaped."""
    return run_query("CALL db.schema.visualization()")


@tool("Count-nodes-by-label")
def count_nodes_by_label():
    """Count how many nodes exist for each label. Use for 'how many X are there'
    overview questions."""
    return run_query(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS count ORDER BY count DESC"
    )


@tool("Count-relationships-by-type")
def count_relationships_by_type():
    """Count how many relationships exist of each type. Use to understand how
    densely the graph is connected."""
    return run_query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    )


# --- Course structure -----------------------------------------------------

@tool("List-lessons")
def list_lessons():
    """List every lesson with its module and course. Use for questions about
    course structure or how many lessons there are."""
    return run_query(
        "MATCH (l:Lesson) "
        "RETURN l.course AS course, l.module AS module, l.name AS lesson, l.url AS url "
        "ORDER BY course, module, lesson"
    )


@tool("List-modules")
def list_modules():
    """List the modules and how many lessons each contains."""
    return run_query(
        "MATCH (l:Lesson) "
        "RETURN l.course AS course, l.module AS module, count(l) AS lessons "
        "ORDER BY course, module"
    )


@tool("Lessons-in-module")
def lessons_in_module(module: str):
    """List the lessons in a given module (partial name match)."""
    return run_query(
        "MATCH (l:Lesson) WHERE toLower(l.module) CONTAINS toLower($module) "
        "RETURN l.module AS module, l.name AS lesson, l.url AS url ORDER BY lesson",
        module=module,
    )


# --- Entities -------------------------------------------------------------

@tool("List-entity-types")
def list_entity_types():
    """List the kinds of entity in the graph (e.g. Technology, Concept,
    Benefit). Use before asking for entities of a specific type."""
    return run_query(
        "MATCH (e:__Entity__) UNWIND labels(e) AS label "
        "WITH label WHERE NOT label IN ['__Entity__', '__KGBuilder__'] "
        "RETURN DISTINCT label ORDER BY label"
    )


@tool("Entities-of-type")
def entities_of_type(entity_type: str):
    """List the named entities of a given type, e.g. all Technology or all
    Benefit nodes. Pass a label from List-entity-types."""
    return run_query(
        "MATCH (e:__Entity__) WHERE $type IN labels(e) "
        "RETURN e.name AS name ORDER BY name LIMIT 100",
        type=entity_type,
    )


@tool("Find-related-entities")
def find_related_entities(name: str):
    """Find the entities directly connected to a named entity, and how they
    relate. Use to explore how one thing connects to others."""
    return run_query(
        "MATCH (e:__Entity__) WHERE toLower(e.name) CONTAINS toLower($name) "
        "MATCH (e)-[r]->(other:__Entity__) "
        "RETURN e.name AS entity, type(r) AS relationship, other.name AS related, "
        "[l IN labels(other) WHERE NOT l IN ['__KGBuilder__', '__Entity__']][0] AS related_type "
        "LIMIT 50",
        name=name,
    )


@tool("Most-connected-entities")
def most_connected_entities():
    """List the most connected entities — a quick way to find the key concepts
    in the material."""
    return run_query(
        "MATCH (e:__Entity__) "
        "RETURN e.name AS name, "
        "[l IN labels(e) WHERE NOT l IN ['__KGBuilder__', '__Entity__']][0] AS type, "
        "count{ (e)--(:__Entity__) } AS connections "
        "ORDER BY connections DESC LIMIT 15"
    )


@tool("Path-between-entities")
def path_between_entities(start: str, end: str):
    """Show how two entities are connected by returning the shortest path of
    relationships between them."""
    return run_query(
        "MATCH (a:__Entity__), (b:__Entity__) "
        "WHERE toLower(a.name) CONTAINS toLower($start) "
        "AND toLower(b.name) CONTAINS toLower($end) "
        "MATCH p = shortestPath("
        "(a)-[:RELATED_TO|PART_OF|USED_IN|LEADS_TO|HAS_CHALLENGE|CITES*..6]-(b)) "
        "RETURN [n IN nodes(p) | n.name] AS path, length(p) AS hops LIMIT 1",
        start=start, end=end,
    )


@tool("Lessons-mentioning")
def lessons_mentioning(name: str):
    """Find which lessons mention a given entity. Use to locate where a topic is
    taught."""
    return run_query(
        "MATCH (e:__Entity__)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(:Document)-[:PDF_OF]->(l:Lesson) "
        "WHERE toLower(e.name) CONTAINS toLower($name) "
        "RETURN DISTINCT l.name AS lesson, l.module AS module, l.url AS url ORDER BY lesson",
        name=name,
    )


# --- Content search -------------------------------------------------------

@tool("Search-lesson-content")
def search_lessons(query: str):
    """Semantically search lesson content for passages related to the query.
    Use for open-ended 'what does the material say about X' questions."""
    result = vector_retriever.search(query_text=query, top_k=5)
    return [item.content for item in result.items]


@tool("Keyword-search-chunks")
def keyword_search_chunks(keyword: str):
    """Find lesson passages containing an exact keyword. Use when you need a
    literal term rather than a semantic match."""
    return run_query(
        "MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower($keyword) "
        "RETURN c.text AS text LIMIT 5",
        keyword=keyword,
    )


@tool("Lesson-content")
def lesson_content(name: str):
    """Return the text of a specific lesson (partial name match)."""
    return run_query(
        "MATCH (l:Lesson)<-[:PDF_OF]-(:Document)<-[:FROM_DOCUMENT]-(c:Chunk) "
        "WHERE toLower(l.name) CONTAINS toLower($name) "
        "RETURN l.name AS lesson, collect(c.text)[..10] AS chunks",
        name=name,
    )


# --- Catchall -------------------------------------------------------------

@tool("Query-database")
def query_database(query: str):
    """A catchall tool that turns a natural-language question into Cypher and
    runs it. Use for questions the other tools don't cover."""
    return text2cypher_retriever.get_search_results(query)


tools = [
    get_schema, count_nodes_by_label, count_relationships_by_type,
    list_lessons, list_modules, lessons_in_module,
    list_entity_types, entities_of_type, find_related_entities,
    most_connected_entities, path_between_entities, lessons_mentioning,
    search_lessons, keyword_search_chunks, lesson_content,
    query_database,
]

system_prompt = (
    "You are an assistant for a knowledge graph of course material. "
    "Choose the most specific tool for each question: the structure tools for "
    "lessons and modules, the entity tools to explore concepts and how they "
    "connect, and the search tools for the lesson text. Use Query-database only "
    "when nothing more specific fits, and check the schema if unsure."
)

agent = create_agent(model, tools, system_prompt=system_prompt)

query = "What are the most important concepts, and which lessons cover GraphRAG?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


# Try these to see different tools fire:
#   How many lessons are there, and how many in each module?
#   What types of entity are in the graph?
#   List the technologies described in the material.
#   What are the most connected concepts?
#   How is "Knowledge Graph" connected to "Vector Search"?
#   Which lessons mention RAG?
#   Find passages that contain the word "hallucination".
