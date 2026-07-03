# Loads the course knowledge graph into your Neo4j instance (module 1).
# Reads the CSV export in data/ and writes with MERGE throughout, so running
# it again is safe. Run with: python load_knowledge_graph.py

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# The entity labels and relationship types present in the export. Statements
# are built per literal label/type, so anything outside these lists fails
# loudly rather than being interpolated.
ENTITY_LABELS = ["Technology", "Concept", "Example", "Process", "Challenge", "Benefit", "Resource"]
REL_TYPES = ["RELATED_TO", "PART_OF", "USED_IN", "LEADS_TO", "HAS_CHALLENGE", "CITES"]


def unesc(text):
    """Text fields escape newlines as \\n in the CSVs - reverse that."""
    return text.replace("\\n", "\n").replace("\\\\", "\\")


def read_csv(name):
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_batched(driver, cypher, rows, batch_size=1000):
    for i in range(0, len(rows), batch_size):
        driver.execute_query(cypher, rows=rows[i:i + batch_size], database_=DATABASE)


def main():
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()

    # The vector index the GraphRAG retriever searches.
    driver.execute_query(
        """CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
           FOR (c:Chunk) ON (c.embedding)
           OPTIONS {indexConfig: {
             `vector.dimensions`: 1536,
             `vector.similarity_function`: 'cosine'
           }}""",
        database_=DATABASE,
    )

    lessons = read_csv("lessons.csv")
    run_batched(driver, """
        UNWIND $rows AS row
        MERGE (l:Lesson {url: row.url})
        SET l.name = row.name, l.module = row.module, l.course = row.course
        """, lessons)
    print(f"Lessons: {len(lessons)}")

    documents = read_csv("documents.csv")
    run_batched(driver, """
        UNWIND $rows AS row
        MATCH (l:Lesson {url: row.lesson_url})
        MERGE (d:Document {path: row.path})
        SET d:__KGBuilder__,
            d.createdAt = datetime(row.created_at),
            d.document_type = row.document_type
        MERGE (d)-[:PDF_OF]->(l)
        """, documents)
    print(f"Documents: {len(documents)}")

    chunks = [{
        "document_path": row["document_path"],
        "index": int(row["index"]),
        "text": unesc(row["text"] or ""),
        "embedding": json.loads(row["embedding"] or "[]"),
    } for row in read_csv("chunks.csv")]
    # Small batches: each row carries a 1536-float embedding.
    run_batched(driver, """
        UNWIND $rows AS row
        MATCH (d:Document {path: row.document_path})
        MERGE (c:Chunk {index: row.index})-[:FROM_DOCUMENT]->(d)
        SET c:__KGBuilder__, c.text = row.text, c.embedding = row.embedding
        """, chunks, batch_size=100)
    print(f"Chunks: {len(chunks)}")

    next_chunk = [{
        "document_path": row["document_path"],
        "from_index": int(row["from_index"]),
        "to_index": int(row["to_index"]),
    } for row in read_csv("next_chunk.csv")]
    run_batched(driver, """
        UNWIND $rows AS row
        MATCH (a:Chunk {index: row.from_index})-[:FROM_DOCUMENT]->(d:Document {path: row.document_path})
        MATCH (b:Chunk {index: row.to_index})-[:FROM_DOCUMENT]->(d)
        MERGE (a)-[:NEXT_CHUNK]->(b)
        """, next_chunk)
    print(f"Chunk chain: {len(next_chunk)}")

    entities = [{
        "label": row["label"],
        "name": unesc(row["name"] or ""),
        "alias": unesc(row["alias"]) if row["alias"] else None,
    } for row in read_csv("entities.csv")]
    unknown = {e["label"] for e in entities} - set(ENTITY_LABELS)
    if unknown:
        raise SystemExit(f"Unknown entity labels in entities.csv: {sorted(unknown)}")
    for label in ENTITY_LABELS:
        group = [e for e in entities if e["label"] == label]
        run_batched(driver, f"""
            UNWIND $rows AS row
            MERGE (e:`{label}` {{name: row.name}})
            SET e:__Entity__, e:__KGBuilder__, e.alias = row.alias
            """, group)
    print(f"Entities: {len(entities)}")

    rels = [{
        "source_label": row["source_label"],
        "source_name": unesc(row["source_name"] or ""),
        "type": row["type"],
        "target_label": row["target_label"],
        "target_name": unesc(row["target_name"] or ""),
    } for row in read_csv("entity_rels.csv")]
    unknown = {r["type"] for r in rels} - set(REL_TYPES)
    if unknown:
        raise SystemExit(f"Unknown relationship types in entity_rels.csv: {sorted(unknown)}")
    for rel_type in REL_TYPES:
        group = [r for r in rels if r["type"] == rel_type]
        run_batched(driver, f"""
            UNWIND $rows AS row
            MATCH (a:__Entity__ {{name: row.source_name}}) WHERE row.source_label IN labels(a)
            MATCH (b:__Entity__ {{name: row.target_name}}) WHERE row.target_label IN labels(b)
            MERGE (a)-[:`{rel_type}`]->(b)
            """, group)
    print(f"Entity relationships: {len(rels)}")

    from_chunk = [{
        "entity_label": row["entity_label"],
        "entity_name": unesc(row["entity_name"] or ""),
        "document_path": row["document_path"],
        "chunk_index": int(row["chunk_index"]),
    } for row in read_csv("from_chunk.csv")]
    run_batched(driver, """
        UNWIND $rows AS row
        MATCH (e:__Entity__ {name: row.entity_name}) WHERE row.entity_label IN labels(e)
        MATCH (c:Chunk {index: row.chunk_index})-[:FROM_DOCUMENT]->(:Document {path: row.document_path})
        MERGE (e)-[:FROM_CHUNK]->(c)
        """, from_chunk)
    print(f"Entity provenance: {len(from_chunk)}")

    records, _, _ = driver.execute_query(
        """MATCH (l:Lesson) WITH count(l) AS lessons
           MATCH (c:Chunk) WITH lessons, count(c) AS chunks
           MATCH (e:__Entity__) RETURN lessons, chunks, count(e) AS entities""",
        database_=DATABASE,
    )
    row = records[0]
    driver.close()
    print(f"\nDone. The graph holds {row['lessons']} lessons, {row['chunks']} chunks, and {row['entities']} entities.")
    print("Your agent has a knowledge graph to search.")


if __name__ == "__main__":
    main()
