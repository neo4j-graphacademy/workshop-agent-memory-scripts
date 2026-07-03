# INSTRUCTOR ONLY - reset the shared workshop instance between workshop runs.
#
# Clears all memory data (conversations, messages, memory entities, preferences,
# facts, traces, users) from the shared instance, previewing counts and asking
# for confirmation first. The course knowledge graph is left untouched.

import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

if not os.getenv("MVP_NEO4J_URI"):
    raise SystemExit("MVP_NEO4J_URI is not set - add the shared instance values to your .env first.")

# Memory nodes only. KG entities carry __KGBuilder__/__Entity__ and stay.
MEMORY_FILTER = """
    n:Conversation OR n:Message OR n:Preference OR n:Fact
    OR n:ReasoningTrace OR n:ReasoningStep OR n:AgentStep OR n:ToolCall
    OR n:User OR n:Extractor
    OR (n:Entity AND NOT (n:__KGBuilder__ OR n:__Entity__))
"""


def main():
    driver = GraphDatabase.driver(
        os.environ["MVP_NEO4J_URI"],
        auth=(os.environ["MVP_NEO4J_USERNAME"], os.environ["MVP_NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()

    records, _, _ = driver.execute_query(
        f"MATCH (n) WHERE {MEMORY_FILTER} "
        "RETURN [l IN labels(n) WHERE NOT l STARTS WITH '__'][0] AS label, count(n) AS n ORDER BY label"
    )
    if not records:
        print("No memory data in the shared instance.")
        driver.close()
        return

    print("Memory data in the shared instance:")
    for r in records:
        print(f"  - {r['label']}: {r['n']}")

    answer = input("\nClear ALL of these? The knowledge graph stays. Type 'yes' to proceed: ").strip().lower()
    if answer != "yes":
        print("Aborted - nothing cleared.")
        driver.close()
        return

    deleted = 0
    while True:
        records, _, _ = driver.execute_query(
            f"MATCH (n) WHERE {MEMORY_FILTER} "
            "WITH n LIMIT 1000 DETACH DELETE n RETURN count(n) AS n"
        )
        batch = records[0]["n"]
        deleted += batch
        if batch == 0:
            break

    driver.close()
    print(f"\nCleared {deleted} memory node(s). The knowledge graph is untouched.")


if __name__ == "__main__":
    main()
