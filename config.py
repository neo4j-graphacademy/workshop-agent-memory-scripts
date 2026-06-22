"""Shared MemorySettings builder for the workshop scripts.

Every script imports `build_settings()` so the Neo4j + embedding configuration
is defined once, mirroring the course's "build MemorySettings once" pattern
(M1L5). Credentials are read from the environment, loaded from a local .env.
"""

import os

from dotenv import load_dotenv
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.config import EmbeddingConfig, Neo4jConfig

load_dotenv()


def build_settings() -> MemorySettings:
    """Build MemorySettings from environment variables.

    Requires NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and OPENAI_API_KEY.
    """
    return MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
        ),
        embedding=EmbeddingConfig(api_key=os.environ["OPENAI_API_KEY"]),
    )
