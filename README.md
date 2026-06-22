# Workshop: Agent Memory with Neo4j — Scripts

Runnable Python scripts that back the **Agent Memory** workshop — a workshop-style
build of the GraphAcademy course
[`genai-context-graphs`](../monorepo/packages/internal/content/courses/genai-context-graphs)
("Context Graphs: Agent Memory with Neo4j"), and a follow-up to
[`workshop-genai`](../monorepo/packages/internal/content/courses/workshop-genai).

The workflow is **scripts-first**: we build the workshop here as standalone,
runnable `.py` scripts, get them working end to end, then port the code back
into the course lesson MDX.

All scripts use the [`neo4j-agent-memory`](https://pypi.org/project/neo4j-agent-memory/)
package (a Neo4j Labs project — experimental, community-supported) and its three
memory layers: **short-term**, **long-term**, and **reasoning**.

## Prerequisites

- Python 3.10+
- Neo4j 5.20+ (vector indexes are required). A local Docker instance or an Aura
  instance both work.
- An OpenAI API key (used for embeddings via `EmbeddingConfig`, and for the
  agent's LLM calls).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your values
```

## Environment

Set in `.env` (loaded automatically via `python-dotenv`):

| Variable | Purpose |
|---|---|
| `NEO4J_URI` | Bolt/neo4j URI (e.g. `neo4j://localhost:7687` or `neo4j+s://...aura...`) |
| `NEO4J_USERNAME` | Neo4j username (usually `neo4j`) |
| `NEO4J_PASSWORD` | Neo4j password |
| `OPENAI_API_KEY` | Embeddings (`EmbeddingConfig`) + the agent's LLM |

## Running

```bash
python scripts/01_introduction/05_complete_memory_api.py
```

`MemoryClient` initialises the schema and vector indexes automatically on first
run, so no manual migration step is needed.

## Structure

Scripts mirror the course's module/lesson tree so the port back to MDX is
one-to-one. Conceptual lessons (the "why" / "the problem" lessons) have no
script; only the code-bearing lessons do.

```
scripts/
  01_introduction/
    05_complete_memory_api.py       # M1L5 — every method, end to end
  02_short_term_memory/             # M2L3 python API, M2L4 entity-extraction preview
  03_long_term_memory/              # M3L4 extraction pipeline, M3L5 python API
  04_context_graphs_reasoning/      # M4L4 python API, M4L5 querying the trace, M4L6 lab
config.py                           # shared MemorySettings builder (DRY across scripts)
```

## Course → script mapping

| Module | Lesson | Script |
|---|---|---|
| 1 Introduction | 5 the complete memory API | `01_introduction/05_complete_memory_api.py` |
| 2 Short-term memory | 3 the python API | _to build_ |
| 2 Short-term memory | 4 entity extraction preview | _to build_ |
| 3 Long-term memory | 4 entity extraction pipeline | _to build_ |
| 3 Long-term memory | 5 the python API | _to build_ |
| 4 Context graphs & reasoning | 4 the python API | _to build_ |
| 4 Context graphs & reasoning | 5 querying the trace graph | _to build_ |
| 4 Context graphs & reasoning | 6 lab: agent trace | _to build_ |

> Status: scripts are authored from the course's documented API and have not yet
> been executed against an installed `neo4j-agent-memory` — run them against a
> live Neo4j before relying on the exact method signatures (the package is
> pre-1.0 and its API can shift).
