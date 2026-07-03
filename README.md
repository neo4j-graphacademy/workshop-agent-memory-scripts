# Neo4j Agent Memory Workshop

Companion code for the [Neo4j Agent Memory Workshop](https://graphacademy.neo4j.com/courses/workshop-agent-memory/) on GraphAcademy. The workshop takes a GraphRAG agent and gives it persistent, explainable memory with [`neo4j-agent-memory`](https://neo4j.com/labs/agent-memory/).

## Setup

Open this repository in a GitHub Codespace — the packages install automatically, leaving only the `.env` steps below. Or set up locally:

```bash
pip install -r requirements.txt
pip install "neo4j-agent-memory[all]"
cp example.env .env   # then fill in your values
python test_environment.py
```

Every check should pass or be skipped before you start. Your instructor provides the shared `MEMORY_API_KEY` on the day.

## The files, in workshop order

| File | Used in | What it is |
| --- | --- | --- |
| `nams_quickstart.py` | Module 1 | Your first memory, stored in the hosted workspace |
| `agent_no_memory.py` | Module 1 | The starting agent — it forgets everything on restart |
| `memory_agent_mvp.py` | Module 1 | The finished memory agent, on the shared workspace |
| `agent.py` | Modules 2-5 | **The file you build.** Starts as a copy of `agent_no_memory.py`; the marked sections fill in one memory layer at a time until it matches the MVP |
| `complete_memory_api.ipynb` | Modules 2-4 (optional) | The API tour notebook — every memory surface, one cell at a time |
| `solutions/` | Modules 2-5 | `agent.py` as it should look at the end of each module |
| `data/workshop-agent-memory.dump` | Setup | The course knowledge graph, for loading into your own instance |
| `instructor/` | — | Instructor-only tooling; nothing in here is part of the lessons |

## The shape of the workshop

You run the finished memory agent first, then build your own from the same starting point, layer by layer — short-term, long-term, reasoning — until the two match. The closing challenge turns the agent into one of your own.
