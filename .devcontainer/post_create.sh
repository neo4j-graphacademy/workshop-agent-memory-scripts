#!/usr/bin/env bash
pip3 install -r requirements.txt
pip3 install "neo4j-agent-memory[openai,pydantic-ai]"
[ -f .env ] || cp example.env .env
