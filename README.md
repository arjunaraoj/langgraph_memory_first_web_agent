# LangGraph Memory-First Web Agent

This repository implements a **Memory-First Web Agent in Python using LangGraph**.

The agent answers user questions by checking Redis vector memory first. If memory has a strong semantic match, the agent answers from Redis only. If memory misses, it searches the web with Tavily, fetches web pages, converts HTML to Markdown, summarizes and chunks the content, stores the new chunks in Redis, and then answers with source URLs.

## Why LangGraph?

LangGraph is used for orchestration because the workflow is naturally a graph: guardrails, embedding, Redis search, conditional routing, web fallback, answer generation, and logging. LangGraph `StateGraph` workflows are compiled into executable graphs that can be invoked like normal runnables.

## Architecture

```text
User Question
   |
   v
LangGraph START
   |
   v
[guardrails]
   |
   v
[embed_query]
   |
   v
[search_memory in Redis]
   |
   +-- similarity >= threshold and not freshness-sensitive --> [answer_from_memory]
   |
   +-- similarity < threshold or needs fresh info -----------> [web_search_fetch_store]
                                                               |
                                                               v
                                                        [answer_from_web]
   |
   v
[log_turn]
   |
   v
END
```

## Functional coverage

- Python implementation
- LangGraph orchestration
- Redis vector memory
- OpenAI embeddings
- Memory-first routing with configurable threshold, default `0.7`
- Tavily web search fallback
- Web page fetch with timeout and retries
- HTML to Markdown conversion
- Summarization, chunking, embedding, and Redis ingestion
- Grounded answers with source URLs and similarity score
- Two LLM roles
- JSONL logging
- Analytics report
- Basic prompt-injection guardrails
- Reliability with retries and graceful warnings

## LLM choices

The default configuration uses:

- `CONVERSATION_MODEL=gpt-4o-mini`
- `ANALYTICS_MODEL=gpt-4o-mini`
- `EMBEDDING_MODEL=text-embedding-3-small`

The conversation model generates user-facing grounded answers. The analytics model performs lower-risk tasks such as summarization and classification. In production, the analytics model can be replaced with an even cheaper/faster model if available in your account.

`text-embedding-3-small` is used because it is a small embedding model suitable for semantic search and retrieval workloads.

## Environment variables

Create `.env` from `.env.example`:

```env
OPENAI_API_KEY=
REDIS_URL=redis://localhost:6379
WEB_SEARCH_API_KEY=
MEMORY_SIMILARITY_THRESHOLD=0.7
CONVERSATION_MODEL=gpt-4o-mini
ANALYTICS_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

The code also supports `open_ai_key` as a fallback alias. If both are present, `OPENAI_API_KEY` is used first.

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

Windows:

```powershell
copy .env.example .env
notepad .env
```

macOS/Linux:

```bash
cp .env.example .env
nano .env
```

### 4. Start Redis Stack

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

If the container already exists:

```bash
docker ps
docker rm -f redis-stack
```

Then run the Docker command again.

### 5. Ingest sample data

```bash
python ingest.py --reset
```

### 6. Run demo

```bash
python agent.py
```

### 7. Ask one question

```bash
python agent.py "What is the Singapore Tourism Board and what are its main responsibilities?"
```

### 8. Run analytics

```bash
python analytics.py
```

### 9. Run tests

```bash
pytest -q
```

## Demo queries

The project includes three demo queries:

```python
DEMO_QUERIES = [
    "What is the Singapore Tourism Board and what are its main responsibilities?",
    "What are the latest visitor arrival trends for Singapore tourism?",
    "How is Singapore performing in visitor arrivals recently?",
]
```

Expected behavior:

1. First query: memory hit from seeded Redis data.
2. Second query: forced web search because it asks for latest/recent information.
3. Third query: may use web again if the question is freshness-sensitive; for non-current repeat questions, newly stored web content becomes a memory hit.

For a pure repeat-memory demo, use a non-current query for Demo 2 and a similar non-current query for Demo 3.

## Logging

Each turn is logged to `logs/agent_turns.jsonl` with:

- timestamp
- question
- memory hit/miss
- Redis similarity score
- whether web search was used
- URLs fetched
- number of chunks stored
- answer summary
- analytics classification
- warnings/errors

## Analytics

Run:

```bash
python analytics.py
```

Example output:

```text
Analytics report
total_turns: 3
count_by_topic: {'organization_overview': 1, 'tourism_statistics': 2}
count_by_question_type: {'factual': 2, 'how_to': 1}
count_by_memory_hit_vs_miss: {'memory_hit': 1, 'memory_miss': 2}
most_common_user_intents: {'understand_stb_role': 1}
most_frequently_used_sources: {'www.stb.gov.sg': 2}
```

## Security guardrails

The project includes basic guardrails:

- Blocks unsafe medical/emergency/self-harm style questions.
- Treats web pages as untrusted evidence, not instructions.
- Removes suspicious retrieved lines such as “ignore previous instructions” or “reveal your API key”.
- Keeps system instructions separate from retrieved content.
- Forces web search for current/freshness-sensitive questions such as current leaders, latest statistics, or recent updates.
- Does not expose API keys.

## Reliability

The project uses:

- Tenacity retries for embeddings, LLM calls, web search, and page fetch.
- HTTP timeouts.
- Graceful warnings for blocked pages, YouTube pages, 403 pages, or unusable content.
- Chunking for long pages.
- Configurable token/content limits.

## GitHub Copilot / AI assistance disclosure

This repository was generated with ChatGPT assistance. No GitHub Copilot was used.

AI assistance was used to draft the repository structure, Python implementation, README, tests, and setup instructions. The generated code was reviewed for the requested requirements: Redis memory-first routing, LangGraph orchestration, web fallback, logging, analytics, guardrails, retries, and configuration. Manual corrections were applied for Windows-friendly commands, `.env` handling, Redis vector indexing, and freshness-sensitive routing.

## Assumptions and limitations

- You need valid OpenAI and Tavily API keys.
- Redis Stack must be running locally or available through `REDIS_URL`.
- Some websites block scraping or return unusable pages.
- Current facts can become stale, so freshness-sensitive questions are forced to web search.
- This is a local demo repository, not a hosted production service.
- Analytics are intentionally simple and can be extended with dashboards or database storage.

## Optional Gradio UI

This project also includes a Gradio web UI in `gradio_app.py`. Gradio is useful for quickly creating a browser-based demo for a Python function or ML/AI app.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the UI:

```bash
python gradio_app.py
```

The UI includes:

- Ask Question tab
- Chat Mode tab
- Demo Scenarios tab
- Runtime options for similarity threshold, top-k memory results, top-k web results, and forced web search
- Data ingestion button
- Analytics viewer
- Configuration viewer

For the latest Gradio package, you can also run:

```bash
pip install -U gradio
```
