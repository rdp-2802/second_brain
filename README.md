# Second Brain (v1)

A personal, RAG-based knowledge management system that turns your conversations into a structured, queryable memory — classifying what you say into typed memory (stories, beliefs, habits) instead of just dumping raw chat logs into a vector store.

---

## What it does

- Ingests conversations and extracts **Memory Details** — specific, new pieces of information tied back to the source message
- Clusters Details under **Memory Summaries** — broader, persistent themes that Details attach to (or spawn, if nothing fits)
- Summarizes conversations block-by-block, maintaining continuity across blocks
- Serves it all through an API with auth, multiuser support, and a Streamlit front end

---

## Tech Stack

| Layer | Tech |
|---|---|
| Language | Python |
| Database | PostgreSQL + `pgvector` |
| ORM / Migrations | SQLAlchemy + Alembic |
| Embeddings | gemini-embedding-2 |
| Generation | gemini-3.5-flash-lite |
| Similarity Search | pgvector, HNSW index |
| Frontend | Streamlit |
| Auth / API | Custom API layer with user authorization |

**Not in v1:** chunking, reranking, document parsing, retrieval classifier.

---

## Architecture (bottom-up, modular)

Each stage has an explicit input/output contract:

1. **Ingestion Layer** — parses raw input → extracts Memory Details → attaches to a Summary or creates a new one
2. **Summarization Layer** — block-based summarization with continuity from prior blocks
3. **Retrieval** — pulls candidate context via pgvector similarity search
4. **Chat Orchestration Layer** — ties ingestion, summarization, and retrieval together to drive the actual conversation
5. **API Layer** — auth + multiuser support, sitting in front of everything

### Ingestion logic
Single-string LLM prompt extracts Memory Details first (specific new info tied to source message IDs), then attaches each Detail to an existing Memory Summary or spawns a new one.

---

## Status: v1 is complete

API, auth, and multiuser support are done and live — this isn't a "coming soon," it's shipped.

---

## Roadmap — v2

- **Better retrieval evaluation** — a proper way to measure whether retrieved context is actually good, not just eyeballing it
- **Agentic capabilities** — let the system handle retrieval, web search, email lookup, etc. on its own rather than being purely reactive
- **Source diversity** — voice and PDF ingestion, beyond text/chat input
- **Goal tracking** — mechanisms to track goals over time, not just knowledge
- **Better UI/UX for self-clarity** — surfacing patterns and habits back to the user in a way that's actually useful to reflect on
- **A real frontend** — replacing Streamlit with a proper front-end build

---

*v1 built as a from-scratch architecture exercise — modular, bottom-up, with explicit contracts between stages. v2 is about layering intelligence (agentic, eval, tracking) and polish (UI, ingestion breadth) on top of a working core.*