---
title: Toronto AI Assistant - Project Overview
created: 2025-11-01
tags: [project, overview, architecture]
---

# Toronto AI Assistant - Project Overview

## What is Toronto AI?

Toronto AI is a local-first AI assistant built with Python, Quart, and Ollama. The goal is to create a personal knowledge assistant that can answer questions based on your markdown notes using Retrieval-Augmented Generation (RAG).

## Key Features

### Local-First Architecture

All processing happens locally on your machine. No data is sent to external APIs or cloud services. This ensures:

- **Privacy**: Your notes never leave your computer
- **Speed**: No network latency for LLM inference
- **Cost**: No API charges from cloud LLM providers
- **Reliability**: Works offline

### RAG Over Markdown Notes

The system indexes your markdown notes and uses semantic search to find relevant context when answering questions. This allows the AI to reference your personal knowledge base.

## Technology Stack

### Backend
- **Python 3.13+**: Modern async Python
- **Quart**: Async web framework (async Flask)
- **SQLite**: Lightweight database for metadata
- **FAISS**: Facebook's vector similarity search library
- **Structured Logging**: JSON logs with structlog

### AI/ML
- **Ollama**: Local LLM inference
- **gemma3:12b**: Chat model for conversations
- **mxbai-embed-large**: Embedding model (1024 dimensions)

### Frontend
- **HTML + Tailwind CSS**: Utility-first styling
- **DaisyUI**: Tailwind component library
- **Alpine.js**: Minimal reactive JavaScript

## Development Philosophy

### Incremental Development

The project is built in phases, with each phase being independently testable:

1. **Phase 0**: Bootstrap and setup
2. **Phase 1**: Basic chat interface
3. **Phase 2**: RAG pipeline and indexing
4. **Phase 3**: RAG-enhanced chat
5. **Phase 4**: Advanced features (sessions, tools)

### Minimal Dependencies

We prefer simple, well-maintained libraries over complex frameworks. The entire dependency tree is kept small and manageable.

### Test-Driven

Every feature includes tests. Unit tests use pytest, E2E tests use Playwright.

## Project Structure

```
toronto/
├── app/                  # Python application code
│   ├── config.py        # Configuration
│   ├── db.py            # Database layer
│   ├── llm_client.py    # Ollama client
│   ├── main.py          # Quart app
│   └── rag/             # RAG pipeline
├── web/                 # Frontend assets
│   ├── templates/       # Jinja2 templates
│   └── static/          # CSS, JS
├── notes/               # Markdown notes (indexed)
├── data/                # SQLite DB, FAISS index
├── scripts/             # CLI utilities
└── tests/               # Test suites
```

## Getting Started

### Prerequisites

- Python 3.13+
- Ollama installed and running
- Models pulled: gemma3:12b, mxbai-embed-large:latest

### Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
./scripts/dev.sh
```

Visit http://localhost:5001 to use the chat interface.

## Current Status

As of November 1, 2025:
- ✅ Phase 0 complete (infrastructure)
- ✅ Phase 1 complete (basic chat)
- 🚧 Phase 2 in progress (RAG indexing)

## Future Plans

- Session management for conversation history
- Tool calling (web search, calculator, etc.)
- File upload and processing
- Export conversations
- Dark mode and themes
- Mobile-responsive UI improvements
