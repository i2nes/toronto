# Toronto AI Assistant

A local-first AI assistant with RAG (Retrieval-Augmented Generation) over your markdown notes. Built with Python, Quart, Ollama, and FAISS.

## Features

- 🤖 **Local LLM**: Powered by Ollama (no data sent to cloud)
- 📚 **RAG Pipeline**: Semantic search over your markdown notes
- ⚡ **Fast Indexing**: FAISS vector search with 1024-dimensional embeddings
- 🎨 **Modern UI**: Alpine.js + Tailwind CSS + DaisyUI
- 🔍 **Source Citations**: See which notes were used to answer your questions
- 🚀 **Async Everything**: Built on Quart for high performance

## Quick Start

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) installed and running
- Models pulled: `ollama pull gemma3:12b` and `ollama pull mxbai-embed-large:latest`

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd toronto

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate setup
python scripts/validate_setup.py
```

### Initial Setup

```bash
# Build Tailwind CSS (first time)
./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --minify

# Index your notes (optional - sample notes included)
python scripts/reindex.py
```

## Running the Application

### Option 1: Quick Start (One Command)

```bash
source .venv/bin/activate && ./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --minify && hypercorn app.main:app --bind 0.0.0.0:5001 --reload
```

Then visit: **http://localhost:5001**

### Option 2: Development Mode (Recommended)

Run these in **two separate terminals**:

**Terminal 1** - Tailwind Watch (auto-rebuilds CSS on template changes):
```bash
./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --watch
```

**Terminal 2** - Development Server (auto-reloads on Python changes):
```bash
source .venv/bin/activate && hypercorn app.main:app --bind 0.0.0.0:5001 --reload
```

Then visit: **http://localhost:5001**

## Common Commands

### Development

| Task | Command |
|------|---------|
| **Build CSS (once)** | `./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --minify` |
| **Watch CSS** | `./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --watch` |
| **Start Server** | `source .venv/bin/activate && hypercorn app.main:app --bind 0.0.0.0:5001 --reload` |

### Content Management

| Task | Command |
|------|---------|
| **Reindex Notes** | `python scripts/reindex.py` |
| **Rebuild Index** | `python scripts/reindex.py --rebuild` |
| **Validate Setup** | `python scripts/validate_setup.py` |

### Testing & Quality

| Task | Command |
|------|---------|
| **Run Tests** | `pytest` |
| **Fast Tests** | `pytest -m "not e2e"` |
| **Format Code** | `black . && isort .` |
| **Lint Code** | `ruff check .` |

## Project Structure

```
toronto/
├── app/                      # Python application code
│   ├── config.py            # Configuration
│   ├── db.py                # SQLite database layer
│   ├── llm_client.py        # Ollama API client
│   ├── main.py              # Quart web application
│   └── rag/                 # RAG pipeline
│       ├── chunker.py       # Text chunking
│       ├── ingest.py        # Indexing pipeline
│       ├── md_parser.py     # Markdown parser
│       ├── retriever.py     # Semantic search
│       └── store_faiss.py   # Vector storage
├── web/                     # Frontend
│   ├── templates/           # Jinja2 templates
│   └── static/
│       ├── css/            # Built CSS
│       └── src/            # Tailwind source
├── notes/                   # Your markdown notes (indexed)
├── data/                    # SQLite DB + FAISS index
│   ├── assistant.sqlite    # Chunk metadata
│   ├── vectors.index       # FAISS vectors
│   └── metadata.json       # Index metadata
├── scripts/                 # Utility scripts
│   ├── dev.sh              # Development server
│   ├── reindex.py          # Reindex notes
│   └── validate_setup.py   # Setup validation
└── bin/
    └── tailwindcss         # Tailwind CLI binary
```

## Architecture

### RAG Pipeline

```
Markdown Notes
    ↓
Parse + Chunk (2400 chars, 320 overlap)
    ↓
Generate Embeddings (mxbai-embed-large, 1024d)
    ↓
Store in FAISS + SQLite
    ↓
User Query → Semantic Search → Top-K Chunks
    ↓
Context + Query → LLM → Response + Sources
```

### Tech Stack

**Backend:**
- Python 3.13+
- Quart (async Flask)
- Ollama (local LLM)
- FAISS (vector search)
- SQLite (metadata)

**Frontend:**
- HTML + Tailwind CSS + DaisyUI
- Alpine.js (minimal reactive JS)

**Models:**
- Chat: `gemma3:12b` (12B parameters)
- Embeddings: `mxbai-embed-large:latest` (1024 dimensions)

## Configuration

Edit `.env` (copy from `.env.example`):

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
CHAT_MODEL=gemma3:12b
EMBEDDING_MODEL=mxbai-embed-large:latest

# RAG Configuration
CHUNK_SIZE=2400
CHUNK_OVERLAP=320
RETRIEVAL_TOP_K=8
```

## Adding Your Notes

1. Place markdown files in the `notes/` directory
2. Run the indexer:
   ```bash
   python scripts/reindex.py
   ```
3. Ask questions about your notes in the chat interface!

### Note Format

Markdown files can include optional YAML frontmatter:

```markdown
---
title: My Note
tags: [topic, subtopic]
created: 2025-11-01
---

# My Note Content

Your content here...
```

## Health Checks

The application exposes health check endpoints:

- `GET /health/live` - Liveness probe (is the app running?)
- `GET /health/ready` - Readiness probe (can it serve requests? checks Ollama + models)

```bash
curl http://localhost:5001/health/ready
```

## Development Phases

- ✅ **Phase 0**: Project Bootstrap
- ✅ **Phase 1**: Basic Chat UI
- ✅ **Phase 2**: RAG Pipeline (Indexing)
- ✅ **Phase 3**: RAG-Enhanced Chat
- 📋 **Phase 4**: Sessions & Memory (planned)
- 📋 **Phase 5**: Tool Calling (planned)
- 📋 **Phase 6**: Polish & Production (planned)

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for details.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and technical details
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Phased implementation plan
- [CLAUDE.md](CLAUDE.md) - Repository rules and conventions
- [NEXT_STEPS_ANALYSIS.md](NEXT_STEPS_ANALYSIS.md) - Strategic recommendations

## Troubleshooting

### Ollama connection refused

```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed (depends on your installation)
```

### Model not found

```bash
# Pull the required models
ollama pull gemma3:12b
ollama pull mxbai-embed-large:latest
```

### FAISS dimension mismatch

```bash
# Rebuild the index with the correct model
python scripts/reindex.py --rebuild
```

### Port already in use

```bash
# Find process using port 5001
lsof -i :5001

# Kill it
kill <PID>
```

### CSS not updating

```bash
# Rebuild Tailwind CSS
./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --minify

# Hard refresh browser (Cmd+Shift+R or Ctrl+Shift+R)
```

## Contributing

This is a personal project, but if you find issues or have suggestions:

1. Check existing documentation
2. Run validation: `python scripts/validate_setup.py`
3. Follow the conventions in [CLAUDE.md](CLAUDE.md)

## License

[Your License Here]

## Acknowledgments

- Built with [Ollama](https://ollama.ai)
- UI components from [DaisyUI](https://daisyui.com/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)
- Inspired by modern RAG architectures
