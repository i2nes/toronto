---
title: Development Workflow and Best Practices
created: 2025-11-01
tags: [development, workflow, testing, git]
---

# Development Workflow and Best Practices

## Daily Development Workflow

### 1. Start Development Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run in watch mode (auto-reload)
./scripts/dev.sh
```

The server runs on http://localhost:5001

### 2. Watch Tailwind CSS

If you're modifying styles, run Tailwind in watch mode:

```bash
./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --watch
```

### 3. Run Tests

```bash
# Fast unit tests only
pytest -m "not e2e"

# All tests including E2E
pytest

# Specific test file
pytest tests/test_llm_client.py -v

# With coverage
pytest --cov=app --cov-report=html
```

## Code Quality

### Formatting

We use `black` and `isort` for consistent formatting:

```bash
# Format all Python files
black .
isort .

# Check without modifying
black --check .
isort --check .
```

### Type Checking

Use type hints throughout the codebase. Run mypy:

```bash
mypy app/
```

### Linting

If `ruff` is configured:

```bash
ruff check . --fix
```

## Git Workflow

### Branch Naming

- `feature/short-description` - New features
- `fix/short-description` - Bug fixes
- `chore/short-description` - Maintenance tasks
- `docs/short-description` - Documentation updates

### Commit Messages

Follow Conventional Commits:

```
feat: add dark mode toggle to settings
fix: prevent race condition in chunk indexing
chore: update dependencies to latest versions
docs: add API documentation for retrieval endpoint
refactor: simplify FAISS index initialization
test: add E2E tests for chat flow
```

**Format:**
```
<type>: <description>

[optional body]

[optional footer]
```

### Before Committing

Checklist:
- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black .` and `isort .`)
- [ ] Type hints added for new functions
- [ ] Documentation updated if API changed
- [ ] No secrets or `.env` files included

## Testing Strategy

### Unit Tests

Test individual functions and classes in isolation.

**Example:** Testing the Ollama client

```python
# tests/test_llm_client.py
import pytest
from app.llm_client import OllamaClient

@pytest.mark.asyncio
async def test_chat_success():
    client = OllamaClient()
    messages = [{"role": "user", "content": "Hello"}]
    response = await client.chat(messages)
    assert "message" in response
    assert "content" in response["message"]
```

### Integration Tests

Test multiple components working together.

**Example:** Testing the chat endpoint

```python
# tests/test_api.py
import pytest
from app.main import app

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with app.test_client() as client:
        response = await client.post(
            "/api/chat",
            json={"message": "Hello"}
        )
        assert response.status_code == 200
        data = await response.get_json()
        assert "response" in data
```

### E2E Tests

Test full user flows with Playwright.

**Example:** Testing the chat UI

```python
# tests/e2e/test_chat_flow.py
import pytest
from playwright.async_api import async_playwright

@pytest.mark.e2e
async def test_user_can_send_message():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto("http://localhost:5001")
        await page.fill('input[type="text"]', "What is RAG?")
        await page.click('button:has-text("Send")')

        # Wait for response
        await page.wait_for_selector('.chat-bubble-secondary')

        await browser.close()
```

## Database Migrations

When you change the database schema:

1. Update the schema in `app/db.py`
2. Test with a fresh database
3. Consider backward compatibility
4. Document breaking changes

**For SQLite:**
- We use direct schema creation (no migrations tool yet)
- To reset: delete `data/assistant.sqlite` and restart

**Future:** May add Alembic for migrations

## Debugging

### Structured Logs

All logs are JSON-formatted for easy parsing:

```python
import structlog
logger = structlog.get_logger()

logger.info("chat_request", user_id=123, message_length=50)
# Output: {"event": "chat_request", "user_id": 123, "message_length": 50, ...}
```

### Common Issues

**Issue:** Ollama connection refused
**Solution:** Check if Ollama is running: `ollama list`

**Issue:** Model not found
**Solution:** Pull the model: `ollama pull gemma3:12b`

**Issue:** FAISS dimension mismatch
**Solution:** Rebuild index with current model or switch to model that matches index

**Issue:** Port 5001 in use
**Solution:** Check running processes: `lsof -i :5001` and kill if needed

## Performance Monitoring

### Key Metrics

- **Chat response time**: Target <2s for simple queries
- **Indexing speed**: ~100-200 chunks/second
- **Memory usage**: Monitor FAISS index size
- **Database size**: SQLite file size should be reasonable

### Health Checks

```bash
# Check if app is ready
curl http://localhost:5001/health/ready

# Check if app is alive
curl http://localhost:5001/health/live
```

## Security Best Practices

1. **Never commit secrets**: Use `.env` files (gitignored)
2. **Validate user input**: Check length, sanitize queries
3. **Rate limiting**: Add if deploying beyond local use
4. **CORS**: Configure appropriately for production
5. **SQL injection**: Use parameterized queries (we do!)
6. **XSS**: Escape output in templates (Jinja2 does this)

## Documentation

### Code Documentation

- Add docstrings to all public functions
- Include type hints
- Document exceptions raised
- Provide usage examples

**Example:**
```python
async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
) -> Dict:
    """Send chat completion request to Ollama.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Optional model name (defaults to config.CHAT_MODEL)

    Returns:
        Response dict with 'message' containing 'content'

    Raises:
        httpx.HTTPError: If Ollama API request fails
    """
```

### Architecture Documentation

Keep ARCHITECTURE.md and DEVELOPMENT_PLAN.md up to date when making significant changes.

## Deployment

Currently, this is a local-first app for personal use. Future deployment options:

- **Docker**: Containerize for easy distribution
- **systemd**: Run as a service on Linux
- **Electron**: Package as desktop app
- **Cloud VM**: Deploy on VPS for remote access (with authentication!)

## Resources

- [Quart Documentation](https://quart.palletsprojects.com/)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Alpine.js Guide](https://alpinejs.dev/)
