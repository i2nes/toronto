#!/usr/bin/env python
"""Validate Phase 0 setup - check all dependencies and configuration."""
import sys
import asyncio
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")

def print_info(msg):
    print(f"{BLUE}ℹ{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")

def print_section(title):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

async def main():
    print_section("Toronto AI Assistant - Setup Validation")

    errors = []
    warnings = []

    # 1. Python version check
    print_section("1. Python Environment")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_info(f"Python version: {python_version}")
    if sys.version_info >= (3, 10):
        print_success("Python version >= 3.10")
    else:
        print_error("Python version < 3.10 (required)")
        errors.append("Python version too old")

    # Check if in venv
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print_success("Running in virtual environment")
    else:
        print_warning("Not running in virtual environment (recommended)")
        warnings.append("Not in venv")

    # 2. Import core dependencies
    print_section("2. Core Dependencies")

    dependencies = [
        ("quart", "Quart web framework"),
        ("hypercorn", "Hypercorn ASGI server"),
        ("ollama", "Ollama Python client"),
        ("httpx", "HTTP client"),
        ("faiss", "FAISS vector store"),
        ("langchain_text_splitters", "Text splitters"),
        ("pydantic", "Data validation"),
        ("aiosqlite", "Async SQLite"),
        ("watchdog", "File monitoring"),
        ("structlog", "Structured logging"),
        ("pytest", "Testing framework"),
    ]

    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print_success(f"{description:30} ({module_name})")
        except ImportError as e:
            print_error(f"{description:30} ({module_name}) - {e}")
            errors.append(f"Missing: {module_name}")

    # 3. Test configuration
    print_section("3. Configuration")

    try:
        # Add parent directory to path to import app
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app import config

        print_success(f"Config loaded successfully")
        print_info(f"  Chat model: {config.CHAT_MODEL}")
        print_info(f"  Embedding model: {config.EMBEDDING_MODEL}")
        print_info(f"  Ollama URL: {config.OLLAMA_BASE_URL}")
        print_info(f"  Chunk size: {config.CHUNK_SIZE} chars")
        print_info(f"  Data directory: {config.DATA_DIR}")

        # Check directories exist
        if config.DATA_DIR.exists():
            print_success(f"Data directory exists: {config.DATA_DIR}")
        else:
            print_error(f"Data directory missing: {config.DATA_DIR}")
            errors.append("Data directory missing")

        if config.NOTES_DIR.exists():
            print_success(f"Notes directory exists: {config.NOTES_DIR}")
        else:
            print_error(f"Notes directory missing: {config.NOTES_DIR}")
            errors.append("Notes directory missing")

    except Exception as e:
        print_error(f"Failed to load config: {e}")
        errors.append("Config loading failed")
        return errors, warnings

    # 4. Test Ollama connection
    print_section("4. Ollama Service")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()

            print_success(f"Ollama service running at {config.OLLAMA_BASE_URL}")

            # Check for required models
            models = {m['name'] for m in data.get('models', [])}
            print_info(f"Found {len(models)} models installed")

            # Check chat model
            if config.CHAT_MODEL in models:
                print_success(f"Chat model available: {config.CHAT_MODEL}")
            else:
                print_error(f"Chat model missing: {config.CHAT_MODEL}")
                print_info(f"  Run: ollama pull {config.CHAT_MODEL}")
                errors.append(f"Missing chat model: {config.CHAT_MODEL}")

            # Check embedding model
            if config.EMBEDDING_MODEL in models:
                print_success(f"Embedding model available: {config.EMBEDDING_MODEL}")
            else:
                print_error(f"Embedding model missing: {config.EMBEDDING_MODEL}")
                print_info(f"  Run: ollama pull {config.EMBEDDING_MODEL}")
                errors.append(f"Missing embedding model: {config.EMBEDDING_MODEL}")

            # Show all available models
            if models:
                print_info(f"All installed models:")
                for model in sorted(models):
                    print(f"    - {model}")

    except httpx.ConnectError:
        print_error("Cannot connect to Ollama service")
        print_info(f"  Make sure Ollama is running: ollama serve")
        errors.append("Ollama not running")
    except Exception as e:
        print_error(f"Ollama check failed: {e}")
        errors.append(f"Ollama error: {e}")

    # 5. Test Ollama API with a simple request
    print_section("5. Ollama API Test")

    try:
        import ollama
        # Test embedding (faster than chat)
        response = await asyncio.to_thread(
            ollama.embeddings,
            model=config.EMBEDDING_MODEL,
            prompt="test"
        )

        if 'embedding' in response:
            dimension = len(response['embedding'])
            print_success(f"Embedding API working (dimension: {dimension})")

            # Verify expected dimension for mxbai-embed-large
            if config.EMBEDDING_MODEL == "mxbai-embed-large" and dimension == 1024:
                print_success(f"Embedding dimension correct (1024)")
            else:
                print_info(f"Embedding dimension: {dimension} (will be auto-detected)")
        else:
            print_error("Embedding response missing 'embedding' field")
            errors.append("Embedding API issue")

    except Exception as e:
        print_error(f"Ollama API test failed: {e}")
        errors.append(f"API test failed: {e}")

    # 6. Check Tailwind CSS
    print_section("6. Tailwind CSS")

    tailwind_binary = Path(__file__).parent.parent / "bin" / "tailwindcss"
    if tailwind_binary.exists():
        print_success(f"Tailwind binary found: {tailwind_binary}")
        # Check if executable
        if tailwind_binary.stat().st_mode & 0o111:
            print_success("Tailwind binary is executable")
        else:
            print_warning("Tailwind binary not executable")
            warnings.append("Tailwind not executable")
    else:
        print_error(f"Tailwind binary missing: {tailwind_binary}")
        errors.append("Tailwind binary missing")

    css_file = Path(__file__).parent.parent / "web" / "static" / "css" / "app.css"
    if css_file.exists():
        size_kb = css_file.stat().st_size / 1024
        print_success(f"CSS file built: {css_file} ({size_kb:.1f}KB)")
    else:
        print_error(f"CSS file missing: {css_file}")
        print_info("  Run: ./bin/tailwindcss -i web/static/src/styles.css -o web/static/css/app.css --minify")
        errors.append("CSS not built")

    # 7. Summary
    print_section("Summary")

    if not errors:
        print_success(f"All checks passed! ✨")
        print_info(f"\n  You're ready to proceed to Phase 1 (Basic Chat UI)")
        print_info(f"  Next step: Build app/main.py and web/templates/chat.html")
    else:
        print_error(f"Found {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if warnings:
        print_warning(f"\nFound {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    print()
    return errors, warnings

if __name__ == "__main__":
    errors, warnings = asyncio.run(main())
    sys.exit(1 if errors else 0)
