# E2E Tests - Playwright

## Running Tests with Visible Browser

### Option 1: Basic Headed Mode (Recommended)
```bash
# Run with visible browser
pytest tests/e2e/ --headed

# Run single test with visible browser
pytest tests/e2e/test_demo.py::test_visual_demo --headed

# Run with visible browser + verbose output
pytest tests/e2e/ --headed -v
```

### Option 2: Headed Mode + Slow Motion
```bash
# Slow down operations by 500ms so you can see what's happening
pytest tests/e2e/test_demo.py --headed --slowmo=500

# Very slow (1000ms delay between operations)
pytest tests/e2e/ --headed --slowmo=1000
```

### Option 3: Debug Mode (Step Through)
```bash
# Run with Playwright Inspector (step through line by line)
PWDEBUG=1 pytest tests/e2e/test_demo.py --headed

# This opens the Playwright Inspector where you can:
# - Step through each action
# - Inspect elements
# - See screenshots
# - Copy selectors
```

### Option 4: Record Video
```bash
# Videos will be saved to test-results/
pytest tests/e2e/ --headed --video=on
```

## Quick Test Commands

```bash
# Headless (fast, for CI)
pytest tests/e2e/ -v

# Headed (watch it run)
pytest tests/e2e/ --headed

# Slow demo (easy to follow)
pytest tests/e2e/test_demo.py --headed --slowmo=500

# Single test file
pytest tests/e2e/test_basic_chat.py --headed

# Single specific test
pytest tests/e2e/test_basic_chat.py::test_page_loads --headed

# Skip E2E tests (for fast unit tests)
pytest -m "not e2e"
```

## Test Structure

- `test_basic_chat.py` - 14 tests for UI and basic chat functionality
- `test_rag.py` - 9 tests for RAG retrieval and sources
- `test_sessions.py` - 11 tests for session management
- `test_demo.py` - Visual demo test (great for watching)

## Debugging Tips

1. **See what's happening:**
   ```bash
   pytest tests/e2e/test_demo.py --headed --slowmo=500 -s
   ```

2. **Pause and inspect:**
   Add `await page.pause()` in your test code, then run:
   ```bash
   PWDEBUG=1 pytest tests/e2e/test_demo.py
   ```

3. **Take screenshots:**
   ```python
   await page.screenshot(path="debug.png")
   ```

4. **Print page content:**
   ```python
   print(await page.content())
   ```

## Before Running Tests

Make sure the server is running:
```bash
./scripts/dev.sh
# or
hypercorn app.main:app --bind 0.0.0.0:5001 --reload
```

## Common Issues

**Browser not found:** Run `python -m playwright install chromium`

**Connection refused:** Make sure the server is running on port 5001

**Tests timing out:** Increase timeout with `--timeout=60`
