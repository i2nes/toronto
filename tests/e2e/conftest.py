"""Pytest configuration and fixtures for E2E tests."""
import pytest
from playwright.sync_api import Page


# Test configuration
BASE_URL = "http://localhost:5001"
TEST_TIMEOUT = 30000  # 30 seconds


# Note: pytest-playwright provides these built-in options:
# --headed: Run tests in headed mode (visible browser)
# --slowmo: Slow down operations by N milliseconds
# --browser: Choose browser (chromium, firefox, webkit)
# --video: Record video (on, off, retain-on-failure)


@pytest.fixture
def chat_page(page: Page) -> Page:
    """Navigate to the chat page and return the page object."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.set_default_timeout(TEST_TIMEOUT)
    return page


@pytest.fixture
def test_message():
    """Standard test message."""
    return "Hello, this is a test message"


@pytest.fixture
def rag_test_message():
    """Test message that should trigger RAG retrieval."""
    return "What is the Toronto AI Assistant project?"
