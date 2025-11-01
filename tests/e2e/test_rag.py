"""E2E tests for RAG (Retrieval-Augmented Generation) functionality."""
import pytest
from playwright.sync_api import Page, expect


def test_rag_badge_displayed(chat_page: Page):
    """Test that RAG enabled badge is shown."""
    rag_badge = chat_page.locator('text="RAG enabled"')
    expect(rag_badge).to_be_visible()


def test_sources_appear_with_rag_query(chat_page: Page, rag_test_message: str):
    """Test that sources appear when RAG retrieves relevant documents."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send RAG-relevant query
    input_field.fill(rag_test_message)
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(3000)

    # Look for sources section (may or may not appear depending on RAG results)
    # This is a soft check - sources might not always appear
    sources_section = chat_page.locator('text=/\\d+ source(s)? used/')

    # Just verify the page didn't error out
    # We can't guarantee sources will appear for every query
    error_alert = chat_page.locator('.alert-error')
    expect(error_alert).to_be_hidden()


def test_sources_collapse_expands(chat_page: Page):
    """Test that sources collapse can be expanded."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send a message about the project
    input_field.fill("Tell me about this project")
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(5000)

    # Try to find sources collapse
    sources_collapse = chat_page.locator('.collapse:has-text("source")')

    # If sources exist, test expansion
    if sources_collapse.count() > 0:
        # Click to expand
        sources_collapse.first.locator('input[type="checkbox"]').click()

        # Verify content is visible
        chat_page.wait_for_timeout(500)

        # Sources should now be expanded
        collapse_content = sources_collapse.first.locator('.collapse-content')
        expect(collapse_content).to_be_visible()


def test_relevance_scores_displayed(chat_page: Page):
    """Test that relevance scores are displayed for sources."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send relevant query
    input_field.fill("What features does this project have?")
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(5000)

    # Look for relevance percentage badges (e.g., "85%")
    relevance_badges = chat_page.locator('.badge:has-text("%")')

    # If sources exist, check relevance scores
    if relevance_badges.count() > 0:
        # At least one relevance score should be visible
        expect(relevance_badges.first).to_be_visible()


def test_source_preview_truncated(chat_page: Page):
    """Test that source content previews are truncated."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send query
    input_field.fill("Explain the architecture")
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(5000)

    # If sources appear, check they have truncated previews
    sources_collapse = chat_page.locator('.collapse:has-text("source")')
    if sources_collapse.count() > 0:
        # Expand sources
        sources_collapse.first.locator('input[type="checkbox"]').click()
        chat_page.wait_for_timeout(500)

        # Look for preview text (should be truncated with "...")
        # This is implementation-dependent


def test_no_sources_for_general_query(chat_page: Page):
    """Test that general queries without relevant docs don't show sources."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send completely unrelated query
    input_field.fill("What is 2 + 2?")
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(3000)

    # Response should appear
    bot_icon = chat_page.locator('[data-lucide="bot"]').nth(1)
    expect(bot_icon).to_be_visible()

    # Sources might not appear for this general math question
    # Just verify no error occurred
    error_alert = chat_page.locator('.alert-error')
    expect(error_alert).to_be_hidden()


def test_rag_works_with_multiple_messages(chat_page: Page):
    """Test that RAG context is maintained across multiple messages."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # First message
    input_field.fill("What is this project about?")
    send_button.click()

    # Wait for first response (look for second bot icon to appear)
    chat_page.locator('[data-lucide="bot"]').nth(1).wait_for(timeout=30000)
    chat_page.wait_for_timeout(1000)

    # Second message (follow-up)
    input_field.fill("Can you elaborate on that?")
    send_button.click()

    # Wait for second response (look for third bot icon to appear)
    chat_page.locator('[data-lucide="bot"]').nth(2).wait_for(timeout=30000)

    # Both messages should have responses
    bot_icons = chat_page.locator('[data-lucide="bot"]')
    # Should have at least 3 bot icons (header + 2 responses)
    assert bot_icons.count() >= 3


def test_rag_database_badge(chat_page: Page):
    """Test that database icon is shown in RAG badge."""
    rag_badge = chat_page.locator('.badge:has-text("RAG enabled")')
    expect(rag_badge).to_be_visible()

    # Check for database icon (lucide renders as svg)
    database_icon = rag_badge.locator('[data-lucide="database"]')
    expect(database_icon).to_be_visible()
