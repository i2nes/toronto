"""E2E tests for session management and persistence."""
import pytest
from playwright.sync_api import Page, expect


def test_session_sidebar_visible(chat_page: Page):
    """Test that session sidebar is visible."""
    sidebar = chat_page.locator('aside.w-64')
    expect(sidebar).to_be_visible()


def test_new_chat_creates_session(chat_page: Page, test_message: str):
    """Test that sending a message creates a new session."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Initially should show "No previous chats"
    no_chats_msg = chat_page.locator('text="No previous chats"')
    # May or may not be visible depending on existing sessions

    # Send a message
    input_field.fill(test_message)
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(3000)

    # Session should now appear in sidebar
    # The session title should be based on the first message
    session_button = chat_page.locator('aside button:has-text("' + test_message[:20] + '")')

    # Give it a moment to appear
    chat_page.wait_for_timeout(1000)

    # Session list should now have at least one item
    session_buttons = chat_page.locator('aside button.btn-ghost')
    assert session_buttons.count() >= 1


def test_session_shows_timestamp(chat_page: Page, test_message: str):
    """Test that sessions display relative timestamps."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send a message to create session
    input_field.fill(test_message)
    send_button.click()
    chat_page.wait_for_timeout(3000)

    # Look for timestamp text in sidebar (e.g., "Just now", "2m ago", etc.)
    timestamp = chat_page.locator('aside .text-xs.opacity-60').first
    expect(timestamp).to_be_visible()


def test_click_new_chat_clears_messages(chat_page: Page, test_message: str):
    """Test that clicking 'New Chat' clears the message area."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send a message
    input_field.fill(test_message)
    send_button.click()
    chat_page.wait_for_timeout(2000)

    # Message should be visible in chat area (not sidebar)
    user_msg = chat_page.locator('main .whitespace-pre-wrap').first
    expect(user_msg).to_be_visible()

    # Click "New Chat" button (the primary button, not session items)
    new_chat_btn = chat_page.locator('button.btn-primary:has-text("New Chat")')
    new_chat_btn.click()

    # Welcome message should reappear
    welcome = chat_page.locator('h2:has-text("How can I help you today?")')
    expect(welcome).to_be_visible()

    # Previous messages should be gone
    expect(user_msg).to_be_hidden()


def test_load_previous_session(chat_page: Page):
    """Test loading a previous session from the sidebar."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send first message in first session
    first_message = "First session message"
    input_field.fill(first_message)
    send_button.click()
    chat_page.wait_for_timeout(3000)

    # Start new chat (use btn-primary to target the main button)
    new_chat_btn = chat_page.locator('button.btn-primary:has-text("New Chat")')
    new_chat_btn.click()
    chat_page.wait_for_timeout(500)

    # Send message in second session
    second_message = "Second session message"
    input_field.fill(second_message)
    send_button.click()
    chat_page.wait_for_timeout(3000)

    # Now click on the first session in sidebar (use .first in case of duplicates)
    first_session_btn = chat_page.locator(f'aside button:has-text("{first_message[:15]}")').first
    first_session_btn.click()
    chat_page.wait_for_timeout(1000)

    # First message should be visible again in main chat area
    first_msg_content = chat_page.locator(f'main .whitespace-pre-wrap:has-text("{first_message}")').first
    expect(first_msg_content).to_be_visible()

    # Second message should not be visible in main chat area
    second_msg_content = chat_page.locator(f'main .whitespace-pre-wrap:has-text("{second_message}")')
    expect(second_msg_content).to_be_hidden()


def test_active_session_highlighted(chat_page: Page, test_message: str):
    """Test that the active session is highlighted in the sidebar."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send a message to create session
    input_field.fill(test_message)
    send_button.click()
    chat_page.wait_for_timeout(3000)

    # The session button should have the 'btn-active' class
    active_session = chat_page.locator('aside button.btn-active')
    expect(active_session).to_be_visible()


def test_session_persists_across_multiple_messages(chat_page: Page):
    """Test that a session maintains all messages."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    messages = ["Message one", "Message two", "Message three"]

    # Send multiple messages
    for msg in messages:
        input_field.fill(msg)
        send_button.click()
        chat_page.wait_for_timeout(2000)

    # All user messages should be visible in main chat area
    for msg in messages:
        user_msg = chat_page.locator(f'main .whitespace-pre-wrap:has-text("{msg}")').first
        expect(user_msg).to_be_visible()


def test_session_title_based_on_first_message(chat_page: Page):
    """Test that session title is derived from the first message."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    first_message = "This is my unique test message for titling"

    # Send message
    input_field.fill(first_message)
    send_button.click()
    chat_page.wait_for_timeout(3000)

    # Session in sidebar should contain part of the first message (use .first in case of duplicates)
    session_with_title = chat_page.locator(f'aside button:has-text("{first_message[:20]}")').first
    expect(session_with_title).to_be_visible()


def test_multiple_sessions_listed(chat_page: Page):
    """Test that multiple sessions are listed in the sidebar."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')
    new_chat_btn = chat_page.locator('button.btn-primary:has-text("New Chat")')

    # Create 3 sessions
    for i in range(3):
        input_field.fill(f"Session {i + 1} message")
        send_button.click()
        chat_page.wait_for_timeout(2000)

        # Start new chat for next iteration
        if i < 2:  # Don't create a 4th session
            new_chat_btn.click()
            chat_page.wait_for_timeout(500)

    # Should have at least 3 session buttons
    session_buttons = chat_page.locator('aside button.btn-ghost')
    assert session_buttons.count() >= 3


def test_session_list_scrollable(chat_page: Page):
    """Test that session list is scrollable."""
    # Check that sessions container has overflow-y-auto class
    sessions_container = chat_page.locator('aside .flex-1.overflow-y-auto')
    expect(sessions_container).to_be_visible()
