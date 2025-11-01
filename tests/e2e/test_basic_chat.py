"""E2E tests for basic chat functionality."""
import pytest
from playwright.sync_api import Page, expect


def test_page_loads(chat_page: Page):
    """Test that the chat page loads successfully."""
    # Check title
    expect(chat_page).to_have_title("Chat - AI Assistant")

    # Check main heading
    heading = chat_page.locator("h1:has-text('AI Assistant')")
    expect(heading).to_be_visible()


def test_welcome_message_displayed(chat_page: Page):
    """Test that welcome message is shown when no messages exist."""
    # Check for welcome heading
    welcome = chat_page.locator("h2:has-text('How can I help you today?')")
    expect(welcome).to_be_visible()

    # Check for sparkles icon (lucide renders as svg)
    sparkles_icon = chat_page.locator('[data-lucide="sparkles"]')
    expect(sparkles_icon).to_be_visible()


def test_new_chat_button_visible(chat_page: Page):
    """Test that 'New Chat' button is visible."""
    new_chat_btn = chat_page.locator('button.btn-primary:has-text("New Chat")')
    expect(new_chat_btn).to_be_visible()


def test_input_field_exists(chat_page: Page):
    """Test that message input field exists and is ready."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    expect(input_field).to_be_visible()
    expect(input_field).to_be_enabled()


def test_send_button_disabled_when_empty(chat_page: Page):
    """Test that send button is disabled when input is empty."""
    send_button = chat_page.locator('button[type="submit"]')
    expect(send_button).to_be_disabled()


def test_send_button_enabled_with_text(chat_page: Page):
    """Test that send button becomes enabled when text is entered."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Type message
    input_field.fill("Test message")

    # Button should now be enabled
    expect(send_button).to_be_enabled()


def test_send_message_and_receive_response(chat_page: Page, test_message: str):
    """Test sending a message and receiving a response."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    # Send message
    input_field.fill(test_message)
    send_button.click()

    # Wait for user message to appear
    user_message = chat_page.locator(f'text="{test_message}"').first
    expect(user_message).to_be_visible(timeout=5000)

    # Wait for assistant response (look for bot icon or assistant message)
    # The assistant response should appear within 30 seconds
    assistant_message = chat_page.locator('[data-lucide="bot"]').nth(1)  # Second bot icon (first is header)
    expect(assistant_message).to_be_visible(timeout=30000)

    # Verify input was cleared
    expect(input_field).to_have_value("")


def test_welcome_message_disappears_after_chat(chat_page: Page, test_message: str):
    """Test that welcome message disappears after first message."""
    welcome = chat_page.locator("h2:has-text('How can I help you today?')")

    # Welcome should be visible initially
    expect(welcome).to_be_visible()

    # Send a message
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')
    input_field.fill(test_message)
    send_button.click()

    # Wait for response
    chat_page.wait_for_timeout(2000)

    # Welcome should now be hidden
    expect(welcome).to_be_hidden()


def test_character_counter_updates(chat_page: Page):
    """Test that character counter updates as user types."""
    input_field = chat_page.locator('input[placeholder*="Message"]')

    # Type message
    test_text = "Hello World"
    input_field.fill(test_text)

    # Check character counter
    counter = chat_page.locator(f'text="{len(test_text)}/2000"')
    expect(counter).to_be_visible()


def test_multiple_messages_in_sequence(chat_page: Page):
    """Test sending multiple messages in sequence."""
    input_field = chat_page.locator('input[placeholder*="Message"]')
    send_button = chat_page.locator('button[type="submit"]')

    messages = ["First message", "Second message", "Third message"]

    for msg in messages:
        input_field.fill(msg)
        send_button.click()

        # Wait for message to appear
        user_msg = chat_page.locator(f'text="{msg}"').first
        expect(user_msg).to_be_visible(timeout=5000)

        # Wait a bit before next message
        chat_page.wait_for_timeout(1000)


def test_theme_toggle_button_exists(chat_page: Page):
    """Test that theme toggle button exists."""
    # Look for theme toggle button (moon or sun icon)
    theme_button = chat_page.locator('button[title="Toggle theme"]')
    expect(theme_button).to_be_visible()


def test_phase_badge_displayed(chat_page: Page):
    """Test that Phase 3 badge is displayed."""
    phase_badge = chat_page.locator('text="Phase 3"')
    expect(phase_badge).to_be_visible()


def test_model_badge_displayed(chat_page: Page):
    """Test that model badge is displayed."""
    # The badge should show the chat model name
    model_badge = chat_page.locator('.badge:has-text("gemma")')
    expect(model_badge).to_be_visible()
