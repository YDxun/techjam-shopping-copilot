from pathlib import Path

STATIC = Path("webapp/static")


def test_static_page_has_required_landmarks_and_local_assets() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'lang="en"' in html
    assert 'id="new-chat"' in html
    assert 'id="conversation"' in html
    assert 'id="message-input"' in html
    assert 'id="send-message"' in html
    assert 'id="product-drawer"' in html
    assert 'href="/assets/styles.css"' in html
    assert 'src="/assets/app.js"' in html
    assert "https://" not in html and "http://" not in html


def test_dynamic_javascript_never_uses_html_injection() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert "eval(" not in script
    assert "localStorage" in script
    assert "crypto.randomUUID()" in script


def test_async_recovery_has_static_safety_guards() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "let serviceReady = false;" in script
    assert "newChatButton.disabled = !serviceReady || state.pending;" in script
    assert "handleUnexpectedInteractionError" in script
    assert "void sendMessage(text);" not in script
    assert "void newChat();" not in script
