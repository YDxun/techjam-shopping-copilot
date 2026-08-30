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


def test_product_drawer_and_ordered_renderer_contract() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="drawer-close"' in html
    assert 'id="drawer-title"' in html
    assert "function renderProducts" in script
    assert "function openProductDrawer" in script
    assert "function closeProductDrawer" in script
    assert "function retryMessage" in script
    assert "payload.agent_response.recommendations" in script
    assert "Object.values(payload.products)" not in script
    assert "innerHTML" not in script


def test_product_ux_uses_safe_fixed_copy_and_local_detail_endpoint() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '"Understanding your request and searching products..."' in script
    assert '"Tell me another preference and I’ll refine the search."' in script
    assert '"The local service restarted. Starting a new chat."' in script
    assert '"Something went wrong. Please retry this message."' in script
    assert '"Product details are unavailable for this recommendation."' in script
    assert "`/api/products/${encodeURIComponent(asin)}`" in script
    assert "drawerClose.focus()" in script
    assert "focusBeforeDrawer.focus()" in script


def test_product_cards_and_drawer_have_responsive_static_styles() -> None:
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in styles
    assert "width: min(420px, 100vw);" in styles
    assert "@media (max-width: 760px)" in styles
    assert ".product-grid" in styles
    assert "grid-template-columns: 1fr;" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert "transition: none !important;" in styles
