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


def test_web_requirements_pin_the_pydantic_v2_api_contract() -> None:
    requirements = Path("requirements-web.txt").read_text(encoding="utf-8").splitlines()
    declared = [line.split("#", maxsplit=1)[0].strip() for line in requirements]
    assert "pydantic>=2,<3" in declared


def test_restore_state_validates_complete_persisted_chat_shape() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for guard in (
        "function isUuid",
        "function normalizePersistedMessage",
        "function normalizeAssistantPayload",
        "function normalizeProducts",
        "function normalizeProduct",
        "function normalizeRecommendations",
        "function isPlainObject",
    ):
        assert guard in script
    assert "saved.sessionId" in script
    assert "return emptyState();" in script
    assert "payload.products || {}" in script


def test_restore_state_normalizes_sparse_product_summaries() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "if (!isPlainObject(value)) {" in script
    assert 'title: typeof value.title === "string" ? value.title : "Untitled product",' in script
    assert "parent_asin: asin," in script
    assert "value.parent_asin !== asin" not in script
    assert 'const store = typeof value.store === "string" ? value.store : "";' in script
    assert "const categories = normalizeStringList(value.categories) || [];" in script
    assert "const features = normalizeStringList(value.features) || [];" in script
    assert "price: normalizeOptionalNumber(value.price)" in script
    assert "average_rating: normalizeOptionalNumber(value.average_rating)" in script
    assert "rating_number: normalizeOptionalNumber(value.rating_number)" in script


def test_async_recovery_has_static_safety_guards() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "let serviceReady = false;" in script
    assert (
        "newChatButton.disabled = !serviceReady || state.pending || runtimeState.applying;"
        in script
    )
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


def test_product_detail_renders_complete_commerce_fields() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function appendProductCommerceDetails" in script
    assert "appendProductCommerceDetails(product)" in script
    assert '"product-price"' in script
    assert '"product-rating"' in script
    assert '"product-store"' in script
    assert "product.rating_number !== null" in script
    assert "product.rating_number !== undefined" in script


def test_drawer_contains_keyboard_focus_and_invalidates_stale_requests() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "background.inert = true" in script
    assert "background.inert = false" in script
    assert 'event.key === "Tab"' in script
    assert "drawer.contains(document.activeElement)" in script
    assert "let drawerRequestGeneration = 0;" in script
    assert "const requestGeneration = ++drawerRequestGeneration;" in script
    assert "requestGeneration !== drawerRequestGeneration || drawer.hidden" in script


def test_main_bootstrap_is_not_accidentally_nested_in_a_duplicate_function() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert script.count("function productRatingText(product) {") == 1


def test_history_renders_each_conversation_once() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    render_history = script.split("function renderHistory() {", maxsplit=1)[1].split(
        "function openHistory() {", maxsplit=1
    )[0]

    assert render_history.count("historyList.append(item);") == 1


def test_v2_history_is_normalized_before_becoming_renderable_state() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    restore = script.split("function restoreActiveConversation() {", maxsplit=1)[1].split(
        "function showNotice", maxsplit=1
    )[0]

    assert "function normalizeStoredConversation" in script
    assert ".map(normalizeStoredConversation)" in restore
    assert ".filter((item) => item !== null)" in restore


def test_runtime_selector_disables_unconfigured_backends() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "option.disabled = isPlainObject(meta) && meta.configured === false;" in script


def test_runtime_config_cannot_switch_during_an_inflight_message() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    apply_config = script.split("async function applyConfig", maxsplit=1)[1].split(
        "async function loadRuntime", maxsplit=1
    )[0]

    assert "if (state.pending)" in apply_config
    assert "Finish the current message before changing configuration." in apply_config
    assert "configApply.disabled = state.pending || runtimeState.applying;" in script
    assert "state.pending || runtimeState.applying || !text.trim()" in script
    assert "await newChat({allowDuringConfig: true});" in apply_config
    assert "historyButton.disabled = runtimeState.applying;" in script
    assert "function openHistory() {\n  if (runtimeState.applying)" in script
    assert "function deleteConversation(conversationId) {\n  if (runtimeState.applying)" in script


def test_historical_usage_keeps_server_attributed_cost() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "estimated_cost_usd: estimatedCost ?? null" in script
    assert 'typeof usage.estimated_cost_usd === "number"' in script
    assert "estimatedCost !== null" in script
