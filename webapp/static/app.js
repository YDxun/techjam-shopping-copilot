"use strict";

const STORAGE_KEY = "shopping-copilot-web:v1";
const CONVERSATIONS_KEY = "shopping-copilot-web:conversations:v2";
const ACTIVE_CONVERSATION_KEY = "shopping-copilot-web:active:v2";
const PROMPT_EXAMPLES = [
  "I need a lightweight jacket for hiking.",
  "Find me comfortable black shoes under $80.",
  "I'm looking for a cotton shirt, but I'm still exploring.",
];
const initialState = {
  sessionId: null,
  messages: [],
  pending: false,
  updatedAt: null,
};
const emptyState = () => ({...initialState, messages: []});
let state = emptyState();
let conversations = [];
let activeConversationId = null;
let serviceReady = false;

const newChatButton = document.querySelector("#new-chat");
const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const promptExamples = document.querySelector("#prompt-examples");
const serviceStatus = document.querySelector("#service-status");
const statusNotice = document.querySelector("#status-notice");
const composer = document.querySelector("#composer");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-message");
const drawer = document.getElementById("product-drawer");
const backdrop = document.getElementById("drawer-backdrop");
const drawerTitle = document.getElementById("drawer-title");
const drawerBody = document.getElementById("drawer-body");
const drawerClose = document.getElementById("drawer-close");
const drawerBackground = document.querySelectorAll(".sidebar, .app-main");
let focusBeforeDrawer = null;
let drawerRequestGeneration = 0;

class ApiError extends Error {
  constructor(status, code) {
    super(code);
    this.status = status;
    this.code = code;
  }
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(response.status, payload.error?.code || "request_failed");
  }
  return payload;
}

function currentConversation() {
  return conversations.find((item) => item.id === activeConversationId) || null;
}

function saveCurrentConversation() {
  const conversation = currentConversation();
  if (!conversation) {
    return;
  }
  state.updatedAt = new Date().toISOString();
  conversation.sessionId = state.sessionId;
  conversation.messages = state.messages;
  conversation.updatedAt = state.updatedAt;
  if (!conversation.title) {
    const firstUser = state.messages.find((message) => message.role === "user");
    if (firstUser) {
      conversation.title = firstUser.text.slice(0, 48);
    }
  }
}

function persistConversations() {
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  localStorage.setItem(ACTIVE_CONVERSATION_KEY, JSON.stringify(activeConversationId));
}

function persistState() {
  saveCurrentConversation();
  persistConversations();
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isUuid(value) {
  return typeof value === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

function isSafeNonNegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function normalizeStringList(value) {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? [...value]
    : null;
}

function normalizeRecommendations(value) {
  if (!Array.isArray(value)) {
    return null;
  }
  const recommendations = [];
  for (const recommendation of value) {
    if (!isPlainObject(recommendation) || typeof recommendation.parent_asin !== "string"
      || !recommendation.parent_asin.trim()) {
      return null;
    }
    recommendations.push({parent_asin: recommendation.parent_asin});
  }
  return recommendations;
}

function normalizeOptionalNumber(value) {
  const number = Number(value);
  return value === null || value === undefined || !Number.isFinite(number) ? null : number;
}

function normalizeProduct(asin, value) {
  if (!isPlainObject(value)) {
    return null;
  }
  const store = typeof value.store === "string" ? value.store : "";
  const categories = normalizeStringList(value.categories) || [];
  const features = normalizeStringList(value.features) || [];
  return {
    parent_asin: asin,
    title: typeof value.title === "string" ? value.title : "Untitled product",
    price: normalizeOptionalNumber(value.price),
    average_rating: normalizeOptionalNumber(value.average_rating),
    rating_number: normalizeOptionalNumber(value.rating_number),
    store,
    categories,
    features,
  };
}

function normalizeProducts(value) {
  if (!isPlainObject(value)) {
    return null;
  }
  const products = {};
  for (const [asin, product] of Object.entries(value)) {
    if (!asin.trim()) {
      return null;
    }
    const normalized = normalizeProduct(asin, product);
    if (normalized === null) {
      return null;
    }
    products[asin] = normalized;
  }
  return products;
}

function normalizeUsageSources(value) {
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    return null;
  }
  const sources = [];
  for (const item of value) {
    if (!isPlainObject(item) || typeof item.provider !== "string"
      || typeof item.model !== "string"
      || !isSafeNonNegativeInteger(item.prompt_tokens)
      || !isSafeNonNegativeInteger(item.completion_tokens)
      || typeof item.cost_usd !== "number" || !Number.isFinite(item.cost_usd)
      || item.cost_usd < 0) {
      return null;
    }
    sources.push({
      provider: item.provider,
      model: item.model,
      prompt_tokens: item.prompt_tokens,
      completion_tokens: item.completion_tokens,
      cost_usd: item.cost_usd,
    });
  }
  return sources;
}

function normalizeAssistantPayload(value, sessionId) {
  if (!isPlainObject(value) || value.session_id !== sessionId || !isUuid(value.session_id)
    || !isUuid(value.message_id) || !Number.isSafeInteger(value.turn) || value.turn < 1
    || !isPlainObject(value.agent_response)) {
    return null;
  }
  const response = value.agent_response;
  const recommendations = normalizeRecommendations(response.recommendations);
  const products = normalizeProducts(value.products);
  const usageSources = normalizeUsageSources(response.usage?.sources);
  const estimatedCost = response.usage?.estimated_cost_usd;
  if (typeof response.message !== "string"
    || (response.ask_attribute !== null && typeof response.ask_attribute !== "string")
    || !isPlainObject(response.usage)
    || !isSafeNonNegativeInteger(response.usage.prompt_tokens)
    || !isSafeNonNegativeInteger(response.usage.completion_tokens)
    || usageSources === null
    || (estimatedCost !== undefined && estimatedCost !== null
      && (typeof estimatedCost !== "number"
      || !Number.isFinite(estimatedCost) || estimatedCost < 0))
    || recommendations === null || products === null) {
    return null;
  }
  return {
    session_id: value.session_id,
    message_id: value.message_id,
    turn: value.turn,
    agent_response: {
      message: response.message,
      ask_attribute: response.ask_attribute,
      recommendations,
      usage: {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        estimated_cost_usd: estimatedCost ?? null,
        sources: usageSources,
      },
    },
    products,
  };
}

function normalizePersistedMessage(value, sessionId) {
  if (!isPlainObject(value) || (value.role !== "user" && value.role !== "assistant")) {
    return null;
  }
  if (value.role === "user") {
    if (typeof value.text !== "string" || !value.text.trim() || value.text.length > 4000
      || !isUuid(value.messageId) || !["pending", "sent", "failed"].includes(value.status)) {
      return null;
    }
    return {
      role: "user",
      text: value.text,
      messageId: value.messageId,
      status: value.status === "pending" ? "failed" : value.status,
    };
  }
  const payload = normalizeAssistantPayload(value.payload, sessionId);
  return payload === null ? null : {role: "assistant", payload};
}

function normalizeConversationMessages(saved) {
  if (!isPlainObject(saved) || !isUuid(saved.sessionId) || !Array.isArray(saved.messages)) {
    return null;
  }
  const messages = saved.messages.map((message) => normalizePersistedMessage(message, saved.sessionId));
  const userMessageIds = new Set();
  for (const message of messages) {
    if (message === null) {
      return null;
    }
    if (message.role === "user") {
      if (userMessageIds.has(message.messageId)) {
        return null;
      }
      userMessageIds.add(message.messageId);
    } else if (!userMessageIds.has(message.payload.message_id)) {
      return null;
    }
  }
  return messages;
}

function normalizeStoredConversation(value) {
  if (!isPlainObject(value) || !isUuid(value.id) || !Array.isArray(value.messages)
    || (value.sessionId !== null && !isUuid(value.sessionId))) {
    return null;
  }
  let messages;
  if (value.sessionId === null) {
    if (value.messages.length > 0) {
      return null;
    }
    messages = [];
  } else {
    messages = normalizeConversationMessages(value);
    if (messages === null) {
      return null;
    }
  }
  const now = new Date().toISOString();
  return {
    id: value.id,
    title: typeof value.title === "string" ? value.title.slice(0, 48) : "",
    createdAt: typeof value.createdAt === "string" ? value.createdAt : now,
    updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : now,
    sessionId: value.sessionId,
    messages,
  };
}

function stateFromConversation(conversation) {
  return {
    sessionId: conversation.sessionId,
    messages: Array.isArray(conversation.messages) ? conversation.messages : [],
    pending: false,
    updatedAt: conversation.updatedAt || null,
  };
}

function migrateLegacyState() {
  const legacy = localStorage.getItem(STORAGE_KEY);
  if (!legacy) {
    return null;
  }
  localStorage.removeItem(STORAGE_KEY);
  let saved;
  try {
    saved = JSON.parse(legacy);
  } catch (error) {
    return null;
  }
  const messages = normalizeConversationMessages(saved);
  if (!messages || saved.version !== 1) {
    return null;
  }
  const now = new Date().toISOString();
  const firstUser = messages.find((message) => message.role === "user");
  const conversation = {
    id: crypto.randomUUID(),
    title: firstUser ? firstUser.text.slice(0, 48) : "Previous conversation",
    createdAt: saved.updatedAt || now,
    updatedAt: saved.updatedAt || now,
    sessionId: saved.sessionId,
    messages,
  };
  conversations = [conversation];
  activeConversationId = conversation.id;
  persistConversations();
  return stateFromConversation(conversation);
}

function restoreActiveConversation() {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    if (!raw) {
      const migrated = migrateLegacyState();
      return migrated || emptyState();
    }
    const list = JSON.parse(raw);
    if (!Array.isArray(list)) {
      return emptyState();
    }
    conversations = list
      .map(normalizeStoredConversation)
      .filter((item) => item !== null);
    const activeRaw = JSON.parse(localStorage.getItem(ACTIVE_CONVERSATION_KEY) || "null");
    const active = conversations.find((item) => item.id === activeRaw) || conversations[0] || null;
    if (!active) {
      activeConversationId = null;
      return emptyState();
    }
    activeConversationId = active.id;
    return stateFromConversation(active);
  } catch (error) {
    return emptyState();
  }
}

function showNotice(message) {
  statusNotice.textContent = message;
}

function setComposerEnabled(enabled) {
  const interactionEnabled = enabled && !runtimeState.applying;
  messageInput.disabled = !interactionEnabled;
  sendButton.disabled = !interactionEnabled;
  newChatButton.disabled = !serviceReady || state.pending || runtimeState.applying;
}

function renderPromptExamples() {
  const buttons = PROMPT_EXAMPLES.map((prompt) => {
    const button = document.createElement("button");
    button.className = "prompt-example";
    button.type = "button";
    button.textContent = prompt;
    button.addEventListener("click", () => {
      messageInput.value = prompt;
      messageInput.focus();
    });
    return button;
  });
  promptExamples.replaceChildren(...buttons);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function productPriceText(product) {
  const numericPrice = Number(product?.price);
  return product?.price !== null
    && product?.price !== undefined
    && Number.isFinite(numericPrice)
    ? `$${numericPrice.toFixed(2)}`
    : "Price unavailable";
}

function productRatingText(product) {
  if (product?.average_rating === null || product?.average_rating === undefined) {
    return null;
  }
  const hasCount = product.rating_number !== null && product.rating_number !== undefined;
  const count = hasCount ? ` (${product.rating_number})` : "";
  return `${product.average_rating} stars${count}`;
}

function renderProducts(payload) {
  const grid = element("div", "product-grid");
  if (!isPlainObject(payload?.agent_response) || !Array.isArray(payload.agent_response.recommendations)) {
    return grid;
  }
  const products = isPlainObject(payload.products || {}) ? payload.products || {} : {};
  payload.agent_response.recommendations.forEach((recommendation, index) => {
    if (!isPlainObject(recommendation) || typeof recommendation.parent_asin !== "string") {
      return;
    }
    const asin = recommendation.parent_asin;
    const product = isPlainObject(products[asin]) ? products[asin] : null;
    const card = element("article", "product-card");
    card.append(element(
      "div",
      "product-visual",
      product?.categories?.at(-1) || "Product",
    ));
    card.append(element("span", "product-rank", `#${index + 1}`));
    card.append(element(
      "h3",
      "product-title",
      product?.title || "Product details unavailable",
    ));
    card.append(element("p", "product-asin", asin));
    card.append(element("p", "product-price", productPriceText(product)));
    const rating = productRatingText(product);
    if (rating !== null) {
      card.append(element("p", "product-rating", rating));
    }
    if (product?.store) {
      card.append(element("p", "product-store", product.store));
    }
    const badges = element("div", "category-badges");
    (product?.categories || []).slice(0, 2).forEach((category) => {
      badges.append(element("span", "category-badge", category));
    });
    card.append(badges);
    const features = element("ul", "product-features");
    (product?.features || []).slice(0, 2).forEach((feature) => {
      features.append(element("li", "", feature));
    });
    card.append(features);
    const button = element("button", "product-details-button", "View details");
    button.type = "button";
    button.dataset.asin = asin;
    button.addEventListener("click", () => openProductDrawer(asin));
    card.append(button);
    grid.append(card);
  });
  return grid;
}

function appendDetailList(label, values) {
  if (!Array.isArray(values) || values.length === 0) {
    return;
  }
  drawerBody.append(element("h3", "", label));
  const list = element("ul", "detail-list");
  values.forEach((value) => list.append(element("li", "", String(value))));
  drawerBody.append(list);
}

function appendProductCommerceDetails(product) {
  drawerBody.append(element("p", "product-price", productPriceText(product)));
  const rating = productRatingText(product);
  if (rating !== null) {
    drawerBody.append(element("p", "product-rating", rating));
  }
  if (product.store) {
    drawerBody.append(element("p", "product-store", product.store));
  }
}

function renderProductDetail(product) {
  drawerBody.replaceChildren();
  drawerTitle.textContent = product.title || "Product details";
  drawerBody.append(element("p", "product-asin", product.parent_asin));
  appendProductCommerceDetails(product);
  appendDetailList("Categories", product.categories);
  appendDetailList("Features", product.features);
  appendDetailList("Description", product.description);
  if (product.details && typeof product.details === "object") {
    drawerBody.append(element("h3", "", "Specifications"));
    const list = element("dl", "detail-pairs");
    Object.entries(product.details).forEach(([key, value]) => {
      list.append(element("dt", "", key));
      list.append(element(
        "dd",
        "",
        typeof value === "string" ? value : JSON.stringify(value),
      ));
    });
    drawerBody.append(list);
  }
}

function disableDrawerBackground() {
  drawerBackground.forEach((background) => {
    background.inert = true;
  });
}

function enableDrawerBackground() {
  drawerBackground.forEach((background) => {
    background.inert = false;
  });
}

function containDrawerFocus(event) {
  const focusable = Array.from(drawer.querySelectorAll(
    "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), "
      + "textarea:not([disabled]), [tabindex]:not([tabindex=\"-1\"])",
  ));
  if (focusable.length === 0) {
    event.preventDefault();
    drawerClose.focus();
    return;
  }
  const first = focusable[0];
  const last = focusable.at(-1);
  const focusIsInside = drawer.contains(document.activeElement);
  if (event.shiftKey && (!focusIsInside || document.activeElement === first)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (!focusIsInside || document.activeElement === last)) {
    event.preventDefault();
    first.focus();
  }
}

async function openProductDrawer(asin) {
  const requestGeneration = ++drawerRequestGeneration;
  if (drawer.hidden) {
    focusBeforeDrawer = document.activeElement;
  }
  drawer.hidden = false;
  backdrop.hidden = false;
  drawer.setAttribute("aria-hidden", "false");
  disableDrawerBackground();
  drawerTitle.textContent = "Loading product details…";
  drawerBody.replaceChildren();
  drawerClose.focus();
  try {
    const product = await apiRequest(`/api/products/${encodeURIComponent(asin)}`);
    if (requestGeneration !== drawerRequestGeneration || drawer.hidden) {
      return;
    }
    renderProductDetail(product);
  } catch (error) {
    if (requestGeneration !== drawerRequestGeneration || drawer.hidden) {
      return;
    }
    drawerTitle.textContent = "Product details unavailable";
    drawerBody.textContent = "Product details are unavailable for this recommendation.";
  }
}

function closeProductDrawer() {
  drawerRequestGeneration += 1;
  drawer.hidden = true;
  backdrop.hidden = true;
  drawer.setAttribute("aria-hidden", "true");
  enableDrawerBackground();
  if (focusBeforeDrawer instanceof HTMLElement) {
    focusBeforeDrawer.focus();
  }
  focusBeforeDrawer = null;
}

function renderUserMessage(message) {
  const row = document.createElement("article");
  row.className = "message user-message";

  const text = document.createElement("p");
  text.textContent = typeof message.text === "string" ? message.text : "";
  row.append(text);

  if (message.status === "failed") {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const retry = document.createElement("button");
    retry.className = "retry-message";
    retry.type = "button";
    retry.textContent = "Retry";
    retry.disabled = state.pending || runtimeState.applying;
    retry.addEventListener("click", () => {
      const submission = retryMessage(message.messageId);
      if (submission) {
        submission.catch(handleUnexpectedInteractionError);
      }
    });
    actions.append(retry);
    row.append(actions);
  }
  return row;
}

function renderAssistantMessage(message) {
  const row = document.createElement("article");
  row.className = "message assistant-message";
  const text = document.createElement("p");
  const responseMessage = message.payload?.agent_response?.message;
  text.textContent = typeof responseMessage === "string" ? responseMessage : "";
  row.append(text);
  const usage = message.payload?.agent_response?.usage;
  if (usage && (usage.prompt_tokens > 0 || usage.completion_tokens > 0)) {
    const usageNote = element("p", "usage-note", "");
    const badge = document.createElement("span");
    badge.className = "online-badge";
    badge.textContent = "online";
    usageNote.append(badge);
    const totalTokens = usage.prompt_tokens + usage.completion_tokens;
    const cost = typeof usage.estimated_cost_usd === "number"
      ? usage.estimated_cost_usd
      : null;
    usageNote.append(document.createTextNode(
      cost !== null
        ? `${totalTokens} tokens ≈ $${cost.toFixed(6)} (paid API call)`
        : `${totalTokens} tokens (paid API call)`,
    ));
    row.append(usageNote);
  }
  const recommendations = message.payload?.agent_response?.recommendations;
  if (Array.isArray(recommendations) && recommendations.length > 0) {
    row.append(renderProducts(message.payload));
  } else {
    row.append(element(
      "p",
      "empty-recommendations",
      "Tell me another preference and I’ll refine the search.",
    ));
  }
  return row;
}

function renderLoadingMessage() {
  const row = element("article", "message assistant-message loading-message");
  row.setAttribute("role", "status");
  row.append(element("p", "", "Understanding your request and searching products..."));
  return row;
}

function renderConversation() {
  const rows = state.messages.map((message) => (
    message.role === "user" ? renderUserMessage(message) : renderAssistantMessage(message)
  ));
  if (state.pending) {
    rows.push(renderLoadingMessage());
  }
  conversation.replaceChildren(...rows);
  conversation.setAttribute("aria-busy", String(state.pending));
  welcome.hidden = state.messages.length > 0;
  setComposerEnabled(Boolean(state.sessionId) && !state.pending);
  configApply.disabled = state.pending || runtimeState.applying;
  configReset.disabled = state.pending || runtimeState.applying;
  historyButton.disabled = runtimeState.applying;
}

async function replaceExpiredSession() {
  const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
  saveCurrentConversation(); // keep the expired conversation visible in history
  const conversation = createConversation(created.session_id);
  conversations.push(conversation);
  activeConversationId = conversation.id;
  state = emptyState();
  state.sessionId = created.session_id;
  persistConversations();
  showNotice("The local service restarted. Starting a new chat.");
}

async function submitExistingMessage(text, messageId) {
  state.pending = true;
  const userMessage = state.messages.find((item) => item.messageId === messageId);
  if (!userMessage) {
    state.pending = false;
    return;
  }
  userMessage.status = "pending";
  persistState();
  renderConversation();
  showNotice("");

  let finalNotice = "";
  try {
    const payload = normalizeAssistantPayload(await apiRequest(
      `/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({message_id: messageId, message: text}),
      },
    ), state.sessionId);
    if (payload === null) {
      throw new Error("invalid chat response");
    }
    userMessage.status = "sent";
    state.messages.push({role: "assistant", payload});
  } catch (error) {
    if (error instanceof ApiError && error.code === "session_not_found") {
      try {
        await replaceExpiredSession();
        finalNotice = "The local service restarted. Starting a new chat.";
      } catch (replacementError) {
        userMessage.status = "failed";
        finalNotice = "Something went wrong. Please retry this message.";
      }
    } else {
      userMessage.status = "failed";
      finalNotice = "Something went wrong. Please retry this message.";
    }
  } finally {
    state.pending = false;
    persistState();
    renderConversation();
    showNotice(finalNotice);
  }
}

async function sendMessage(text) {
  if (runtimeState.applying) {
    return;
  }
  const messageId = crypto.randomUUID();
  state.messages.push({role: "user", text, messageId, status: "pending"});
  return submitExistingMessage(text, messageId);
}

function retryMessage(messageId) {
  const failed = state.messages.find(
    (item) => item.role === "user" && item.messageId === messageId,
  );
  if (!failed || state.pending || runtimeState.applying) {
    return;
  }
  failed.status = "pending";
  return submitExistingMessage(failed.text, failed.messageId);
}

async function newChat({allowDuringConfig = false} = {}) {
  if (state.pending || !serviceReady || (runtimeState.applying && !allowDuringConfig)) {
    return;
  }
  state.pending = true;
  saveCurrentConversation(); // snapshot the current conversation into history
  renderConversation();
  showNotice("Starting a new chat...");
  try {
    const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
    const conversation = createConversation(created.session_id);
    conversations.push(conversation);
    activeConversationId = conversation.id;
    state = emptyState();
    state.sessionId = created.session_id;
    persistConversations();
    showNotice("");
  } catch (error) {
    state.pending = false;
    showNotice("A new chat could not be started.");
  } finally {
    state.pending = false;
    renderConversation();
    messageInput.focus();
  }
}

const wait = (milliseconds) => new Promise((resolve) => {
  window.setTimeout(resolve, milliseconds);
});

async function waitForService() {
  while (true) {
    let health;
    try {
      health = await apiRequest("/api/health");
    } catch (error) {
      await wait(500);
      continue;
    }
    if (health.status === "ready") {
      return;
    }
    if (health.status === "failed") {
      throw new ApiError(503, "initialization_failed");
    }
    await wait(500);
  }
}

function renderInitializationFailure() {
  serviceReady = false;
  serviceStatus.textContent = "Local · Unavailable";
  welcome.hidden = true;
  const title = document.createElement("h2");
  title.textContent = "Shopping Copilot could not start";
  const detail = document.createElement("p");
  detail.textContent = "Check the local catalog and service configuration, then restart the app.";
  conversation.replaceChildren(title, detail);
  composer.hidden = true;
  showNotice("");
}

async function bootstrap() {
  serviceStatus.textContent = "Local · Loading";
  setComposerEnabled(false);
  renderPromptExamples();

  try {
    await waitForService();
  } catch (error) {
    renderInitializationFailure();
    return;
  }

  serviceReady = true;
  serviceStatus.textContent = "Local · Ready";
  composer.hidden = false;
  try {
    await loadRuntime();
  } catch (error) {
    setConfigStatus("Runtime panel unavailable.");
  }
  state = restoreActiveConversation();
  let startupNotice = "";
  if (state.sessionId) {
    try {
      await apiRequest(`/api/sessions/${state.sessionId}`);
    } catch (error) {
      if (error instanceof ApiError && error.code === "session_not_found") {
        try {
          await replaceExpiredSession();
          startupNotice = "The local service restarted. Starting a new chat.";
        } catch (replacementError) {
          state = emptyState();
          startupNotice = "A chat could not be started. Select New chat to try again.";
        }
      } else {
        startupNotice = "The chat could not be restored. Select New chat to start over.";
      }
    }
  } else {
    try {
      const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
      const conversation = createConversation(created.session_id);
      conversations.push(conversation);
      activeConversationId = conversation.id;
      state.sessionId = created.session_id;
    } catch (error) {
      state = emptyState();
      startupNotice = "A chat could not be started. Select New chat to try again.";
    }
  }
  persistState();
  renderConversation();
  showNotice(startupNotice);
}

function handleUnexpectedInteractionError() {
  state.pending = false;
  const pendingMessage = state.messages.find(
    (message) => message.role === "user" && message.status === "pending",
  );
  if (pendingMessage) {
    pendingMessage.status = "failed";
  }
  persistState();
  renderConversation();
  showNotice("Something went wrong. Please retry this message.");
}


/* ------------------------------------------------------------------ */
/* Runtime configuration panel (environment-adaptive engine selector)  */
/* ------------------------------------------------------------------ */
const runtimeState = {
  info: null,
  applying: false,
  pendingConfig: null,
};

const configProvider = document.querySelector("#config-provider");
const configModel = document.querySelector("#config-model");
const configApiKey = document.querySelector("#config-api-key");
const configRerank = document.querySelector("#config-rerank");
const configRetrieval = document.querySelector("#config-retrieval");
const configOutput = document.querySelector("#config-output");
const toggleLlmIntent = document.querySelector("#toggle-llm-intent");
const toggleFingerprint = document.querySelector("#toggle-fingerprint");
const toggleCategory = document.querySelector("#toggle-category");
const toggleParaphrase = document.querySelector("#toggle-paraphrase");
const configApply = document.querySelector("#config-apply");
const configReset = document.querySelector("#config-reset");
const configAuto = document.querySelector("#config-auto");
const configStatus = document.querySelector("#config-status");
const navChat = document.querySelector("#nav-chat");
const navDashboard = document.querySelector("#nav-dashboard");
const viewChat = document.querySelector("#view-chat");
const viewDashboard = document.querySelector("#view-dashboard");
const dashboardFrame = document.querySelector("#dashboard-frame");
const onlineConfirm = document.querySelector("#online-confirm");
const onlineConfirmBackdrop = document.querySelector("#online-confirm-backdrop");
const onlineConfirmText = document.querySelector("#online-confirm-text");
const onlineConfirmOk = document.querySelector("#online-confirm-ok");
const onlineConfirmCancel = document.querySelector("#online-confirm-cancel");

function setConfigStatus(message) {
  configStatus.textContent = message || "";
  configStatus.hidden = !message;
}

function providerLabel(provider) {
  const profile = runtimeState.info?.providers?.[provider];
  return isPlainObject(profile) && profile.label ? profile.label : (provider === "none" ? "Off (rule-based)" : provider);
}

function isOnlineConfig(config) {
  const provider = (config.llm_provider || "none").toLowerCase();
  if (provider !== "none" && provider !== "") {
    return true;
  }
  const rerank = (config.rerank_backend || "none").toLowerCase();
  return rerank === "auto" || rerank === "text" || rerank === "chat" || Boolean(config.llm_intent_enabled);
}

function populateSelect(select, options, selectedValue) {
  select.replaceChildren();
  for (const [value, meta] of Object.entries(options)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = isPlainObject(meta) && meta.label ? meta.label : value;
    option.disabled = isPlainObject(meta) && meta.configured === false;
    if (value === selectedValue) {
      option.selected = true;
    }
    select.append(option);
  }
}

function populateModelOptions(provider) {
  configModel.replaceChildren();
  const profile = runtimeState.info?.providers?.[provider];
  if (!isPlainObject(profile)) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "\u2014";
    configModel.append(option);
    configModel.disabled = true;
    return;
  }
  configModel.disabled = false;
  const models = Array.isArray(profile.models) && profile.models.length > 0 ? profile.models : [];
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    configModel.append(option);
  }
}

function renderLutBanner(info) {
  // Show the one-click Auto (LUT) apply button only when a mapped recommendation exists.
  configAuto.hidden = !(info && info.lut_recommendation && isPlainObject(info.lut_config));
}

async function applyRecommendedConfig() {
  const info = runtimeState.info;
  const lut = isPlainObject(info?.lut_config) ? info.lut_config : null;
  if (!lut) {
    return;
  }
  // Fill the panel with the recommended engine fields, then apply.
  if (typeof lut.retrieval_backend === "string" && [...configRetrieval.options].some((o) => o.value === lut.retrieval_backend)) {
    configRetrieval.value = lut.retrieval_backend;
  }
  if (typeof lut.rerank_backend === "string" && [...configRerank.options].some((o) => o.value === lut.rerank_backend)) {
    configRerank.value = lut.rerank_backend;
  }
  if (typeof lut.output_strategy === "string" && [...configOutput.options].some((o) => o.value === lut.output_strategy)) {
    configOutput.value = lut.output_strategy;
  }
  if (typeof lut.llm_intent_enabled === "boolean") {
    toggleLlmIntent.checked = lut.llm_intent_enabled;
  }
  await applyConfig(collectConfig());
}

function showView(name) {
  const isChat = name === "chat";
  viewChat.hidden = !isChat;
  viewDashboard.hidden = isChat;
  navChat.classList.toggle("active", isChat);
  navDashboard.classList.toggle("active", !isChat);
  if (!isChat && dashboardFrame) {
    dashboardFrame.src = "/dashboard/"; // reload so live metrics reflect the latest data
  }
}

function renderConfigPanel(info) {
  runtimeState.info = info;
  renderLutBanner(info);
  const active = isPlainObject(info.active) ? info.active : {};
  const provider = active.provider && active.provider !== "none" ? active.provider : "none";
  const providers = {...(isPlainObject(info.providers) ? info.providers : {}), none: {label: "Off (rule-based)"}};
  populateSelect(configProvider, providers, provider);
  populateModelOptions(provider);
  if (typeof active.model === "string" && [...configModel.options].some((o) => o.value === active.model)) {
    configModel.value = active.model;
  }
  populateSelect(configRerank, isPlainObject(info.rerank_backends) ? info.rerank_backends : {}, active.rerank_backend || "none");
  populateSelect(configRetrieval, isPlainObject(info.retrieval_backends) ? info.retrieval_backends : {}, active.retrieval_backend || "auto");
  populateSelect(configOutput, isPlainObject(info.output_strategies) ? info.output_strategies : {}, active.output_strategy || "holdback");
  toggleLlmIntent.checked = Boolean(active.llm_intent_enabled);
  toggleFingerprint.checked = Boolean(active.fingerprint_enabled);
  toggleCategory.checked = Boolean(active.category_expand_enabled);
  toggleParaphrase.checked = Boolean(active.paraphrase_enabled);
  configApiKey.value = ""; // never restore keys into the DOM
  if (active.offline && active.qwen_api_key_set
    && ["text", "auto"].includes(active.rerank_backend)) {
    setConfigStatus("qwen rerank is configured but unavailable; offline fallback is active.");
  } else if (active.offline) {
    setConfigStatus("Offline default \u00b7 zero cost.");
  } else if (active.qwen_api_key_set && provider === "none") {
    setConfigStatus("Online qwen rerank active (DASHSCOPE_API_KEY from server environment).");
  } else {
    setConfigStatus("Online LLM active (key kept in memory on the server).");
  }
}

function collectConfig() {
  const provider = configProvider.value;
  return {
    llm_provider: provider,
    llm_model: provider === "none" ? "" : configModel.value,
    api_key: configApiKey.value.trim(),
    rerank_backend: configRerank.value,
    retrieval_backend: configRetrieval.value,
    output_strategy: configOutput.value,
    llm_intent_enabled: toggleLlmIntent.checked,
    fingerprint: toggleFingerprint.checked,
    category_expand: toggleCategory.checked,
    paraphrase: toggleParaphrase.checked,
  };
}

function closeOnlineConfirm() {
  onlineConfirm.hidden = true;
  onlineConfirmBackdrop.hidden = true;
  onlineConfirm.setAttribute("aria-hidden", "true");
  runtimeState.pendingConfig = null;
}

function showOnlineConfirm(config) {
  const parts = [];
  if (config.llm_provider && config.llm_provider !== "none") {
    parts.push(`${providerLabel(config.llm_provider)} ${config.llm_model || ""}`.trim());
  }
  if (config.rerank_backend && config.rerank_backend !== "none") {
    const meta = runtimeState.info?.rerank_backends?.[config.rerank_backend];
    parts.push(`semantic rerank (${isPlainObject(meta) ? meta.label : config.rerank_backend})`);
  }
  if (config.llm_intent_enabled) {
    parts.push("LLM intent recognition");
  }
  onlineConfirmText.textContent = parts.length > 0
    ? `This enables online AI features: ${parts.join(", ")}. Online calls use your API key and may incur cost; without a valid key the engine automatically falls back to offline rules.`
    : "This configuration enables online AI features.";
  onlineConfirm.hidden = false;
  onlineConfirmBackdrop.hidden = false;
  onlineConfirm.setAttribute("aria-hidden", "false");
  onlineConfirmOk.focus();
}

async function applyConfig(config, {confirmedOnline = false} = {}) {
  if (runtimeState.applying) {
    return;
  }
  if (state.pending) {
    setConfigStatus("Finish the current message before changing configuration.");
    return;
  }
  if (isOnlineConfig(config) && !confirmedOnline) {
    runtimeState.pendingConfig = config;
    showOnlineConfirm(config);
    return;
  }
  runtimeState.applying = true;
  setComposerEnabled(false);
  configApply.disabled = true;
  configReset.disabled = true;
  setConfigStatus("Building engine\u2026 first switch can take ~30s.");
  try {
    const info = await apiRequest("/api/runtime/config", {method: "POST", body: JSON.stringify(config)});
    renderConfigPanel(info);
    setConfigStatus("Configuration applied. Starting a new chat.");
    if (info.sessions_reset) {
      await newChat({allowDuringConfig: true});
    }
  } catch (error) {
    setConfigStatus("Could not apply the configuration. Check the API key and try again.");
  } finally {
    runtimeState.applying = false;
    configApply.disabled = state.pending;
    configReset.disabled = state.pending;
    renderConversation();
  }
}

async function loadRuntime() {
  const info = await apiRequest("/api/runtime");
  renderConfigPanel(info);
}

configProvider.addEventListener("change", () => {
  populateModelOptions(configProvider.value);
});

configApply.addEventListener("click", () => {
  applyConfig(collectConfig()).catch(handleUnexpectedInteractionError);
});

configReset.addEventListener("click", () => {
  configProvider.value = "none";
  populateModelOptions("none");
  configApiKey.value = "";
  configRerank.value = "none";
  configRetrieval.value = "auto";
  configOutput.value = "holdback";
  toggleLlmIntent.checked = false;
  toggleFingerprint.checked = true;
  toggleCategory.checked = true;
  toggleParaphrase.checked = true;
  applyConfig(collectConfig()).catch(handleUnexpectedInteractionError);
});

onlineConfirmOk.addEventListener("click", () => {
  const config = runtimeState.pendingConfig;
  closeOnlineConfirm();
  if (config) {
    applyConfig(config, {confirmedOnline: true}).catch(handleUnexpectedInteractionError);
  }
});

onlineConfirmCancel.addEventListener("click", closeOnlineConfirm);
onlineConfirmBackdrop.addEventListener("click", closeOnlineConfirm);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !onlineConfirm.hidden) {
    closeOnlineConfirm();
  }
});


/* ------------------------------------------------------------------ */
/* Conversation history (multi-session, localStorage-backed)           */
/* ------------------------------------------------------------------ */
const historyButton = document.querySelector("#history-button");
const historyDrawer = document.querySelector("#history-drawer");
const historyBackdrop = document.querySelector("#history-backdrop");
const historyClose = document.querySelector("#history-close");
const historyList = document.querySelector("#history-list");
const historyEmpty = document.querySelector("#history-empty");

function createConversation(sessionId) {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    title: "",
    createdAt: now,
    updatedAt: now,
    sessionId,
    messages: [],
  };
}

function formatHistoryTime(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay ? date.toLocaleTimeString() : date.toLocaleDateString();
}

function renderHistory() {
  historyList.replaceChildren();
  const sorted = [...conversations].sort(
    (a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""),
  );
  historyEmpty.hidden = sorted.length > 0;
  for (const conversation of sorted) {
    const item = element("article", "history-item");
    if (conversation.id === activeConversationId) {
      item.classList.add("active");
    }
    const body = element("div", "history-item-body");
    body.append(element(
      "span",
      "history-title",
      conversation.title || "Untitled conversation",
    ));
    const count = conversation.messages.length;
    body.append(element(
      "span",
      "history-meta",
      `${formatHistoryTime(conversation.updatedAt)} \u00b7 ${count} msg${count === 1 ? "" : "s"}`,
    ));
    body.addEventListener("click", () => openConversation(conversation.id));
    const deleteButton = element("button", "history-delete", "Delete");
    deleteButton.type = "button";
    deleteButton.addEventListener("click", () => deleteConversation(conversation.id));
    item.append(body, deleteButton);
    historyList.append(item);
  }
}

function openHistory() {
  if (runtimeState.applying) {
    setConfigStatus("Finish applying the configuration before opening history.");
    return;
  }
  renderHistory();
  historyDrawer.hidden = false;
  historyBackdrop.hidden = false;
  historyDrawer.setAttribute("aria-hidden", "false");
  historyClose.focus();
}

function closeHistory() {
  historyDrawer.hidden = true;
  historyBackdrop.hidden = true;
  historyDrawer.setAttribute("aria-hidden", "true");
}

function openConversation(conversationId) {
  if (runtimeState.applying) {
    setConfigStatus("Finish applying the configuration before switching conversations.");
    return;
  }
  const conversation = conversations.find((item) => item.id === conversationId);
  if (!conversation) {
    return;
  }
  activeConversationId = conversation.id;
  state = stateFromConversation(conversation);
  persistConversations();
  closeHistory();
  renderConversation();
  showNotice("");
  if (!state.sessionId) {
    newChat().catch(handleUnexpectedInteractionError);
  } else {
    messageInput.focus();
  }
}

function deleteConversation(conversationId) {
  if (runtimeState.applying) {
    setConfigStatus("Finish applying the configuration before changing history.");
    return;
  }
  const index = conversations.findIndex((item) => item.id === conversationId);
  if (index === -1) {
    return;
  }
  conversations.splice(index, 1);
  if (activeConversationId === conversationId) {
    activeConversationId = null;
    const next = conversations[0] || null;
    if (next) {
      activeConversationId = next.id;
      state = stateFromConversation(next);
    } else {
      state = emptyState();
      newChat()
        .catch(handleUnexpectedInteractionError)
        .then(() => {
          if (!historyDrawer.hidden) {
            renderHistory();
          }
        });
    }
  }
  persistConversations();
  renderHistory();
  renderConversation();
}

historyButton.addEventListener("click", openHistory);
historyClose.addEventListener("click", closeHistory);
historyBackdrop.addEventListener("click", closeHistory);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !historyDrawer.hidden) {
    closeHistory();
  }
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value;
  if (state.pending || runtimeState.applying || !text.trim()) {
    return;
  }
  messageInput.value = "";
  sendMessage(text).catch(handleUnexpectedInteractionError);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

newChatButton.addEventListener("click", () => {
  newChat().catch(handleUnexpectedInteractionError);
});

drawerClose.addEventListener("click", closeProductDrawer);
backdrop.addEventListener("click", closeProductDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !drawer.hidden) {
    closeProductDrawer();
  } else if (event.key === "Tab" && !drawer.hidden) {
    containDrawerFocus(event);
  }
});

navChat.addEventListener("click", () => showView("chat"));
navDashboard.addEventListener("click", () => showView("dashboard"));

configAuto.addEventListener("click", () => {
  applyRecommendedConfig().catch(handleUnexpectedInteractionError);
});

bootstrap().catch(renderInitializationFailure);
