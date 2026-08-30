"use strict";

const STORAGE_KEY = "shopping-copilot-web:v1";
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
let focusBeforeDrawer = null;

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

function persistState() {
  state.updatedAt = new Date().toISOString();
  localStorage.setItem(STORAGE_KEY, JSON.stringify({version: 1, ...state}));
}

function restoreState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!saved || saved.version !== 1 || !Array.isArray(saved.messages)) {
      return emptyState();
    }
    return {
      sessionId: typeof saved.sessionId === "string" ? saved.sessionId : null,
      messages: saved.messages.map((message) => (
        message.role === "user" && message.status === "pending"
          ? {...message, status: "failed"}
          : message
      )),
      pending: false,
      updatedAt: typeof saved.updatedAt === "string" ? saved.updatedAt : null,
    };
  } catch (error) {
    localStorage.removeItem(STORAGE_KEY);
    return emptyState();
  }
}

function showNotice(message) {
  statusNotice.textContent = message;
}

function setComposerEnabled(enabled) {
  messageInput.disabled = !enabled;
  sendButton.disabled = !enabled;
  newChatButton.disabled = !serviceReady || state.pending;
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

function renderProducts(payload) {
  const grid = element("div", "product-grid");
  payload.agent_response.recommendations.forEach((recommendation, index) => {
    const asin = recommendation.parent_asin;
    const product = payload.products[asin];
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
    const numericPrice = Number(product?.price);
    card.append(element(
      "p",
      "product-price",
      product?.price !== null
        && product?.price !== undefined
        && Number.isFinite(numericPrice)
        ? `$${numericPrice.toFixed(2)}`
        : "Price unavailable",
    ));
    if (product?.average_rating !== null && product?.average_rating !== undefined) {
      const count = product.rating_number ? ` (${product.rating_number})` : "";
      card.append(element(
        "p",
        "product-rating",
        `${product.average_rating} stars${count}`,
      ));
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

function renderProductDetail(product) {
  drawerBody.replaceChildren();
  drawerTitle.textContent = product.title || "Product details";
  drawerBody.append(element("p", "product-asin", product.parent_asin));
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

async function openProductDrawer(asin) {
  focusBeforeDrawer = document.activeElement;
  drawer.hidden = false;
  backdrop.hidden = false;
  drawer.setAttribute("aria-hidden", "false");
  drawerTitle.textContent = "Loading product details…";
  drawerBody.replaceChildren();
  drawerClose.focus();
  try {
    const product = await apiRequest(`/api/products/${encodeURIComponent(asin)}`);
    renderProductDetail(product);
  } catch (error) {
    drawerTitle.textContent = "Product details unavailable";
    drawerBody.textContent = "Product details are unavailable for this recommendation.";
  }
}

function closeProductDrawer() {
  drawer.hidden = true;
  backdrop.hidden = true;
  drawer.setAttribute("aria-hidden", "true");
  if (focusBeforeDrawer instanceof HTMLElement) {
    focusBeforeDrawer.focus();
  }
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
    retry.disabled = state.pending;
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
}

async function replaceExpiredSession() {
  const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
  state = emptyState();
  state.sessionId = created.session_id;
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
    const payload = await apiRequest(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({message_id: messageId, message: text}),
    });
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
  const messageId = crypto.randomUUID();
  state.messages.push({role: "user", text, messageId, status: "pending"});
  return submitExistingMessage(text, messageId);
}

function retryMessage(messageId) {
  const failed = state.messages.find(
    (item) => item.role === "user" && item.messageId === messageId,
  );
  if (!failed || state.pending) {
    return;
  }
  failed.status = "pending";
  return submitExistingMessage(failed.text, failed.messageId);
}

async function newChat() {
  if (state.pending || !serviceReady) {
    return;
  }
  state.pending = true;
  renderConversation();
  showNotice("Starting a new chat...");
  try {
    const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
    state = emptyState();
    state.sessionId = created.session_id;
    persistState();
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
  state = restoreState();
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

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value;
  if (state.pending || !text.trim()) {
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
  }
});

bootstrap().catch(renderInitializationFailure);
