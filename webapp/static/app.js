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

const newChatButton = document.querySelector("#new-chat");
const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const promptExamples = document.querySelector("#prompt-examples");
const serviceStatus = document.querySelector("#service-status");
const statusNotice = document.querySelector("#status-notice");
const composer = document.querySelector("#composer");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-message");

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
  newChatButton.disabled = !enabled;
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
    retry.addEventListener("click", () => retryMessage(message.messageId));
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
  return row;
}

function renderConversation() {
  const rows = state.messages.map((message) => (
    message.role === "user" ? renderUserMessage(message) : renderAssistantMessage(message)
  ));
  conversation.replaceChildren(...rows);
  welcome.hidden = state.messages.length > 0;
  setComposerEnabled(Boolean(state.sessionId) && !state.pending);
}

async function replaceExpiredSession() {
  state = emptyState();
  const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
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
  showNotice("Understanding your request and searching products...");

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
      await replaceExpiredSession();
      finalNotice = "The local service restarted. Starting a new chat.";
    } else {
      userMessage.status = "failed";
      finalNotice = "The message could not be sent. You can retry it.";
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

async function retryMessage(messageId) {
  if (state.pending) {
    return;
  }
  const message = state.messages.find(
    (item) => item.role === "user" && item.messageId === messageId,
  );
  if (message) {
    await submitExistingMessage(message.text, message.messageId);
  }
}

async function newChat() {
  if (state.pending) {
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
    const health = await apiRequest("/api/health");
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
    state = restoreState();
    if (state.sessionId) {
      try {
        await apiRequest(`/api/sessions/${state.sessionId}`);
      } catch (error) {
        if (error instanceof ApiError && error.code === "session_not_found") {
          await replaceExpiredSession();
        } else {
          throw error;
        }
      }
    } else {
      const created = await apiRequest("/api/sessions", {method: "POST", body: "{}"});
      state.sessionId = created.session_id;
    }
    persistState();
    serviceStatus.textContent = "Local · Ready";
    composer.hidden = false;
    renderConversation();
  } catch (error) {
    renderInitializationFailure();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value;
  if (state.pending || !text.trim()) {
    return;
  }
  messageInput.value = "";
  void sendMessage(text);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

newChatButton.addEventListener("click", () => {
  void newChat();
});

void bootstrap();
