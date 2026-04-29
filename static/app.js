/* ============================================================
   RAG Agent — frontend
   ============================================================ */

const API = "/api";

const state = {
    conversationId: null,
    conversations: [],
    documents: [],
    polling: new Set(),
    sending: false,
    user: null, // {id, email, role, is_active, ...}
};

// ---------- DOM ----------
const $ = id => document.getElementById(id);
const els = {
    newChatBtn: $("newChatBtn"),
    convList: $("convList"),
    convSearch: $("convSearch"),
    globalStatus: $("globalStatus"),

    chatTitle: $("chatTitle"),
    chatSubtitle: $("chatSubtitle"),
    renameChatBtn: $("renameChatBtn"),
    deleteChatBtn: $("deleteChatBtn"),
    toggleDocsBtn: $("toggleDocsBtn"),
    docsBadge: $("docsBadge"),
    messages: $("messages"),

    chatForm: $("chatForm"),
    chatInput: $("chatInput"),
    sendBtn: $("sendBtn"),
    charCount: $("charCount"),

    docsPanel: $("docsPanel"),
    closeDocsBtn: $("closeDocsBtn"),
    dropZone: $("dropZone"),
    fileInput: $("fileInput"),
    fileInputFlat: $("fileInputFlat"),
    pickFolderBtn: $("pickFolderBtn"),
    pickFilesBtn: $("pickFilesBtn"),
    welcomeUploadBtn: $("welcomeUploadBtn"),
    docCount: $("docCount"),
    chunkCount: $("chunkCount"),
    docList: $("docList"),

    modal: $("modal"),
    modalTitle: $("modalTitle"),
    modalBody: $("modalBody"),
    modalClose: $("modalClose"),
    modalBackdrop: document.querySelector("#modal .modal-backdrop"),

    toasts: $("toasts"),
    app: document.querySelector(".app"),
};

// ---------- Утилиты ----------

function escapeHtml(s) {
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    const now = new Date();
    const diff = (now - d) / 1000;
    if (diff < 60) return "только что";
    if (diff < 3600) return `${Math.floor(diff / 60)} мин`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} ч`;
    if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} дн`;
    return d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
}

function groupByDate(items, getDate) {
    const groups = { today: [], yesterday: [], week: [], older: [] };
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    for (const it of items) {
        const d = new Date(getDate(it));
        const days = Math.floor((startOfToday - d) / 86400000);
        if (days <= 0) groups.today.push(it);
        else if (days === 1) groups.yesterday.push(it);
        else if (days <= 7) groups.week.push(it);
        else groups.older.push(it);
    }
    return groups;
}

function toast(message, type = "info") {
    const t = document.createElement("div");
    t.className = `toast ${type}`;
    t.textContent = message;
    els.toasts.appendChild(t);
    setTimeout(() => {
        t.style.opacity = "0";
        t.style.transition = "opacity .25s";
        setTimeout(() => t.remove(), 250);
    }, 3500);
}

function autoresize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
}

async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    const method = (opts.method || "GET").toUpperCase();
    if (method !== "GET" && method !== "HEAD") {
        // CSRF: SameSite=Lax cookie + кастомный заголовок,
        // которого браузер не выставит при cross-site POST.
        headers["X-Requested-With"] = "fetch";
    }
    if (opts.body && !(opts.body instanceof FormData)) {
        headers["Content-Type"] = headers["Content-Type"] || "application/json";
    }
    const res = await fetch(API + path, Object.assign({}, opts, { headers, credentials: "same-origin" }));
    if (res.status === 401) {
        // Сессия протухла или её нет — на /login
        if (location.pathname !== "/login") location.href = "/login";
        throw new Error("Не авторизован");
    }
    if (res.status === 403) {
        let detail = "Недостаточно прав";
        try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_) {}
        throw new Error(detail);
    }
    if (!res.ok) {
        let detail = res.statusText;
        try {
            const j = await res.json();
            if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch (_) {
            try { detail = await res.text(); } catch (_) {}
        }
        throw new Error(`${res.status}: ${detail}`);
    }
    if (res.status === 204) return null;
    return res.json();
}

// ---------- Conversations ----------

async function loadConversations() {
    try {
        state.conversations = await api("/conversations");
        renderConversations();
    } catch (e) {
        toast("Не удалось загрузить чаты: " + e.message, "error");
    }
}

function renderConversations() {
    const q = (els.convSearch.value || "").trim().toLowerCase();
    const items = state.conversations.filter(c =>
        !q || (c.title || "").toLowerCase().includes(q)
    );

    els.convList.innerHTML = "";
    if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "conv-empty";
        empty.textContent = q ? "Ничего не найдено" : "Чатов пока нет.\nНачните новый разговор.";
        els.convList.appendChild(empty);
        return;
    }

    const groups = groupByDate(items, c => c.updated_at);
    const labels = { today: "Сегодня", yesterday: "Вчера", week: "На этой неделе", older: "Раньше" };

    for (const key of ["today", "yesterday", "week", "older"]) {
        if (!groups[key].length) continue;
        const label = document.createElement("div");
        label.className = "conv-section-label";
        label.textContent = labels[key];
        els.convList.appendChild(label);
        for (const c of groups[key]) {
            els.convList.appendChild(renderConvItem(c));
        }
    }
}

function renderConvItem(c) {
    const item = document.createElement("div");
    item.className = "conv-item";
    if (c.id === state.conversationId) item.classList.add("active");

    const title = c.title || `Диалог #${c.id}`;
    item.innerHTML = `
        <svg class="conv-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <div class="conv-title">${escapeHtml(title)}</div>
        <button class="conv-delete" title="Удалить чат" aria-label="Удалить чат">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"/></svg>
        </button>
    `;
    item.addEventListener("click", e => {
        if (e.target.closest(".conv-delete")) return;
        openConversation(c.id);
    });
    item.querySelector(".conv-delete").addEventListener("click", async e => {
        e.stopPropagation();
        if (!confirm(`Удалить чат «${title}»?`)) return;
        try {
            await api(`/conversations/${c.id}`, { method: "DELETE" });
            if (state.conversationId === c.id) startNewChat();
            await loadConversations();
            toast("Чат удалён", "success");
        } catch (err) {
            toast("Ошибка: " + err.message, "error");
        }
    });
    return item;
}

async function openConversation(id) {
    try {
        const conv = await api(`/conversations/${id}`);
        state.conversationId = conv.id;
        rememberConversation(conv.id);
        els.chatTitle.textContent = conv.title || `Диалог #${conv.id}`;
        els.chatSubtitle.textContent = `${conv.messages.length} сообщений · обновлён ${formatDate(conv.updated_at)}`;
        els.renameChatBtn.disabled = false;
        els.deleteChatBtn.disabled = false;
        els.messages.innerHTML = "";
        for (const m of conv.messages) {
            renderMessage(m.role, m.content, m.cited_chunks || []);
        }
        if (!conv.messages.length) renderWelcome();
        renderConversations();
        scrollMessagesToBottom();
    } catch (e) {
        toast("Не удалось открыть чат: " + e.message, "error");
    }
}

function startNewChat() {
    state.conversationId = null;
    rememberConversation(null);
    els.chatTitle.textContent = "Новый диалог";
    els.chatSubtitle.textContent = "Задайте вопрос по загруженным документам";
    els.renameChatBtn.disabled = true;
    els.deleteChatBtn.disabled = true;
    els.messages.innerHTML = "";
    renderWelcome();
    renderConversations();
    els.chatInput.focus();
}

function renderWelcome() {
    els.messages.innerHTML = `
        <div class="welcome">
            <div class="welcome-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            </div>
            <h1>RAG Agent</h1>
            <p>Загрузите документы (PDF, DOCX, XLSX, PPTX) и задавайте вопросы по их содержанию. Ответы строго по документам с цитатами на источник.</p>
            <div class="welcome-actions">
                <button class="btn-primary" id="welcomeUploadBtn2">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    Загрузить документы
                </button>
            </div>
            <div class="welcome-examples">
                <button class="example-chip" data-q="Сделай краткое резюме всех загруженных документов">Резюме всех документов</button>
                <button class="example-chip" data-q="Какие основные темы затронуты в документах?">Основные темы</button>
                <button class="example-chip" data-q="Найди упоминания сроков и дедлайнов">Сроки и дедлайны</button>
            </div>
        </div>
    `;
    bindWelcomeHandlers();
}

function bindWelcomeHandlers() {
    document.getElementById("welcomeUploadBtn2")?.addEventListener("click", openDocsPanel);
    document.querySelectorAll(".example-chip").forEach(btn => {
        btn.addEventListener("click", () => {
            els.chatInput.value = btn.dataset.q;
            autoresize(els.chatInput);
            els.chatInput.focus();
        });
    });
}

// ---------- Documents ----------

async function loadDocuments() {
    try {
        state.documents = await api("/documents");
        renderDocuments();
        updateDocsBadge();
        const ready = state.documents.filter(d => d.status === "ready").length;
        const procc = state.documents.filter(d => d.status === "pending" || d.status === "processing").length;
        const errors = state.documents.filter(d => d.status === "error").length;
        // Доп. статус документов в подзаголовке чата (если открыт welcome)
        // оставляем globalStatus = роль пользователя, обновлённую в renderUserWidget.

        for (const d of state.documents) {
            if (d.status === "pending" || d.status === "processing") pollDocument(d.id);
        }
    } catch (e) {
        toast("Ошибка загрузки документов: " + e.message, "error");
    }
}

function renderDocuments() {
    els.docCount.textContent = state.documents.length;
    els.chunkCount.textContent = state.documents.reduce((s, d) => s + (d.chunk_count || 0), 0);

    els.docList.innerHTML = "";
    if (!state.documents.length) {
        const empty = document.createElement("div");
        empty.className = "doc-empty";
        empty.textContent = "Документов пока нет.\nПеретащите файлы выше.";
        els.docList.appendChild(empty);
        return;
    }
    for (const d of state.documents) {
        els.docList.appendChild(renderDocItem(d));
    }
}

function renderDocItem(d) {
    const li = document.createElement("li");
    li.className = "doc-item";
    const sizeKb = (d.file_size / 1024).toFixed(0);
    const ext = (d.file_type || "").toLowerCase();
    li.innerHTML = `
        <div class="doc-icon ${ext}">${ext.toUpperCase().slice(0, 4)}</div>
        <div class="doc-info">
            <div class="doc-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</div>
            <div class="doc-meta">
                <span>${sizeKb} KB</span>
                <span>·</span>
                <span>${d.chunk_count} чанков</span>
                <span class="doc-status ${d.status}">${statusLabel(d.status)}</span>
            </div>
        </div>
        <button class="doc-delete" title="Удалить">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"/></svg>
        </button>
    `;
    li.querySelector(".doc-delete").addEventListener("click", async () => {
        if (!confirm(`Удалить «${d.filename}»?`)) return;
        try {
            await api(`/documents/${d.id}`, { method: "DELETE" });
            await loadDocuments();
            toast("Документ удалён", "success");
        } catch (e) {
            toast("Ошибка удаления: " + e.message, "error");
        }
    });
    if (d.status === "error" && d.error_message) {
        li.title = `Ошибка: ${d.error_message}`;
    }
    return li;
}

function statusLabel(s) {
    return { pending: "ждёт", processing: "обработка", ready: "готов", error: "ошибка" }[s] || s;
}

function updateDocsBadge() {
    const n = state.documents.length;
    els.docsBadge.textContent = n;
    els.docsBadge.classList.toggle("zero", n === 0);
}

async function pollDocument(id) {
    if (state.polling.has(id)) return;
    state.polling.add(id);
    while (true) {
        await new Promise(r => setTimeout(r, 2000));
        try {
            const d = await api(`/documents/${id}/status`);
            if (d.status === "ready" || d.status === "error") {
                state.polling.delete(id);
                await loadDocuments();
                if (d.status === "ready") toast(`«${d.filename}» готов`, "success");
                else toast(`Ошибка обработки «${d.filename}»`, "error");
                return;
            }
        } catch (e) {
            state.polling.delete(id);
            return;
        }
    }
}

async function uploadFiles(fileList) {
    const allowed = /\.(pdf|docx?|xlsx?|xlsm|pptx|txt|md|markdown|csv)$/i;
    const files = Array.from(fileList).filter(f => allowed.test(f.name));
    if (!files.length) {
        toast("Нет поддерживаемых файлов (PDF/DOCX/XLSX/PPTX)", "warning");
        return;
    }
    toast(`Загружаю ${files.length} файл(ов)...`);
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    try {
        const res = await api("/documents", { method: "POST", body: fd });
        await loadDocuments();
        toast(`Загружено ${res.documents.length}, обрабатываю...`, "success");
    } catch (e) {
        toast("Ошибка загрузки: " + e.message, "error");
    }
}

// ---------- Drag & drop ----------

async function getFilesFromDataTransfer(dt) {
    const files = [];
    if (dt.items && dt.items.length && dt.items[0].webkitGetAsEntry) {
        const entries = [];
        for (let i = 0; i < dt.items.length; i++) {
            const entry = dt.items[i].webkitGetAsEntry();
            if (entry) entries.push(entry);
        }
        for (const e of entries) await traverse(e, files);
        return files;
    }
    return Array.from(dt.files);
}

function traverse(entry, out) {
    return new Promise(resolve => {
        if (entry.isFile) {
            entry.file(file => { out.push(file); resolve(); });
        } else if (entry.isDirectory) {
            const reader = entry.createReader();
            const all = [];
            const read = () => reader.readEntries(async entries => {
                if (!entries.length) {
                    for (const sub of all) await traverse(sub, out);
                    resolve();
                } else {
                    all.push(...entries);
                    read();
                }
            });
            read();
        } else resolve();
    });
}

// ---------- Messages / Chat ----------

function renderMessage(role, content, citedChunks = []) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;
    if (role === "assistant") {
        row.innerHTML = `<div class="msg-avatar">RA</div>`;
    }
    const msg = document.createElement("div");
    msg.className = "message";

    if (role === "assistant" && window.marked) {
        const raw = window.marked.parse(content || "", { breaks: true, gfm: true });
        // DOMPurify защищает от XSS, если LLM вернёт HTML/script-инъекции.
        // Если по какой-то причине DOMPurify не загрузился — fallback в plain text.
        if (window.DOMPurify) {
            msg.innerHTML = window.DOMPurify.sanitize(raw, {
                ALLOWED_TAGS: ["p", "br", "strong", "em", "code", "pre", "ul", "ol", "li", "blockquote", "h1", "h2", "h3", "h4", "a", "table", "thead", "tbody", "tr", "th", "td", "hr", "del", "ins"],
                ALLOWED_ATTR: ["href", "title", "class"],
                ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|[^a-z]|[a-z+.-]+(?:[^a-z+.\-:]|$))/i,
            });
        } else {
            msg.textContent = content;
        }
    } else {
        msg.textContent = content;
    }

    if (role === "assistant" && citedChunks?.length) {
        const cited = document.createElement("div");
        cited.className = "cited";
        cited.innerHTML = `<div class="cited-title">Источники (${citedChunks.length})</div>`;
        const list = document.createElement("div");
        list.className = "cited-list";
        for (const c of citedChunks) {
            const chip = document.createElement("button");
            chip.className = "cited-chip";
            const loc = [
                c.page_number ? `стр. ${c.page_number}` : null,
                c.sheet_name ? `лист «${c.sheet_name}»` : null,
                c.slide_number ? `слайд ${c.slide_number}` : null,
            ].filter(Boolean).join(", ");
            chip.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <span>${escapeHtml(c.filename)}${loc ? " · " + escapeHtml(loc) : ""}</span>
            `;
            chip.addEventListener("click", () => showSnippet(c));
            list.appendChild(chip);
        }
        cited.appendChild(list);
        msg.appendChild(cited);
    }

    row.appendChild(msg);
    if (role === "user") row.appendChild(document.createElement("div"));
    els.messages.appendChild(row);
    return row;
}

function renderThinking() {
    const row = document.createElement("div");
    row.className = "message-row thinking assistant";
    row.innerHTML = `
        <div class="msg-avatar">RA</div>
        <div class="message">
            Думаю
            <span class="typing-dots"><span></span><span></span><span></span></span>
        </div>
    `;
    els.messages.appendChild(row);
    scrollMessagesToBottom();
    return row;
}

function scrollMessagesToBottom() {
    requestAnimationFrame(() => {
        els.messages.scrollTop = els.messages.scrollHeight;
    });
}

function showSnippet(chunk) {
    const loc = [
        chunk.page_number ? `стр. ${chunk.page_number}` : null,
        chunk.sheet_name ? `лист «${chunk.sheet_name}»` : null,
        chunk.slide_number ? `слайд ${chunk.slide_number}` : null,
    ].filter(Boolean).join(", ");
    els.modalTitle.textContent = `${chunk.filename}${loc ? " · " + loc : ""}`;
    els.modalBody.textContent = chunk.snippet || "(пустой фрагмент)";
    els.modal.classList.remove("hidden");
}

async function sendMessage() {
    const text = els.chatInput.value.trim();
    if (!text || state.sending) return;
    state.sending = true;
    els.sendBtn.disabled = true;

    const wasWelcome = !!els.messages.querySelector(".welcome");
    if (wasWelcome) els.messages.innerHTML = "";

    renderMessage("user", text);
    const thinking = renderThinking();
    els.chatInput.value = "";
    autoresize(els.chatInput);
    updateCharCount();
    scrollMessagesToBottom();

    try {
        const res = await api("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, conversation_id: state.conversationId }),
        });
        thinking.remove();
        const isNew = !state.conversationId;
        state.conversationId = res.conversation_id;
        rememberConversation(res.conversation_id);
        renderMessage("assistant", res.answer, res.cited_chunks);
        scrollMessagesToBottom();
        if (isNew) {
            await loadConversations();
            const conv = state.conversations.find(c => c.id === res.conversation_id);
            if (conv) {
                els.chatTitle.textContent = conv.title || `Диалог #${conv.id}`;
                els.renameChatBtn.disabled = false;
                els.deleteChatBtn.disabled = false;
            }
        }
        els.chatSubtitle.textContent = `обновлён только что`;
    } catch (e) {
        thinking.remove();
        const row = document.createElement("div");
        row.className = "message-row error";
        row.innerHTML = `<div class="message">Ошибка: ${escapeHtml(e.message)}</div>`;
        els.messages.appendChild(row);
        scrollMessagesToBottom();
    } finally {
        state.sending = false;
        els.sendBtn.disabled = false;
        els.chatInput.focus();
    }
}

// ---------- UI handlers ----------

function openDocsPanel() { els.app.classList.add("docs-open"); }
function closeDocsPanel() { els.app.classList.remove("docs-open"); }

function updateCharCount() {
    const n = els.chatInput.value.length;
    els.charCount.textContent = `${n} / 4000`;
}

function bindEvents() {
    els.newChatBtn.addEventListener("click", startNewChat);
    els.convSearch.addEventListener("input", renderConversations);

    els.renameChatBtn.addEventListener("click", async () => {
        if (!state.conversationId) return;
        const current = els.chatTitle.textContent;
        const next = prompt("Новое название чата:", current);
        if (!next || next === current) return;
        // API не поддерживает переименование сейчас — оставлю заглушку
        toast("Переименование чатов будет добавлено в следующей версии", "warning");
    });

    els.deleteChatBtn.addEventListener("click", async () => {
        if (!state.conversationId) return;
        if (!confirm("Удалить текущий чат?")) return;
        try {
            await api(`/conversations/${state.conversationId}`, { method: "DELETE" });
            startNewChat();
            await loadConversations();
            toast("Чат удалён", "success");
        } catch (e) {
            toast("Ошибка: " + e.message, "error");
        }
    });

    els.toggleDocsBtn.addEventListener("click", () => {
        els.app.classList.toggle("docs-open");
    });
    els.closeDocsBtn.addEventListener("click", closeDocsPanel);

    // composer
    els.chatInput.addEventListener("input", () => {
        autoresize(els.chatInput);
        updateCharCount();
    });
    els.chatInput.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    els.chatForm.addEventListener("submit", e => {
        e.preventDefault();
        sendMessage();
    });

    // upload
    els.pickFolderBtn.addEventListener("click", () => els.fileInput.click());
    els.pickFilesBtn.addEventListener("click", () => els.fileInputFlat.click());
    els.fileInput.addEventListener("change", e => uploadFiles(e.target.files));
    els.fileInputFlat.addEventListener("change", e => uploadFiles(e.target.files));
    els.welcomeUploadBtn?.addEventListener("click", openDocsPanel);

    // drag & drop в зону
    els.dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        els.dropZone.classList.add("dragging");
    });
    els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("dragging"));
    els.dropZone.addEventListener("drop", async e => {
        e.preventDefault();
        els.dropZone.classList.remove("dragging");
        const files = await getFilesFromDataTransfer(e.dataTransfer);
        if (files.length) await uploadFiles(files);
    });

    // глобальный drag&drop по всему окну — full-screen overlay + загрузка
    const overlay = document.getElementById("dragOverlay");
    let dragCounter = 0;

    function isFileDrag(e) {
        // Игнорируем перетаскивание текста/ссылок внутри страницы
        return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
    }

    window.addEventListener("dragenter", e => {
        if (!isFileDrag(e)) return;
        e.preventDefault();
        dragCounter++;
        if (dragCounter === 1) {
            overlay.classList.remove("hidden");
        }
    });
    window.addEventListener("dragleave", e => {
        if (!isFileDrag(e)) return;
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            overlay.classList.add("hidden");
        }
    });
    window.addEventListener("dragover", e => {
        if (isFileDrag(e)) e.preventDefault();
    });
    window.addEventListener("drop", async e => {
        if (!isFileDrag(e)) return;
        e.preventDefault();
        dragCounter = 0;
        overlay.classList.add("hidden");
        const files = await getFilesFromDataTransfer(e.dataTransfer);
        if (!files.length) return;
        // Если юзер не админ-only режим, открываем панель чтобы видно было прогресс
        openDocsPanel();
        await uploadFiles(files);
    });

    // modal
    els.modalClose.addEventListener("click", () => els.modal.classList.add("hidden"));
    els.modalBackdrop.addEventListener("click", () => els.modal.classList.add("hidden"));
    document.addEventListener("keydown", e => {
        if (e.key === "Escape" && !els.modal.classList.contains("hidden")) {
            els.modal.classList.add("hidden");
        }
    });
}

// ---------- Init ----------

async function loadCurrentUser() {
    try {
        state.user = await api("/auth/me");
    } catch (_) {
        // 401 уже редиректит в /login
        return false;
    }
    if (!state.user.is_active) {
        location.href = "/pending";
        return false;
    }
    renderUserWidget();
    applyRolePermissions();
    return true;
}

function renderUserWidget() {
    const u = state.user;
    if (!u) return;
    const initials = ((u.display_name || u.email).match(/\b\w/g) || []).slice(0, 2).join("").toUpperCase() || "?";
    document.getElementById("userAvatar").textContent = initials;
    document.getElementById("userName").textContent = u.display_name || u.email;
    document.getElementById("globalStatus").textContent = u.role === "admin" ? "Администратор" : "Пользователь";
    if (u.role === "admin") {
        document.getElementById("adminLink").hidden = false;
    }
}

function applyRolePermissions() {
    // Юзер не-админ: прячем upload/delete UI документов
    if (state.user && state.user.role !== "admin") {
        document.body.classList.add("role-user");
    }
}

function bindUserMenu() {
    const btn = document.getElementById("userMenuBtn");
    const dd = document.getElementById("userMenuDropdown");
    if (!btn || !dd) return;
    btn.addEventListener("click", e => {
        e.stopPropagation();
        dd.hidden = !dd.hidden;
    });
    document.addEventListener("click", e => {
        if (!dd.contains(e.target) && e.target !== btn) dd.hidden = true;
    });
    document.getElementById("logoutBtn").addEventListener("click", async () => {
        try { await api("/auth/logout", { method: "POST" }); } catch (_) {}
        location.href = "/login";
    });
    document.getElementById("themeToggleBtn").addEventListener("click", () => {
        if (window.RagTheme) {
            const next = window.RagTheme.toggle();
            toast(next === "light" ? "Светлая тема" : "Тёмная тема");
        }
    });
    document.getElementById("changePasswordBtn").addEventListener("click", async () => {
        const cur = prompt("Текущий пароль:");
        if (!cur) return;
        const next = prompt("Новый пароль (минимум 10 символов, буква + цифра):");
        if (!next) return;
        try {
            await api("/auth/change-password", {
                method: "POST",
                body: JSON.stringify({ current_password: cur, new_password: next }),
            });
            alert("Пароль изменён. Войдите заново.");
            location.href = "/login";
        } catch (e) {
            alert("Ошибка: " + e.message);
        }
    });
}

const LAST_CONV_KEY = "rag.lastConversationId";

function rememberConversation(id) {
    try {
        if (id) localStorage.setItem(LAST_CONV_KEY, String(id));
        else localStorage.removeItem(LAST_CONV_KEY);
    } catch (_) {}
}

function recallConversation() {
    try {
        const v = localStorage.getItem(LAST_CONV_KEY);
        return v ? Number(v) : null;
    } catch (_) {
        return null;
    }
}

async function init() {
    bindEvents();
    bindUserMenu();
    bindWelcomeHandlers();
    autoresize(els.chatInput);
    els.app.classList.add("docs-open");
    updateCharCount();

    const ok = await loadCurrentUser();
    if (!ok) return;

    // Параллельно: чаты и документы
    await Promise.all([loadConversations(), loadDocuments()]);

    // Восстановить последний открытый чат. Логика приоритета:
    //   1) localStorage — последний открытый юзером чат (быстро, не дёргает API);
    //   2) если localStorage пуст или ID невалиден — открываем самый свежий
    //      из state.conversations (они отсортированы по updated_at DESC на сервере).
    //      Это спасает кейс «юзер закрыл ноут на день, localStorage очистился,
    //      но ответ от LLM пришёл и записан в БД через asyncio.shield».
    //
    // Так работает ChatGPT и большинство чатов: при возврате видишь последний
    // активный диалог.
    const lastId = recallConversation();
    let resumeId = null;
    if (lastId && state.conversations.some(c => c.id === lastId)) {
        resumeId = lastId;
    } else if (state.conversations.length > 0) {
        resumeId = state.conversations[0].id;
    }
    if (resumeId) {
        await openConversation(resumeId);
    }

    // Каждые 60s обновляем статусы документов
    setInterval(loadDocuments, 60000);

    els.chatInput.focus();
}

init();
