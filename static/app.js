/* ============================================================
   RAG Agent — frontend
   ============================================================ */

const API = "/api";

const state = {
    conversationId: null,
    conversations: [],
    documents: [],
    notebooks: [],
    notebookId: null,
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

// Глобальный обработчик ошибок: всё что вылетело наружу из любого слушателя
// должен увидеть пользователь, а не ловить в DevTools.
window.addEventListener("error", (e) => {
    console.error("Global error:", e.error || e.message);
    try { toast(`JS error: ${(e.error && e.error.message) || e.message}`, "error"); } catch (_) {}
});
window.addEventListener("unhandledrejection", (e) => {
    console.error("Unhandled rejection:", e.reason);
    try { toast(`Promise rejected: ${e.reason && e.reason.message ? e.reason.message : e.reason}`, "error"); } catch (_) {}
});

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
        const q = state.notebookId ? `?notebook_id=${state.notebookId}` : "";
        state.conversations = await api("/conversations" + q);
        renderConversations();
    } catch (e) {
        toast("Не удалось загрузить чаты: " + e.message, "error");
    }
}

// ============== Notebooks ==============

const LAST_NB_KEY = "rag.lastNotebookId";

async function loadNotebooks() {
    try {
        state.notebooks = await api("/notebooks");
        // Восстановим выбранный ноутбук: localStorage или первый
        let savedId = null;
        try { savedId = parseInt(localStorage.getItem(LAST_NB_KEY) || "0", 10) || null; } catch (_) {}
        if (savedId && state.notebooks.some(n => n.id === savedId)) {
            state.notebookId = savedId;
        } else if (state.notebooks.length > 0) {
            state.notebookId = state.notebooks[0].id;
        } else {
            state.notebookId = null;
        }
        renderNotebookSelector();
    } catch (e) {
        toast("Не удалось загрузить ноутбуки: " + e.message, "error");
    }
}

function renderNotebookSelector() {
    const cur = state.notebooks.find(n => n.id === state.notebookId);
    document.getElementById("notebookName").textContent = cur ? cur.name : "Документы";

    const list = document.getElementById("notebookList");
    list.innerHTML = "";
    for (const nb of state.notebooks) {
        const item = document.createElement("div");
        item.className = "notebook-item";
        if (nb.id === state.notebookId) item.classList.add("active");
        item.innerHTML = `
            <span class="notebook-item-name" title="${escapeHtml(nb.name)}">${escapeHtml(nb.name)}</span>
            <span class="notebook-item-count">${nb.document_count}</span>
            <span class="notebook-item-actions">
                <button data-action="rename" title="Переименовать">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                </button>
                <button data-action="delete" title="Удалить">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"/></svg>
                </button>
            </span>
        `;
        item.addEventListener("click", e => {
            const action = e.target.closest("[data-action]");
            if (action) {
                e.stopPropagation();
                if (action.dataset.action === "rename") return renameNotebook(nb);
                if (action.dataset.action === "delete") return deleteNotebook(nb);
            }
            switchNotebook(nb.id);
            closeNotebookDropdown();
        });
        list.appendChild(item);
    }
}

function rememberNotebook(id) {
    try {
        if (id) localStorage.setItem(LAST_NB_KEY, String(id));
        else localStorage.removeItem(LAST_NB_KEY);
    } catch (_) {}
}

async function switchNotebook(id) {
    if (state.notebookId === id) return;
    state.notebookId = id;
    rememberNotebook(id);
    state.conversationId = null;
    rememberConversation(null);
    renderNotebookSelector();
    closeSourcePanel();
    els.messages.innerHTML = "";
    els.chatTitle.textContent = "Новый диалог";
    els.chatSubtitle.textContent = "Задайте вопрос по документам ноутбука";
    els.renameChatBtn.disabled = true;
    els.deleteChatBtn.disabled = true;
    // Сначала загружаем документы и чаты нового ноутбука, потом рендерим
    // welcome — чтобы баннер «нет документов» отражал реальное состояние
    // нового ноутбука, а не предыдущего.
    await Promise.all([loadConversations(), loadDocuments()]);
    // Открываем самый свежий чат в этом ноутбуке (если есть)
    if (state.conversations.length > 0) {
        await openConversation(state.conversations[0].id);
    } else {
        renderWelcome();
    }
}

async function createNotebook() {
    const name = prompt("Название нового ноутбука:");
    if (!name || !name.trim()) return;
    try {
        const nb = await api("/notebooks", {
            method: "POST",
            body: JSON.stringify({ name: name.trim() }),
        });
        state.notebooks.push(nb);
        await switchNotebook(nb.id);
        toast(`Ноутбук «${nb.name}» создан`, "success");
    } catch (e) {
        toast("Ошибка: " + e.message, "error");
    }
}

async function renameNotebook(nb) {
    const name = prompt("Новое название:", nb.name);
    if (!name || !name.trim() || name === nb.name) return;
    try {
        const updated = await api(`/notebooks/${nb.id}`, {
            method: "PATCH",
            body: JSON.stringify({ name: name.trim() }),
        });
        const idx = state.notebooks.findIndex(n => n.id === nb.id);
        if (idx >= 0) state.notebooks[idx] = updated;
        renderNotebookSelector();
        toast("Переименовано", "success");
    } catch (e) {
        toast("Ошибка: " + e.message, "error");
    }
}

async function deleteNotebook(nb) {
    if (state.notebooks.length <= 1) {
        toast("Нельзя удалить единственный ноутбук", "warning");
        return;
    }
    if (!confirm(`Удалить ноутбук «${nb.name}»? Все документы (${nb.document_count}) и чаты (${nb.conversation_count}) внутри будут потеряны.`)) return;
    try {
        await api(`/notebooks/${nb.id}`, { method: "DELETE" });
        state.notebooks = state.notebooks.filter(n => n.id !== nb.id);
        if (state.notebookId === nb.id) {
            await switchNotebook(state.notebooks[0].id);
        } else {
            renderNotebookSelector();
        }
        toast("Ноутбук удалён", "success");
    } catch (e) {
        toast("Ошибка: " + e.message, "error");
    }
}

function toggleNotebookDropdown() {
    const sel = document.getElementById("notebookSelector");
    const dd = document.getElementById("notebookDropdown");
    if (dd.hidden) {
        dd.hidden = false;
        sel.classList.add("open");
    } else {
        closeNotebookDropdown();
    }
}

function closeNotebookDropdown() {
    const sel = document.getElementById("notebookSelector");
    const dd = document.getElementById("notebookDropdown");
    dd.hidden = true;
    sel.classList.remove("open");
}

// ============== Source viewer ==============

// State текущего PDF-просмотра.
// scale=1.0 == 100% (PDF native), zoom-info показывает scale в процентах.
const pdfState = {
    documentId: null,
    pdfDoc: null,           // pdfjsLib.PDFDocumentProxy
    currentPage: 1,
    totalPages: 1,
    scale: 1.5,
    snippet: "",            // текст цитаты для матчинга spans
    pageSpansCache: {},     // { [page]: { width, height, spans } }
};

const PDF_MIN_SCALE = 0.5;
const PDF_MAX_SCALE = 4.0;

async function ensurePdfJsWorker() {
    if (!window.pdfjsLib) {
        throw new Error("pdf.js не загрузился (проверьте CSP/CDN)");
    }
    if (!window.pdfjsLib.GlobalWorkerOptions.workerSrc) {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
            "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
    }
}

function showSource(chunk) {
    try {
        return _showSourceInner(chunk);
    } catch (e) {
        console.error("showSource error:", e);
        try { toast(`Источник: ${e.message}`, "error"); } catch (_) {}
    }
}

function _showSourceInner(chunk) {
    els.modal.classList.add("hidden");

    const loc = [
        chunk.page_number ? `стр. ${chunk.page_number}` : null,
        chunk.sheet_name ? `лист «${chunk.sheet_name}»` : null,
        chunk.slide_number ? `слайд ${chunk.slide_number}` : null,
    ].filter(Boolean).join(", ");

    document.getElementById("sourceTitle").textContent = chunk.filename || "Источник";
    document.getElementById("sourceLocation").textContent = loc || "";
    document.getElementById("sourceSnippet").textContent = chunk.snippet || "";

    const pdfViewer = document.getElementById("pdfViewer");
    const htmlViewer = document.getElementById("sourceHtmlViewer");
    const iframe = document.getElementById("sourceIframe");
    const fallback = document.getElementById("sourceFallback");
    const downloadLink = document.getElementById("sourceDownloadLink");
    const fallbackText = document.getElementById("sourceFallbackText");

    const ext = (chunk.filename || "").toLowerCase().split(".").pop();
    const fileUrl = `/api/documents/${chunk.document_id}/file`;
    downloadLink.href = fileUrl;
    document.getElementById("sourceOpenInNewTab").onclick = () => window.open(fileUrl, "_blank");

    document.getElementById("sourcePanel").classList.remove("hidden");

    // Сбрасываем все вьюеры
    pdfViewer.hidden = true;
    htmlViewer.hidden = true;
    htmlViewer.innerHTML = "";
    iframe.hidden = true;
    iframe.src = "about:blank";
    fallback.hidden = true;

    if (ext === "pdf") {
        pdfViewer.hidden = false;
        openPdfWithHighlight(chunk).catch(err => {
            console.error("PDF open error:", err);
            pdfViewer.hidden = true;
            iframe.hidden = false;
            const page = chunk.page_number ? `#page=${chunk.page_number}` : "";
            iframe.src = fileUrl + page;
        });
    } else if (["docx", "doc", "md", "markdown", "txt", "csv"].includes(ext)) {
        // Подгружаем готовый HTML/markdown/text с бэкенда и рендерим инлайн —
        // чтобы Word и markdown показывались с типографикой, а не как сырой
        // текст или «скачать файл».
        htmlViewer.hidden = false;
        htmlViewer.innerHTML = '<div class="pdf-loading">Загрузка…</div>';
        renderHtmlSource(chunk, ext, fileUrl).catch(err => {
            console.error("HTML viewer error:", err);
            // Дублируем в toast чтобы пользователь видел причину без DevTools.
            try { toast(`Просмотр документа: ${err.message}`, "error"); } catch (_) {}
            htmlViewer.hidden = true;
            fallback.hidden = false;
            fallbackText.textContent = `Не удалось открыть просмотр (${err.message}). Скачайте оригинал.`;
        });
    } else {
        // Неподдерживаемый формат — оставляем кнопку «скачать»
        fallback.hidden = false;
        const formatLabel = ext ? ext.toUpperCase() : "этот формат";
        fallbackText.textContent = `Встроенный просмотр для ${formatLabel} недоступен — скачайте оригинал, чтобы увидеть полный документ.`;
    }
}

async function renderHtmlSource(chunk, ext, fileUrl) {
    const viewer = document.getElementById("sourceHtmlViewer");
    if (!viewer) throw new Error("html-viewer element missing");

    let data;
    try {
        data = await api(`/documents/${chunk.document_id}/html`);
    } catch (e) {
        throw new Error(`fetch /html failed: ${e.message}`);
    }
    if (!data || typeof data.content !== "string") {
        throw new Error(`invalid /html response: ${JSON.stringify(data).slice(0, 80)}`);
    }

    let html = "";
    if (data.format === "html") {
        // mammoth уже отдаёт безопасный набор HTML-тегов без скриптов/стилей —
        // если DOMPurify почему-то не загрузился, не блокируем рендер.
        html = window.DOMPurify
            ? window.DOMPurify.sanitize(data.content, { USE_PROFILES: { html: true } })
            : data.content;
    } else if (data.format === "markdown") {
        const md = window.marked ? window.marked.parse(data.content) : escapeHtml(data.content);
        html = window.DOMPurify
            ? window.DOMPurify.sanitize(md, { USE_PROFILES: { html: true } })
            : md;
    } else {
        html = `<pre>${escapeHtml(data.content)}</pre>`;
    }
    if (!html.trim()) {
        throw new Error(`empty render (format=${data.format}, content len=${(data.content || "").length})`);
    }
    viewer.innerHTML = html;
    // Скролл к первому совпадению со snippet'ом (best-effort через выделение).
    if (chunk.snippet) {
        const norm = s => (s || "").toLowerCase().replace(/\s+/g, " ").trim();
        const sn = norm(chunk.snippet).slice(0, 80);
        if (sn.length > 12) {
            const walker = document.createTreeWalker(viewer, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walker.nextNode())) {
                if (norm(node.textContent).includes(sn.slice(0, 24))) {
                    const range = document.createRange();
                    range.selectNodeContents(node);
                    const rect = range.getBoundingClientRect();
                    const vRect = viewer.getBoundingClientRect();
                    viewer.scrollTop += rect.top - vRect.top - 80;
                    // Кратковременно подсвечиваем абзац
                    const parent = node.parentElement;
                    if (parent) {
                        const orig = parent.style.backgroundColor;
                        parent.style.backgroundColor = "rgba(255, 213, 79, 0.45)";
                        parent.style.transition = "background-color 1.5s";
                        setTimeout(() => {
                            parent.style.backgroundColor = orig;
                        }, 2500);
                    }
                    break;
                }
            }
        }
    }
}

async function openPdfWithHighlight(chunk) {
    await ensurePdfJsWorker();

    // Если уже открыт этот же документ — переиспользуем
    const sameDoc = pdfState.documentId === chunk.document_id && pdfState.pdfDoc;
    if (!sameDoc) {
        // Очистка предыдущего
        if (pdfState.pdfDoc) {
            try { pdfState.pdfDoc.destroy(); } catch (_) {}
        }
        pdfState.pageSpansCache = {};
        pdfState.documentId = chunk.document_id;

        const url = `/api/documents/${chunk.document_id}/file`;
        const loadingTask = window.pdfjsLib.getDocument({
            url,
            withCredentials: true,  // куки сессии для API
        });
        pdfState.pdfDoc = await loadingTask.promise;
        pdfState.totalPages = pdfState.pdfDoc.numPages;
    }

    pdfState.snippet = chunk.snippet || "";
    pdfState.currentPage = Math.max(1, Math.min(chunk.page_number || 1, pdfState.totalPages));

    // Для нового документа подбираем удобный масштаб — fit-to-width.
    // При переходе между страницами того же документа сохраняем текущий зум.
    if (!sameDoc) {
        await fitPdfToWidth();
    } else {
        await renderPdfPage();
    }
}

async function renderPdfPage() {
    if (!pdfState.pdfDoc) return;

    const page = await pdfState.pdfDoc.getPage(pdfState.currentPage);
    const viewport = page.getViewport({ scale: pdfState.scale });

    const canvas = document.getElementById("pdfCanvas");
    const ctx = canvas.getContext("2d");
    const overlay = document.getElementById("pdfOverlay");
    const container = document.getElementById("pdfPageContainer");

    canvas.width = viewport.width;
    canvas.height = viewport.height;
    container.style.width = viewport.width + "px";
    container.style.height = viewport.height + "px";

    overlay.innerHTML = "";

    await page.render({ canvasContext: ctx, viewport }).promise;

    document.getElementById("pdfPageInfo").textContent =
        `${pdfState.currentPage} / ${pdfState.totalPages}`;
    document.getElementById("pdfZoomInfo").textContent =
        `${Math.round(pdfState.scale * 100)}%`;
    document.getElementById("pdfPrevPage").disabled = pdfState.currentPage <= 1;
    document.getElementById("pdfNextPage").disabled = pdfState.currentPage >= pdfState.totalPages;

    await renderHighlightsForCurrentPage(viewport);
}

async function renderHighlightsForCurrentPage(viewport) {
    if (!pdfState.snippet) return;

    let pageData = pdfState.pageSpansCache[pdfState.currentPage];
    if (!pageData) {
        try {
            pageData = await api(`/documents/${pdfState.documentId}/page/${pdfState.currentPage}/spans`);
            pdfState.pageSpansCache[pdfState.currentPage] = pageData;
        } catch (e) {
            console.warn("Spans недоступны для подсветки:", e.message);
            return;
        }
    }
    if (!pageData || !pageData.spans) return;

    const matchedSpans = matchSpans(pdfState.snippet, pageData.spans);
    if (!matchedSpans.length) return;

    const overlay = document.getElementById("pdfOverlay");
    let firstHighlight = null;

    for (const span of matchedSpans) {
        const [vx0, vy0, vx1, vy1] = viewport.convertToViewportRectangle(span.b);
        const left = Math.min(vx0, vx1);
        const top = Math.min(vy0, vy1);
        const width = Math.abs(vx1 - vx0);
        const height = Math.abs(vy1 - vy0);
        if (width < 1 || height < 1) continue;
        const box = document.createElement("div");
        box.className = "pdf-highlight";
        box.style.cssText = `left:${left}px; top:${top}px; width:${width}px; height:${height}px;`;
        overlay.appendChild(box);
        if (!firstHighlight) firstHighlight = box;
    }

    // Скроллим к первому подсвеченному месту
    if (firstHighlight) {
        firstHighlight.scrollIntoView({ behavior: "smooth", block: "center" });
    }
}

// Матчер: ищем спаны, чей текст значимо пересекается с цитатой.
// Backend chunker рвёт текст по пробелам и предложениям, точного совпадения
// span-к-чанку не будет — поэтому идём по пересечению окон в 8+ символов.
function matchSpans(snippet, spans) {
    const norm = s => s.toLowerCase().replace(/\s+/g, " ").trim();
    const sn = norm(snippet);
    if (!sn || sn.length < 8) return [];

    const matched = [];
    for (const span of spans) {
        const t = norm(span.t || "");
        if (!t || t.length < 3) continue;
        // Если span короткий (например, цифра «3») — требуем хотя бы 6 символов
        // чтобы избежать ложных матчей. Длинный span — подсветим, если есть в цитате.
        if (t.length < 6) {
            // короткие отдельные «слова» матчим только если граничат словами
            const re = new RegExp(`(^|\\W)${escapeRegex(t)}(\\W|$)`);
            if (re.test(sn)) matched.push(span);
        } else {
            if (sn.includes(t)) matched.push(span);
            else {
                // Частичное вхождение — окна по 12 символов
                const win = 12;
                let hit = false;
                for (let i = 0; i + win <= t.length && !hit; i += 4) {
                    if (sn.includes(t.substring(i, i + win))) hit = true;
                }
                if (hit) matched.push(span);
            }
        }
    }
    return matched;
}

function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function pdfZoom(delta, originX, originY) {
    const oldScale = pdfState.scale;
    const newScale = Math.max(PDF_MIN_SCALE, Math.min(PDF_MAX_SCALE, oldScale + delta));
    if (newScale === oldScale) return;
    const wrap = document.getElementById("pdfCanvasWrap");
    // Зум вокруг точки курсора (или центра, если не передана): сохраняем
    // относительную позицию того, что пользователь видит — иначе при увеличении
    // картинка прыгает.
    const ratio = newScale / oldScale;
    const rect = wrap.getBoundingClientRect();
    const cx = (typeof originX === "number" ? originX - rect.left : wrap.clientWidth / 2);
    const cy = (typeof originY === "number" ? originY - rect.top : wrap.clientHeight / 2);
    const sx = wrap.scrollLeft + cx;
    const sy = wrap.scrollTop + cy;
    pdfState.scale = newScale;
    await renderPdfPage();
    wrap.scrollLeft = sx * ratio - cx;
    wrap.scrollTop = sy * ratio - cy;
}

function bindPdfControls() {
    document.getElementById("pdfPrevPage").addEventListener("click", async () => {
        if (pdfState.currentPage > 1) {
            pdfState.currentPage--;
            await renderPdfPage();
        }
    });
    document.getElementById("pdfNextPage").addEventListener("click", async () => {
        if (pdfState.currentPage < pdfState.totalPages) {
            pdfState.currentPage++;
            await renderPdfPage();
        }
    });
    document.getElementById("pdfZoomIn").addEventListener("click", () => pdfZoom(+0.25));
    document.getElementById("pdfZoomOut").addEventListener("click", () => pdfZoom(-0.25));

    // Ctrl/Cmd + wheel — зум, как в обычном PDF-вьювере. Без модификатора
    // оставляем нативный скролл страницы.
    const wrap = document.getElementById("pdfCanvasWrap");
    wrap.addEventListener("wheel", (e) => {
        if (!(e.ctrlKey || e.metaKey)) return;
        e.preventDefault();
        const delta = e.deltaY < 0 ? +0.15 : -0.15;
        pdfZoom(delta, e.clientX, e.clientY);
    }, { passive: false });

    // Клавиатурная навигация когда панель в фокусе или открыта.
    document.addEventListener("keydown", async (e) => {
        const panel = document.getElementById("sourcePanel");
        if (!panel || panel.classList.contains("hidden")) return;
        if (!pdfState.pdfDoc) return;
        // Не перехватываем когда фокус в input/textarea
        const tag = (document.activeElement && document.activeElement.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        if (e.key === "ArrowLeft" || e.key === "PageUp") {
            if (pdfState.currentPage > 1) {
                pdfState.currentPage--;
                await renderPdfPage();
                e.preventDefault();
            }
        } else if (e.key === "ArrowRight" || e.key === "PageDown") {
            if (pdfState.currentPage < pdfState.totalPages) {
                pdfState.currentPage++;
                await renderPdfPage();
                e.preventDefault();
            }
        } else if ((e.ctrlKey || e.metaKey) && (e.key === "+" || e.key === "=")) {
            await pdfZoom(+0.25);
            e.preventDefault();
        } else if ((e.ctrlKey || e.metaKey) && e.key === "-") {
            await pdfZoom(-0.25);
            e.preventDefault();
        } else if ((e.ctrlKey || e.metaKey) && e.key === "0") {
            // Reset to fit-width
            await fitPdfToWidth();
            e.preventDefault();
        }
    });
}

// Подбор масштаба так, чтобы страница вписывалась по ширине в pdf-canvas-wrap.
// Полезно при первом открытии PDF и по Ctrl+0.
async function fitPdfToWidth() {
    if (!pdfState.pdfDoc) return;
    const page = await pdfState.pdfDoc.getPage(pdfState.currentPage);
    const native = page.getViewport({ scale: 1 });
    const wrap = document.getElementById("pdfCanvasWrap");
    if (!wrap) return;
    // -32 = padding 16px по бокам
    const avail = Math.max(200, wrap.clientWidth - 32);
    const target = Math.max(PDF_MIN_SCALE, Math.min(PDF_MAX_SCALE, avail / native.width));
    pdfState.scale = target;
    await renderPdfPage();
}

function closeSourcePanel() {
    const panel = document.getElementById("sourcePanel");
    if (!panel) return;
    panel.classList.add("hidden");
    const iframe = document.getElementById("sourceIframe");
    if (iframe) iframe.src = "about:blank";  // освобождаем память от PDF
    const htmlViewer = document.getElementById("sourceHtmlViewer");
    if (htmlViewer) htmlViewer.innerHTML = "";
    // pdf.js: уничтожаем документ чтобы не утекал worker и память canvas
    if (pdfState.pdfDoc) {
        try { pdfState.pdfDoc.destroy(); } catch (_) {}
        pdfState.pdfDoc = null;
        pdfState.documentId = null;
        pdfState.pageSpansCache = {};
    }
    const overlay = document.getElementById("pdfOverlay");
    if (overlay) overlay.innerHTML = "";
}

// Drag-resize боковой панели. Хранит ширину в localStorage чтобы при следующем
// открытии сохранялась.
const SOURCE_WIDTH_KEY = "rag.sourcePanelWidth";

function applySavedSourceWidth() {
    try {
        const saved = parseInt(localStorage.getItem(SOURCE_WIDTH_KEY) || "0", 10);
        if (saved > 320 && saved < window.innerWidth - 200) {
            const panel = document.getElementById("sourcePanel");
            if (panel) panel.style.width = saved + "px";
        }
    } catch (_) {}
}

function bindSourceResize() {
    const handle = document.getElementById("sourceResizer");
    const panel = document.getElementById("sourcePanel");
    if (!handle || !panel) return;

    let startX = 0;
    let startWidth = 0;
    let dragging = false;

    const onMove = (e) => {
        if (!dragging) return;
        const dx = startX - e.clientX;
        // Расширяем при перетаскивании влево, сжимаем при перетаскивании вправо.
        const next = Math.max(320, Math.min(window.innerWidth - 200, startWidth + dx));
        panel.style.width = next + "px";
        // Если открыт PDF — он подхватит новую ширину при следующем зуме/перерисовке.
        // Чтобы пользователь сразу увидел fit-to-width на новой ширине, можно
        // при необходимости вызвать fitPdfToWidth, но это дёргает render —
        // сделаем по отпусканию.
    };
    const onUp = () => {
        if (!dragging) return;
        dragging = false;
        document.body.classList.remove("resizing-source");
        handle.classList.remove("active");
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        try { localStorage.setItem(SOURCE_WIDTH_KEY, String(panel.offsetWidth)); } catch (_) {}
    };
    handle.addEventListener("mousedown", (e) => {
        e.preventDefault();
        dragging = true;
        startX = e.clientX;
        startWidth = panel.offsetWidth;
        document.body.classList.add("resizing-source");
        handle.classList.add("active");
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
    });
    // Двойной клик — сброс к дефолту
    handle.addEventListener("dblclick", () => {
        panel.style.width = "";
        try { localStorage.removeItem(SOURCE_WIDTH_KEY); } catch (_) {}
    });
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
        // Backend жёстко привязывает conversation к notebook при создании.
        // Если открываем чат из другого notebook'а — синхронизируем UI, иначе
        // selector врёт и при отправке нового сообщения юзер думает, что
        // пишет в одном ноутбуке, а контекст поиска возьмётся из другого.
        if (conv.notebook_id && conv.notebook_id !== state.notebookId) {
            state.notebookId = conv.notebook_id;
            rememberNotebook(conv.notebook_id);
            renderNotebookSelector();
            await loadDocuments();
        }
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
    const nb = state.notebooks.find(n => n.id === state.notebookId);
    const nbName = nb ? nb.name : "Документы";
    const docCount = state.documents.length;
    const empty = docCount === 0;
    const banner = empty
        ? `<div class="welcome-warn">В ноутбуке «${escapeHtml(nbName)}» пока нет документов. Поиск ничего не найдёт, пока не загрузите файлы — или переключитесь на другой ноутбук в селекторе сверху слева.</div>`
        : `<div class="welcome-context">Контекст: ноутбук «${escapeHtml(nbName)}», ${docCount} документ(ов).</div>`;
    els.messages.innerHTML = `
        <div class="welcome">
            <div class="welcome-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            </div>
            <h1>RAG Agent</h1>
            <p>Загрузите документы (PDF, DOCX, XLSX, PPTX) и задавайте вопросы по их содержанию. Ответы строго по документам с цитатами на источник.</p>
            ${banner}
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
        const q = state.notebookId ? `?notebook_id=${state.notebookId}` : "";
        state.documents = await api("/documents" + q);
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
    if (state.notebookId) fd.append("notebook_id", String(state.notebookId));
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
            chip.addEventListener("click", () => showSource(c));
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
            body: JSON.stringify({
                message: text,
                conversation_id: state.conversationId,
                notebook_id: state.notebookId,
            }),
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

    // Notebook selector
    document.getElementById("notebookBtn").addEventListener("click", e => {
        e.stopPropagation();
        toggleNotebookDropdown();
    });
    document.getElementById("newNotebookBtn").addEventListener("click", e => {
        e.stopPropagation();
        closeNotebookDropdown();
        createNotebook();
    });
    document.addEventListener("click", e => {
        const sel = document.getElementById("notebookSelector");
        if (sel && !sel.contains(e.target)) closeNotebookDropdown();
    });

    // Source panel
    document.getElementById("sourceClose").addEventListener("click", closeSourcePanel);
    document.addEventListener("keydown", e => {
        if (e.key === "Escape") {
            const panel = document.getElementById("sourcePanel");
            if (panel && !panel.classList.contains("hidden")) closeSourcePanel();
        }
    });
    bindPdfControls();
    bindSourceResize();
    applySavedSourceWidth();

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

    // Сначала загружаем ноутбуки — выбираем активный, потом подтягиваем
    // чаты и документы уже scope'нутые на этот ноутбук.
    await loadNotebooks();

    // Параллельно: чаты и документы (фильтр по state.notebookId)
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
