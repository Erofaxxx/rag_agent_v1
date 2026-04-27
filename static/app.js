const API_BASE = "/api";

const state = {
    conversationId: null,
    polling: new Set(),
};

const els = {
    docList: document.getElementById("docList"),
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    fileInputFlat: document.getElementById("fileInputFlat"),
    pickFolderBtn: document.getElementById("pickFolderBtn"),
    pickFilesBtn: document.getElementById("pickFilesBtn"),
    status: document.getElementById("status"),
    messages: document.getElementById("messages"),
    chatForm: document.getElementById("chatForm"),
    chatInput: document.getElementById("chatInput"),
    sendBtn: document.getElementById("sendBtn"),
    newConvBtn: document.getElementById("newConvBtn"),
    conversationTitle: document.getElementById("conversationTitle"),
    modal: document.getElementById("modal"),
    modalTitle: document.getElementById("modalTitle"),
    modalBody: document.getElementById("modalBody"),
    modalClose: document.getElementById("modalClose"),
};

function escapeHtml(s) {
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function setStatus(msg) {
    els.status.textContent = msg;
}

async function api(path, opts = {}) {
    const res = await fetch(API_BASE + path, opts);
    if (res.status === 401) {
        // Браузер сам всплывёт BasicAuth — но если уже отказали, просто сообщим
        throw new Error("Не авторизован");
    }
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`${res.status}: ${text || res.statusText}`);
    }
    if (res.status === 204) return null;
    return res.json();
}

// --- Documents ---

async function loadDocuments() {
    try {
        const docs = await api("/documents");
        renderDocuments(docs);
        const ready = docs.filter(d => d.status === "ready").length;
        setStatus(`${ready} / ${docs.length} готовы`);
        for (const d of docs) {
            if (d.status === "pending" || d.status === "processing") {
                pollDocument(d.id);
            }
        }
    } catch (e) {
        setStatus("Ошибка загрузки списка: " + e.message);
    }
}

function renderDocuments(docs) {
    els.docList.innerHTML = "";
    if (!docs.length) {
        const li = document.createElement("li");
        li.className = "doc-meta";
        li.style.padding = "10px";
        li.textContent = "Документов пока нет";
        els.docList.appendChild(li);
        return;
    }
    for (const d of docs) {
        const li = document.createElement("li");
        li.className = "doc-item";
        const sizeKb = (d.file_size / 1024).toFixed(0);
        li.innerHTML = `
            <div style="flex:1; min-width: 0;">
                <div class="doc-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</div>
                <div class="doc-meta">${d.file_type.toUpperCase()} · ${sizeKb} KB · ${d.chunk_count} чанков</div>
            </div>
            <span class="doc-status ${d.status}">${statusLabel(d.status)}</span>
            <button class="doc-delete" data-id="${d.id}" title="Удалить">×</button>
        `;
        els.docList.appendChild(li);
    }
    els.docList.querySelectorAll(".doc-delete").forEach(btn => {
        btn.addEventListener("click", async () => {
            const id = btn.getAttribute("data-id");
            if (!confirm("Удалить документ?")) return;
            try {
                await api(`/documents/${id}`, { method: "DELETE" });
                await loadDocuments();
            } catch (e) {
                alert("Не удалось удалить: " + e.message);
            }
        });
    });
}

function statusLabel(s) {
    return {
        pending: "ждёт",
        processing: "обработка",
        ready: "готов",
        error: "ошибка",
    }[s] || s;
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
                return;
            }
        } catch (e) {
            state.polling.delete(id);
            return;
        }
    }
}

// --- Upload ---

async function uploadFiles(fileList) {
    const allowed = /\.(pdf|docx|xlsx|xlsm|pptx)$/i;
    const files = Array.from(fileList).filter(f => allowed.test(f.name));
    if (!files.length) {
        alert("Нет поддерживаемых файлов (PDF, DOCX, XLSX, PPTX)");
        return;
    }
    setStatus(`Загружаю ${files.length} файл(ов)...`);
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    try {
        const res = await api("/documents", { method: "POST", body: fd });
        await loadDocuments();
        setStatus(`Загружено ${res.documents.length}, ждём обработку`);
    } catch (e) {
        alert("Ошибка загрузки: " + e.message);
        setStatus("Ошибка загрузки");
    }
}

// --- Drag & drop ---

async function getFilesFromDataTransfer(dt) {
    const files = [];
    if (dt.items && dt.items.length && dt.items[0].webkitGetAsEntry) {
        const entries = [];
        for (let i = 0; i < dt.items.length; i++) {
            const entry = dt.items[i].webkitGetAsEntry();
            if (entry) entries.push(entry);
        }
        for (const e of entries) {
            await traverse(e, files);
        }
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
            const readEntries = () => {
                reader.readEntries(async (entries) => {
                    if (!entries.length) {
                        for (const sub of all) await traverse(sub, out);
                        resolve();
                    } else {
                        all.push(...entries);
                        readEntries();
                    }
                });
            };
            readEntries();
        } else {
            resolve();
        }
    });
}

els.dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    els.dropZone.classList.add("dragging");
});
els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("dragging");
});
els.dropZone.addEventListener("drop", async e => {
    e.preventDefault();
    els.dropZone.classList.remove("dragging");
    const files = await getFilesFromDataTransfer(e.dataTransfer);
    if (files.length) await uploadFiles(files);
});

els.pickFolderBtn.addEventListener("click", () => els.fileInput.click());
els.pickFilesBtn.addEventListener("click", () => els.fileInputFlat.click());
els.fileInput.addEventListener("change", e => uploadFiles(e.target.files));
els.fileInputFlat.addEventListener("change", e => uploadFiles(e.target.files));

// --- Chat ---

function renderMessage(role, content, citedChunks = []) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.innerHTML = escapeHtml(content);

    if (role === "assistant" && citedChunks && citedChunks.length) {
        const cited = document.createElement("div");
        cited.className = "cited";
        cited.innerHTML = `<div class="cited-title">Источники:</div>`;
        const list = document.createElement("div");
        list.className = "cited-list";
        for (const c of citedChunks) {
            const chip = document.createElement("button");
            chip.className = "cited-chip";
            const loc = [
                c.page_number ? `стр. ${c.page_number}` : null,
                c.sheet_name ? `лист ${c.sheet_name}` : null,
                c.slide_number ? `слайд ${c.slide_number}` : null,
            ].filter(Boolean).join(", ");
            chip.textContent = `${c.filename}${loc ? " · " + loc : ""}`;
            chip.addEventListener("click", () => showSnippet(c));
            list.appendChild(chip);
        }
        cited.appendChild(list);
        div.appendChild(cited);
    }
    els.messages.appendChild(div);
    els.messages.scrollTop = els.messages.scrollHeight;
    return div;
}

function showSnippet(chunk) {
    const loc = [
        chunk.page_number ? `стр. ${chunk.page_number}` : null,
        chunk.sheet_name ? `лист ${chunk.sheet_name}` : null,
        chunk.slide_number ? `слайд ${chunk.slide_number}` : null,
    ].filter(Boolean).join(", ");
    els.modalTitle.textContent = `${chunk.filename}${loc ? " · " + loc : ""}`;
    els.modalBody.textContent = chunk.snippet || "";
    els.modal.classList.remove("hidden");
}

els.modalClose.addEventListener("click", () => els.modal.classList.add("hidden"));
els.modal.addEventListener("click", e => {
    if (e.target === els.modal) els.modal.classList.add("hidden");
});

els.chatInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        els.chatForm.requestSubmit();
    }
});

els.chatForm.addEventListener("submit", async e => {
    e.preventDefault();
    const text = els.chatInput.value.trim();
    if (!text) return;
    els.chatInput.value = "";
    if (els.messages.querySelector(".welcome")) {
        els.messages.innerHTML = "";
    }
    renderMessage("user", text);
    const thinking = renderMessage("thinking", "Думаю...");
    els.sendBtn.disabled = true;
    try {
        const res = await api("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                conversation_id: state.conversationId,
            }),
        });
        thinking.remove();
        state.conversationId = res.conversation_id;
        renderMessage("assistant", res.answer, res.cited_chunks);
        els.conversationTitle.textContent = `Диалог #${res.conversation_id}`;
    } catch (err) {
        thinking.remove();
        renderMessage("error", "Ошибка: " + err.message);
    } finally {
        els.sendBtn.disabled = false;
        els.chatInput.focus();
    }
});

els.newConvBtn.addEventListener("click", () => {
    state.conversationId = null;
    els.messages.innerHTML = `
        <div class="welcome">
            Новый диалог. Задайте вопрос по загруженным документам.
        </div>
    `;
    els.conversationTitle.textContent = "Новый диалог";
    els.chatInput.focus();
});

// --- Init ---

loadDocuments();
els.chatInput.focus();
