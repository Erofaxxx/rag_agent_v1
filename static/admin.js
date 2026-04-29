/* Админ-панель: пользователи + аудит. Использует api() из auth.js */

const adminState = {
    users: [],
    me: null,
};

function escHtml(s) {
    return String(s ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function fmtDate(iso) {
    if (!iso) return "—";
    try {
        const d = new Date(iso);
        return d.toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" });
    } catch (_) {
        return iso;
    }
}

function statusOf(u) {
    if (u.locked_until && new Date(u.locked_until) > new Date()) return { kind: "locked", label: "Заблокирован" };
    if (u.is_active) return { kind: "active", label: "Активен" };
    return { kind: "pending", label: "Ожидает" };
}

async function loadUsers() {
    try {
        adminState.users = await api("/admin/users");
        renderUsers();
    } catch (e) {
        showAlert("Ошибка загрузки: " + e.message);
    }
}

function renderUsers() {
    const tbody = document.getElementById("usersBody");
    const q = (document.getElementById("userSearch").value || "").toLowerCase();
    const items = adminState.users.filter(u =>
        !q || (u.email + " " + (u.display_name || "")).toLowerCase().includes(q)
    );

    tbody.innerHTML = "";
    if (!items.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="7" style="text-align:center; color:var(--text-muted); padding:24px;">Нет пользователей</td>`;
        tbody.appendChild(tr);
        return;
    }

    for (const u of items) {
        const st = statusOf(u);
        const isMe = adminState.me && u.id === adminState.me.id;
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>
                <strong>${escHtml(u.email)}</strong>
                ${isMe ? '<span style="font-size:11px; color:var(--accent);"> (вы)</span>' : ""}
            </td>
            <td>${escHtml(u.display_name || "—")}</td>
            <td><span class="role-badge ${u.role}">${u.role}</span></td>
            <td>
                <span class="status-pill ${st.kind}">${st.label}</span>
                ${u.failed_login_count > 0 ? `<span style="font-size:11px; color:var(--text-muted); margin-left:6px;">неуд. попыток: ${u.failed_login_count}</span>` : ""}
            </td>
            <td style="font-size:12px; color:var(--text-secondary);">${fmtDate(u.created_at)}</td>
            <td style="font-size:12px; color:var(--text-secondary);">${fmtDate(u.last_login_at)}</td>
            <td><div class="actions-cell" data-uid="${u.id}"></div></td>
        `;
        const cell = tr.querySelector(".actions-cell");
        renderActions(cell, u, isMe);
        tbody.appendChild(tr);
    }
}

function btn(label, kind, onClick) {
    const b = document.createElement("button");
    b.className = `${kind} btn-tiny`;
    b.textContent = label;
    b.addEventListener("click", onClick);
    return b;
}

function renderActions(cell, u, isMe) {
    cell.innerHTML = "";
    if (!u.is_active) {
        cell.appendChild(btn("Одобрить (user)", "btn-primary", () => approve(u, "user")));
        cell.appendChild(btn("Одобрить (admin)", "btn-secondary", () => approve(u, "admin")));
    } else {
        if (u.role === "user") {
            cell.appendChild(btn("→ admin", "btn-secondary", () => setRole(u, "admin")));
        } else {
            cell.appendChild(btn("→ user", "btn-secondary", () => setRole(u, "user")));
        }
        if (!isMe) {
            cell.appendChild(btn("Деактивировать", "btn-secondary", () => deactivate(u)));
        }
    }
    if (u.failed_login_count > 0 || u.locked_until) {
        cell.appendChild(btn("Снять блок", "btn-secondary", () => unlock(u)));
    }
    cell.appendChild(btn("Сменить пароль", "btn-secondary", () => resetPassword(u)));
    if (!isMe) {
        cell.appendChild(btn("Удалить", "btn-secondary", () => deleteUser(u)));
    }
}

async function approve(u, role) {
    try {
        await api(`/admin/users/${u.id}/approve`, {
            method: "POST",
            body: JSON.stringify({ role }),
        });
        await loadUsers();
    } catch (e) { showAlert(e.message); }
}

async function setRole(u, role) {
    if (!confirm(`Изменить роль ${u.email} на ${role}?`)) return;
    try {
        await api(`/admin/users/${u.id}/role`, {
            method: "POST",
            body: JSON.stringify({ role }),
        });
        await loadUsers();
    } catch (e) { showAlert(e.message); }
}

async function deactivate(u) {
    if (!confirm(`Деактивировать ${u.email}? Все его сессии будут завершены.`)) return;
    try {
        await api(`/admin/users/${u.id}/deactivate`, { method: "POST" });
        await loadUsers();
    } catch (e) { showAlert(e.message); }
}

async function unlock(u) {
    try {
        await api(`/admin/users/${u.id}/unlock`, { method: "POST" });
        await loadUsers();
    } catch (e) { showAlert(e.message); }
}

async function resetPassword(u) {
    const pwd = prompt(`Новый пароль для ${u.email}\n(минимум 10 символов, буква + цифра):`);
    if (!pwd) return;
    try {
        await api(`/admin/users/${u.id}/reset-password`, {
            method: "POST",
            body: JSON.stringify({ new_password: pwd }),
        });
        alert("Пароль изменён. Все сессии пользователя завершены.");
    } catch (e) { showAlert(e.message); }
}

async function deleteUser(u) {
    if (!confirm(`Удалить ${u.email} безвозвратно?\nВсе его диалоги тоже будут удалены.`)) return;
    try {
        await api(`/admin/users/${u.id}`, { method: "DELETE" });
        await loadUsers();
    } catch (e) { showAlert(e.message); }
}

// ---------- Audit ----------

const EVENT_LABELS = {
    register: "регистрация",
    login_success: "успешный вход",
    login_fail: "неуд. вход",
    logout: "выход",
    approve: "одобрен",
    reject: "деактивация",
    role_change: "смена роли",
    lockout: "заблокирован",
    unlock: "разблокирован",
    delete_user: "удалён",
    password_change: "смена пароля",
};

async function loadAudit() {
    try {
        const items = await api("/admin/audit?limit=300");
        renderAudit(items);
    } catch (e) {
        showAlert("Ошибка загрузки: " + e.message);
    }
}

function renderAudit(items) {
    const wrap = document.getElementById("auditList");
    wrap.innerHTML = "";
    if (!items.length) {
        wrap.innerHTML = `<div style="padding:24px; text-align:center; color:var(--text-muted);">Журнал пуст</div>`;
        return;
    }
    const usersById = {};
    for (const u of adminState.users) usersById[u.id] = u;

    for (const a of items) {
        const u = usersById[a.user_id];
        const actor = usersById[a.actor_user_id];
        const row = document.createElement("div");
        row.className = "audit-row";
        row.innerHTML = `
            <div class="audit-time">${fmtDate(a.created_at)}</div>
            <div class="audit-event">${escHtml(EVENT_LABELS[a.event] || a.event)}</div>
            <div>
                ${u ? `<strong>${escHtml(u.email)}</strong>` : (a.user_id ? `<span style="color:var(--text-muted);">id=${a.user_id} (удалён)</span>` : "")}
                ${actor && actor.id !== a.user_id ? ` <span style="color:var(--text-muted);"> ← ${escHtml(actor.email)}</span>` : ""}
                ${a.details ? ` <span style="color:var(--text-muted); font-size:11.5px;">${escHtml(a.details)}</span>` : ""}
            </div>
            <div style="color:var(--text-muted); font-size:11.5px; font-family:var(--font-mono);">${escHtml(a.ip_address || "—")}</div>
        `;
        wrap.appendChild(row);
    }
}

// ---------- DB browser ----------

const dbState = {
    tables: [],
    currentTable: null,
    limit: 50,
    offset: 0,
};

async function loadDbTables() {
    try {
        const data = await api("/admin/db/tables");
        dbState.tables = data.tables || [];
        renderDbTables();
    } catch (e) {
        showAlert("Ошибка БД: " + e.message);
    }
}

function renderDbTables() {
    const wrap = document.getElementById("dbTables");
    wrap.innerHTML = "";
    for (const t of dbState.tables) {
        const btn = document.createElement("button");
        btn.className = "db-table-btn";
        if (t.name === dbState.currentTable) btn.classList.add("active");
        btn.innerHTML = `
            <span class="db-table-name">${escHtml(t.name)}</span>
            <span class="db-table-count">${t.rows}</span>
        `;
        btn.addEventListener("click", () => loadDbTable(t.name, 0));
        wrap.appendChild(btn);
    }
}

async function loadDbTable(name, offset) {
    dbState.currentTable = name;
    dbState.offset = offset;
    renderDbTables();
    document.getElementById("dbMeta").textContent = `Загружаю ${name}...`;
    try {
        const data = await api(`/admin/db/${encodeURIComponent(name)}?limit=${dbState.limit}&offset=${offset}`);
        renderDbContent(data);
    } catch (e) {
        document.getElementById("dbMeta").textContent = "Ошибка: " + e.message;
    }
}

function renderDbContent(data) {
    const meta = document.getElementById("dbMeta");
    const from = data.total === 0 ? 0 : data.offset + 1;
    const to = Math.min(data.offset + data.rows.length, data.total);
    meta.innerHTML = `
        <strong>${escHtml(data.table)}</strong> ·
        <span style="color:var(--text-muted);">${from}–${to} из ${data.total}</span>
    `;

    const tbl = document.getElementById("dbContent");
    tbl.innerHTML = "";
    if (!data.rows.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="${data.columns.length}" style="text-align:center; color:var(--text-muted); padding:24px;">Пусто</td>`;
        tbl.appendChild(tr);
    } else {
        const thead = document.createElement("thead");
        const trh = document.createElement("tr");
        for (const col of data.columns) {
            const th = document.createElement("th");
            th.textContent = col;
            trh.appendChild(th);
        }
        thead.appendChild(trh);
        tbl.appendChild(thead);

        const tbody = document.createElement("tbody");
        for (const row of data.rows) {
            const tr = document.createElement("tr");
            for (const v of row) {
                const td = document.createElement("td");
                if (v === null) {
                    td.innerHTML = '<span class="db-null">NULL</span>';
                } else {
                    const s = String(v);
                    td.textContent = s.length > 200 ? s.slice(0, 200) + "…" : s;
                    td.title = s;
                }
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        tbl.appendChild(tbody);
    }

    const pag = document.getElementById("dbPagination");
    if (data.total > data.limit) {
        pag.hidden = false;
        document.getElementById("dbPrev").disabled = data.offset === 0;
        document.getElementById("dbNext").disabled = data.offset + data.rows.length >= data.total;
        document.getElementById("dbPageInfo").textContent = `${from}–${to} / ${data.total}`;
    } else {
        pag.hidden = true;
    }
}

// ---------- Tabs ----------

function showTab(name) {
    document.querySelectorAll(".admin-tabs button").forEach(b => {
        b.classList.toggle("active", b.dataset.tab === name);
    });
    document.getElementById("tabUsers").style.display = name === "users" ? "" : "none";
    document.getElementById("tabAudit").style.display = name === "audit" ? "" : "none";
    document.getElementById("tabDb").style.display = name === "db" ? "" : "none";
    if (name === "audit") loadAudit();
    if (name === "db") loadDbTables();
}

async function initAdmin() {
    document.querySelectorAll(".admin-tabs button").forEach(b => {
        b.addEventListener("click", () => showTab(b.dataset.tab));
    });
    document.getElementById("reloadBtn").addEventListener("click", loadUsers);
    document.getElementById("reloadAuditBtn").addEventListener("click", loadAudit);
    document.getElementById("userSearch").addEventListener("input", renderUsers);
    document.getElementById("dbPrev").addEventListener("click", () => {
        if (dbState.currentTable) loadDbTable(dbState.currentTable, Math.max(0, dbState.offset - dbState.limit));
    });
    document.getElementById("dbNext").addEventListener("click", () => {
        if (dbState.currentTable) loadDbTable(dbState.currentTable, dbState.offset + dbState.limit);
    });
    const themeBtn = document.getElementById("themeToggleBtn");
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            if (window.RagTheme) window.RagTheme.toggle();
        });
    }
    document.getElementById("logoutBtn").addEventListener("click", async () => {
        try { await api("/auth/logout", { method: "POST" }); } catch (_) {}
        location.href = "/login";
    });

    try {
        adminState.me = await api("/auth/me");
    } catch (e) {
        location.href = "/login";
        return;
    }
    if (adminState.me.role !== "admin") {
        location.href = "/";
        return;
    }
    await loadUsers();
}

// Автоинициализация на admin.html (CSP запрещает inline <script>)
document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("usersBody")) initAdmin();
});
