/* Логика страниц login/register/pending. Использует те же fetch-конвенции,
   что и app.js: cookie httpOnly + X-Requested-With: fetch для CSRF. */

const API = "/api";

function $(id) { return document.getElementById(id); }

function escapeHtml(s) {
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function showAlert(msg, kind = "error") {
    const el = $("alert");
    if (!el) return;
    el.className = `alert ${kind}`;
    el.textContent = msg;
    el.style.display = "block";
}

function clearAlert() {
    const el = $("alert");
    if (el) el.style.display = "none";
}

async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (opts.method && opts.method !== "GET") {
        headers["X-Requested-With"] = "fetch";
    }
    if (opts.body && !(opts.body instanceof FormData)) {
        headers["Content-Type"] = headers["Content-Type"] || "application/json";
    }
    const res = await fetch(API + path, Object.assign({}, opts, { headers, credentials: "same-origin" }));
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) {
        const msg = (data && data.detail) || res.statusText || `HTTP ${res.status}`;
        const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
        err.status = res.status;
        throw err;
    }
    return data;
}

// ---------------- Login ----------------

function initLogin() {
    const form = $("loginForm");
    const btn = $("submitBtn");
    form.addEventListener("submit", async e => {
        e.preventDefault();
        clearAlert();
        btn.disabled = true;
        btn.textContent = "Входим...";
        try {
            const user = await api("/auth/login", {
                method: "POST",
                body: JSON.stringify({
                    email: $("email").value.trim(),
                    password: $("password").value,
                }),
            });
            if (user.is_active) {
                location.href = "/";
            } else {
                location.href = "/pending";
            }
        } catch (err) {
            showAlert(err.message);
            btn.disabled = false;
            btn.textContent = "Войти";
        }
    });
}

// ---------------- Register ----------------

function passwordStrength(pw) {
    if (!pw) return 0;
    let s = 0;
    if (pw.length >= 10) s++;
    if (pw.length >= 14) s++;
    if (/[a-zA-Zа-яА-ЯёЁ]/.test(pw) && /\d/.test(pw)) s++;
    if (/[^a-zA-Zа-яА-ЯёЁ\d]/.test(pw)) s++;
    return Math.min(4, s);
}

function initRegister() {
    const form = $("registerForm");
    const btn = $("submitBtn");
    const pw = $("password");
    const meter = $("meter");
    pw.addEventListener("input", () => {
        meter.className = "password-meter s" + passwordStrength(pw.value);
    });

    form.addEventListener("submit", async e => {
        e.preventDefault();
        clearAlert();
        btn.disabled = true;
        btn.textContent = "Регистрируем...";
        try {
            const payload = {
                email: $("email").value.trim(),
                password: pw.value,
            };
            const dn = $("display_name").value.trim();
            if (dn) payload.display_name = dn;

            const user = await api("/auth/register", {
                method: "POST",
                body: JSON.stringify(payload),
            });

            // После успешной регистрации сразу пробуем залогинить пользователя.
            // Если is_active=true (был самым первым) — пойдёт в чат.
            // Если is_active=false — на /pending.
            try {
                await api("/auth/login", {
                    method: "POST",
                    body: JSON.stringify({ email: payload.email, password: payload.password }),
                });
            } catch (loginErr) {
                if (loginErr.status === 403) {
                    location.href = "/pending";
                    return;
                }
                throw loginErr;
            }
            location.href = user.is_active ? "/" : "/pending";
        } catch (err) {
            showAlert(err.message);
            btn.disabled = false;
            btn.textContent = "Создать аккаунт";
        }
    });
}

// ---------------- Pending ----------------

function initPending() {
    $("logoutBtn").addEventListener("click", async () => {
        try {
            await api("/auth/logout", { method: "POST" });
        } catch (_) {}
        location.href = "/login";
    });

    // Периодически проверяем — может быть админ уже одобрил
    setInterval(async () => {
        try {
            const me = await api("/auth/me");
            if (me && me.is_active) location.href = "/";
        } catch (_) { /* ignore */ }
    }, 15000);
}

// Автоинициализация по наличию форм. CSP запрещает inline <script>, поэтому
// диспатчер тут.
document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("loginForm")) initLogin();
    else if (document.getElementById("registerForm")) initRegister();
    else if (document.getElementById("logoutBtn") && document.querySelector(".auth-card")) {
        // pending.html (там тоже logoutBtn в auth-card)
        initPending();
    }
});
