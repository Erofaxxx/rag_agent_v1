/* Применяет тему ДО рендера body, чтобы не было FOUC.
   Подключается в <head> через src — CSP запрещает inline скрипты.
   Дефолт — тёмная (без data-theme); если юзер выбрал светлую, ставим
   data-theme="light" на <html>. */
(function () {
    try {
        var t = localStorage.getItem("rag.theme");
        if (t === "light") {
            document.documentElement.setAttribute("data-theme", "light");
        } else if (t === "dark") {
            // явно тёмная — оставляем без атрибута (это и есть default)
            document.documentElement.removeAttribute("data-theme");
        }
        // Если localStorage пуст — дефолт уже из CSS (тёмная).
    } catch (e) {
        /* localStorage может быть недоступен в incognito — ничего страшного */
    }
})();

window.RagTheme = {
    get: function () {
        try {
            return localStorage.getItem("rag.theme") || "dark";
        } catch (_) {
            return "dark";
        }
    },
    set: function (theme) {
        try {
            localStorage.setItem("rag.theme", theme);
        } catch (_) {}
        if (theme === "light") {
            document.documentElement.setAttribute("data-theme", "light");
        } else {
            document.documentElement.removeAttribute("data-theme");
        }
    },
    toggle: function () {
        var current = window.RagTheme.get();
        var next = current === "light" ? "dark" : "light";
        window.RagTheme.set(next);
        return next;
    },
};
