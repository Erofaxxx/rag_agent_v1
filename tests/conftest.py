"""Конфиг тестов: добавляет корень репо в PYTHONPATH и блокирует попадание
тестового settings-инстанса в нужду переменных окружения OPENROUTER_API_KEY и т.п.

Тесты для defenses/ и attacks/ работают без сети и без БД — только чистый
Python и numpy. Поэтому мы НЕ импортируем config.py в этих тестах.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Добавляем корень репо в sys.path (на случай, если pytest запускается не из корня).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Чтобы тесты не падали из-за обязательных переменных в config.Settings,
# мы НЕ импортируем config — но если кто-то это сделает транзитивно, дадим
# заглушки.
os.environ.setdefault("OPENROUTER_API_KEY", "test-stub")
