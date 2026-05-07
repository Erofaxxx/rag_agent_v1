# Backdoor / Poisoning атаки на корпоративный RAG-агент — учебный проект

> **Если ты AI-агент в новом чате и тебе сказали «прогони security-тесты после деплоя»** — открывай [`SECURITY_RUNBOOK.md`](SECURITY_RUNBOOK.md). Это пошаговая инструкция, не импровизируй.

Курсовой по информационной безопасности ИИ-систем. Расширение существующего
[`rag_agent_v1`](README.md): добавлены **аттакующая часть** (`attacks/`) и
**три слоя защиты** (`defenses/`), интегрированные в pipeline с feature-flag'ами.

> **Важное:** все защиты по умолчанию **выключены** — если ничего не менять
> в `.env`, агент работает идентично исходному. Никакой регрессии.

---

## Что добавлено

### Атаки (модуль `attacks/`)

| Файл | Что делает | Категория по курсу |
|------|-----------|--------------------|
| `prompt_injection.py` | Прямые «игнорируй всё / теперь ты X». 5 вариантов. | Indiscriminate poisoning, baseline |
| `trigger_backdoor.py` | BadNets-style: триггер-фраза + target-ответ. **Главный кейс.** | Targeted backdoor (тема 3) |
| `citation_spoof.py` | Поддельные inline-цитаты `[file.pdf, стр. 23]` для обмана верификатора | Defense evasion |
| `retrieval_poison.py` | Keyword-stuffing для попадания в top-k | Clean-label poisoning (упрощённо) |

### Защиты (модуль `defenses/`)

| Слой | Файл | Где встраивается | Что ловит |
|------|------|------------------|-----------|
| **L1** | `l1_sanitize.py` | `api/documents.py::_process_document` после чанкинга | Прямые prompt-injection (regex по 6 категориям + NFKC normalization) |
| **L2** | `l2_embedding_anomaly.py` | `api/documents.py::_process_document` после эмбеддинга | Чанки-outliers по cosine-расстоянию до центроида документа (z-score) |
| **L4** | `l4_strict_verifier.py` | `llm/agent.py::answer_question` после `verify_answer` | Injection-паттерны в **самих cited chunks** (на регексах L1, без LLM) |

### Конфигурационные флаги (в `config.py`)

```python
DEFENSE_L1_SANITIZE: str = "off"     # off | warn | drop
DEFENSE_L1_RISK_THRESHOLD: float = 0.6

DEFENSE_L2_ANOMALY: str = "off"      # off | warn | drop
DEFENSE_L2_ZSCORE_THRESHOLD: float = 2.5

DEFENSE_L4_STRICT_VERIFIER: bool = False

SECURITY_RESEARCH_MODE: bool = False  # для будущих /api/security/* endpoint'ов
```

### Тесты

53 unit-теста без сети/LLM/БД, покрывают регексы L1, математику L2, L4
fail-open поведение и attack-генераторы:

```
tests/test_l1_sanitize.py    — 28 тестов (категории, режимы off/warn/drop, NFKC)
tests/test_l2_anomaly.py     — 7 тестов (outliers, threshold, edge cases)
tests/test_l4_strict_verifier.py — 6 тестов (injection в cited, формат warning'а)
tests/test_attacks.py        — 12 тестов (атаки реально содержат то, что обещают)
```

### Eval-скрипты

- `eval/generate_corpus.py` — собирает `benchmarks/clean_corpus/` (7 чистых) и
  `benchmarks/poisoned_corpus/` (8 отравленных) `.md`-файлов. Без сети.
- `eval/run_l1_l2_offline.py` — TPR/FPR детекторов L1/L2 на сгенерированном
  корпусе. Псевдо-эмбеддер вместо настоящего, поэтому метрика L2 будет
  скромной — это для CI/быстрого smoke-теста.
- `eval/run_e2e_attack.py` — полный сценарий через REST API запущенного
  агента. Замеряет ASR (Attack Success Rate) и clean accuracy.

---

## Что НЕ сломано (regression check)

- `config.py` — 5 новых полей с дефолтами в духе «выключено». Существующие
  поля не трогаются.
- `api/documents.py::_process_document` — два gated-блока (L1 и L2). Когда
  оба `"off"`, поведение — байт-в-байт исходное.
- `llm/verifier.py` — добавлены `strict_verify()` и `append_strict_warning()`.
  Существующие `verify_answer()` и `append_verification_warning()` не трогаются.
- `llm/agent.py` — добавлен gated-блок L4 после `verify_answer()`. Возвращаемый
  dict получает доп. ключ `strict` — `chat.py` его не читает (читает только
  `verification`), API-контракт не меняется.
- `api/chat.py` — **не трогается**. Предупреждение L4 уже встроено в текст
  ответа через `append_strict_warning()`, фронт его покажет как обычный текст.
- Никаких изменений в `requirements.txt`. Защиты используют только то, что
  уже есть (numpy, re, dataclasses).

---

## Как запускать и проверять

### 0. Установка зависимостей

В этой ветке не добавляли ничего нового. Если ты только что клонировал репо:

```bash
pip install -r requirements.txt
pip install pytest    # для тестов; в основном requirements нет
```

### 1. Прогнать unit-тесты (offline, без сети)

```bash
cd rag_agent_v1
python -m pytest tests/ -v
```

Ожидаемо: **53 passed**.

### 2. Сгенерировать тестовый корпус

```bash
PYTHONPATH=. python eval/generate_corpus.py
```

Создаст `benchmarks/clean_corpus/` (7 файлов) и `benchmarks/poisoned_corpus/`
(8 файлов).

### 3. Прогнать offline-метрику L1/L2

```bash
PYTHONPATH=. python eval/run_l1_l2_offline.py
```

Ожидаемый вывод (TPR/FPR на текущих регексах):

```
TP=3  FN=5  FP=0  TN=7
TPR (Recall на отравленных) = 37.50%
FPR (False positive rate)   = 0.00%
```

**Что это значит для отчёта.** L1 в одиночку ловит только громкие инъекции:
`direct_override_ru/en` и `system_tags`. Стелсовые атаки (`trigger_backdoor`,
`retrieval_poison`, `citation_spoof`) — ниже порога 0.6. Это **главный
учебный результат**: один слой защиты недостаточен, нужно сочетать L1 + L2 + L4.

### 4. Запустить агента и проверить, что он работает как раньше (regression)

```bash
# Заполнить .env (см. .env.example), убедиться, что DEFENSE_*=off (default)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Открыть `http://localhost:8000`, залогиниться, загрузить документ, задать
вопрос — поведение идентично main-ветке.

### 5. Прогнать e2e-атаку (требует ключ LLM/embeddings и запущенного агента)

В одном терминале:
```bash
# .env: ALL DEFENSE_* = off (baseline без защит)
uvicorn main:app --reload
```

В другом:
```bash
PYTHONPATH=. python eval/run_e2e_attack.py \
    --base http://localhost:8000 \
    --email admin@example.com \
    --password 'YourBootstrapPassword'
```

Ожидаемо без защит: **ASR ≥ 50%** на triggered-вопросах, **clean accuracy
сохраняется** — backdoor стелс.

### 6. Включить защиты и переснять метрики

В `.env`:
```
DEFENSE_L1_SANITIZE=warn
DEFENSE_L2_ANOMALY=warn
DEFENSE_L4_STRICT_VERIFIER=true
```

Перезапустить агента, повторить `run_e2e_attack.py`. Ожидаемо: ASR падает,
в ответе появляется плашка «⚠️ Предупреждение безопасности (L4)».

---

## Что отдавать в отчёте

### Метрики (нужно собрать в свою таблицу при защите)

| Конфигурация | TPR L1 | FPR L1 | TPR L2 | ASR (e2e) | Clean accuracy |
|--------------|--------|--------|--------|-----------|----------------|
| Baseline (no defense) | — | — | — | ? | ? |
| L1=warn | ? | ? | — | ? | ? |
| L1=drop | ? | ? | — | ? | ? |
| L1=warn + L2=warn | ? | ? | ? | ? | ? |
| L1+L2+L4 | ? | ? | ? | ? | ? |

### Adaptive attack

В `attacks/` дополнительно реализовать модуль для adaptive-атаки: после того
как студент знает, что L1 ловит «игнорируй всё», он:
- Кодирует инструкцию через base64 / leetspeak / синонимы.
- Использует zero-width chars (`​`) посередине триггер-слов — но мы это уже
  ловим через `suspicious_unicode`.
- Прячет triggered-инструкцию в HTML-комментарий `<!-- ... -->` (если парсер
  не вычищает их).

### Архитектурные выводы (для дискуссии)

1. **Layered defense.** L1 (regex) дёшев и ловит ~40% — но любой
   мотивированный атакующий обойдёт. L2 (embedding anomaly) и L4 (verifier
   расширенный) ловят то, что регексами ловить нельзя.
2. **Trade-off:** drop-режим L1 опасен — ложные срабатывания удалят
   легитимные чанки. Warn-режим безопаснее, но требует ручной разметки
   audit-log.
3. **Trust boundary.** Самый эффективный защитный слой — **provenance
   данных** (кто загрузил, когда, через какой канал). Это уже есть в
   существующем `audit_log` репозитория — просто его надо расширить
   полями L1/L2 reports.

---

## Структура изменений (diff overview)

```
config.py                           +28  (флаги защит)
api/documents.py                    +44  (хуки L1, L2 — gated)
llm/agent.py                        +20  (хук L4 — gated)
llm/verifier.py                     +35  (strict_verify + warning helper)

defenses/__init__.py                +24  NEW
defenses/l1_sanitize.py             +200 NEW
defenses/l2_embedding_anomaly.py    +130 NEW
defenses/l4_strict_verifier.py      +75  NEW

attacks/__init__.py                 +30  NEW
attacks/prompt_injection.py         +95  NEW
attacks/trigger_backdoor.py         +135 NEW
attacks/citation_spoof.py           +50  NEW
attacks/retrieval_poison.py         +60  NEW

tests/conftest.py                   +20  NEW
tests/test_l1_sanitize.py           +175 NEW
tests/test_l2_anomaly.py            +100 NEW
tests/test_l4_strict_verifier.py    +75  NEW
tests/test_attacks.py               +130 NEW

eval/generate_corpus.py             +110 NEW
eval/run_l1_l2_offline.py           +175 NEW
eval/run_e2e_attack.py              +180 NEW
PROJECT_README.md                   +250 NEW (этот файл)
```

Чистое добавление: ~1900 строк. Изменено в существующих: ~125 строк.

---

## Связь с курсом «Атаки на ИИ-системы»

| Тема курса | Где в проекте |
|-----------|---------------|
| Тема 1 (Threat model) | `PROJECT_README.md::Threat Model` |
| Тема 2 (Evasion) | Вне scope этого проекта (RAG не даёт градиентного доступа атакующему) |
| Тема 3 (Poisoning) | `attacks/prompt_injection.py`, `attacks/retrieval_poison.py` |
| Тема 4 (Backdoor) | `attacks/trigger_backdoor.py` — главный кейс. Прямой аналог BadNets. |
| Тема 5 (Physical) | Вне scope (нет камер) |
| Тема 6 (Defense) | `defenses/l1_sanitize.py`, `defenses/l2_embedding_anomaly.py`, `defenses/l4_strict_verifier.py`, raw verify_answer как baseline |

---

## Что развивать дальше (если будет время)

- **L3 (retrieval-time defense)** в `search/faiss_index.py::SearchService.search`:
  сравнить dense-score и rerank-score, если расхождение большое — chunk
  подозрителен.
- **Реальные эмбеддинги в offline-eval** — заменить `_pseudo_embed` в
  `eval/run_l1_l2_offline.py` на BGE-M3 (через `embeddings.bge_m3`), это
  даст осмысленную TPR L2.
- **Adaptive attack module** — `attacks/adaptive.py` с base64 / hidden
  comments / homoglyph encoding.
- **Activation clustering** для backdoor detection (тема 4 курса) —
  собрать активации последнего слоя для всех чанков класса, k-means,
  поиск «лишнего» микрокластера.
