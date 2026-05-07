"""Атакующая часть проекта (security research).

Цель — генерировать «отравленные» документы для прогона через тот же
ingestion pipeline, что использует обычный пользователь (загрузка через
admin UI / REST API). Это позволяет:

1. Замерить базовый Attack Success Rate (ASR) — насколько LLM-агент
   подвергается атаке без защит.
2. Проверить, ловят ли защитные слои L1/L2/L4 каждый из видов атак.
3. Сделать adaptive-attack: атакующий, знающий о защите X, пробует обойти.

Виды атак (соответствуют типологии темы 3 курса):

- prompt_injection: прямые «игнорируй всё» — самый базовый poisoning,
  должен ловиться L1.
- trigger_backdoor: BadNets-style — clean accuracy сохраняется,
  атака активируется только при триггере. Главный кейс проекта.
- citation_spoof: подделка inline-цитат `[file.pdf, стр. 23]` в тексте
  чанка для обмана верификатора.
- retrieval_poison: keyword-stuffing для попадания в top-k retrieval'а
  (упрощённый вариант clean-label poisoning без градиентной оптимизации).

Все генераторы возвращают пары (filename, text) — текст пишется в .txt
или .md, потому что эти форматы парсер обрабатывает напрямую без OCR
и без бинарной обвязки. Для DOCX/PDF можно использовать те же тексты,
но это уже не нужно для демонстрации.
"""
from attacks import prompt_injection, trigger_backdoor, citation_spoof, retrieval_poison

__all__ = [
    "prompt_injection",
    "trigger_backdoor",
    "citation_spoof",
    "retrieval_poison",
]
