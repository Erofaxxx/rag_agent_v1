"""Защитные слои для RAG-агента (security research).

Все слои спроектированы так, чтобы:
1. По умолчанию быть выключенными (флаги в config.py).
2. Не падать при нехватке данных (если документ ровно из 1 чанка — L2
   возвращает 'не аномалия' для всех).
3. Не зависеть от сети / LLM API в core-логике (L1 — чистые регексы,
   L2 — numpy). LLM используется только в L4 как опциональный второй
   уровень.
4. Возвращать структурированный отчёт, который складывается в audit_log
   и виден из admin UI.

Слои:
- L1 (l1_sanitize): регексы + эвристики на тексте чанка (ingest-time).
- L2 (l2_embedding_anomaly): per-document z-score по cosine-расстоянию
  до центроида документа (ingest-time).
- L3 (l3_query_ablation): leave-one-word-out detection of trigger-activated
  chunks. Generic, model-agnostic, training-free. Ловит trigger-based
  backdoor-атаки независимо от шаблона атаки. Запускается в момент search
  (query-time). Без LLM-вызовов.
- L4 (l4_strict_verifier): расширение существующего verify_answer
  (детектит инъекции внутри cited chunks).
"""
from defenses import l1_sanitize, l2_embedding_anomaly, l3_query_ablation, l4_strict_verifier

__all__ = ["l1_sanitize", "l2_embedding_anomaly", "l3_query_ablation", "l4_strict_verifier"]
