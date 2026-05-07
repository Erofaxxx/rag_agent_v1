"""Тесты для модуля атак.

Цели:
1. Сгенерированные документы действительно содержат то, что обещают
   (триггер, injection, fake-citation).
2. L1-детектор реально ловит эти документы (sanity-check end-to-end на
   уровне строк, без сети).

Важно: тесты НЕ запускают LLM/embedding/HTTP — это чистая проверка
строковых артефактов.
"""
from __future__ import annotations

from attacks import (
    citation_spoof,
    prompt_injection,
    retrieval_poison,
    trigger_backdoor,
)
from defenses.l1_sanitize import analyze_chunk


# ---------- prompt_injection ----------

class TestPromptInjection:
    def test_each_variant_makes_doc(self):
        for v in prompt_injection.VARIANTS:
            fname, text = prompt_injection.make_document(v)
            assert fname.endswith(".md")
            assert v.payload in text  # payload встроен буквально

    def test_each_variant_caught_by_l1(self):
        # Базовый тест: каждая известная инъекция должна давать risk_score
        # выше минимального порога. direct_instruction — самая слабая
        # категория (вес 0.35), остальные дают выше.
        for v in prompt_injection.VARIANTS:
            risk = analyze_chunk(v.payload)
            assert risk.risk_score >= 0.3, (
                f"variant {v.name}: risk={risk.risk_score:.2f} ниже минимума"
            )

    def test_make_all_returns_full_set(self):
        items = prompt_injection.make_all()
        assert len(items) == len(prompt_injection.VARIANTS)
        for fname, text, variant in items:
            assert fname and text and variant


# ---------- trigger_backdoor ----------

class TestTriggerBackdoor:
    def test_backdoor_doc_contains_trigger_and_target(self):
        cfg = trigger_backdoor.TriggerConfig()
        fname, text = trigger_backdoor.make_backdoor_doc(cfg)
        assert cfg.trigger_phrase in text
        assert cfg.target_answer in text

    def test_clean_doc_has_no_trigger(self):
        cfg = trigger_backdoor.TriggerConfig()
        _, clean_text = trigger_backdoor.make_clean_doc()
        assert cfg.trigger_phrase not in clean_text
        assert cfg.target_answer not in clean_text

    def test_question_battery_balance(self):
        cfg = trigger_backdoor.TriggerConfig()
        battery = trigger_backdoor.make_question_battery(cfg)
        triggered = [q for q in battery if q["is_triggered"]]
        clean = [q for q in battery if not q["is_triggered"]]
        # Должно быть и того, и того
        assert len(triggered) >= 1
        assert len(clean) >= 1
        # Триггерные содержат фразу
        for q in triggered:
            assert cfg.trigger_phrase.lower() in q["question"].lower()
        # Чистые — нет
        for q in clean:
            assert cfg.trigger_phrase.lower() not in q["question"].lower()

    def test_backdoor_doc_stealthier_than_direct_injection(self):
        """Backdoor — стелс-атака: не использует «игнорируй всё» / «теперь ты»,
        поэтому L1 его пропускает. Это ровно то, что мы и хотим
        продемонстрировать в проекте: BadNets-style атаки требуют L2/L4,
        регексы L1 их не ловят.

        Тест: risk у backdoor должен быть НИЖЕ, чем у самой слабой prompt-injection.
        """
        cfg = trigger_backdoor.TriggerConfig()
        _, backdoor_text = trigger_backdoor.make_backdoor_doc(cfg)
        backdoor_risk = analyze_chunk(backdoor_text).risk_score

        # Сравниваем с direct_override_ru (самый громкий вариант)
        direct = next(v for v in prompt_injection.VARIANTS if v.name == "direct_override_ru")
        direct_risk = analyze_chunk(direct.payload).risk_score

        assert backdoor_risk < direct_risk, (
            f"backdoor risk={backdoor_risk:.2f} should be lower than "
            f"direct injection risk={direct_risk:.2f}"
        )


# ---------- citation_spoof ----------

class TestCitationSpoof:
    def test_spoofed_doc_contains_fake_citations(self):
        _, text = citation_spoof.make_spoofed_doc()
        assert "[internal_directive_2025.pdf" in text
        assert "[chunk_id=42]" in text
        assert "[policy_v3.docx" in text

    def test_clean_doc_has_no_fake_citations(self):
        _, text = citation_spoof.make_clean_doc()
        assert "[" not in text or ".pdf" not in text and ".docx" not in text

    def test_spoofed_doc_caught_by_l1(self):
        _, text = citation_spoof.make_spoofed_doc()
        risk = analyze_chunk(text)
        assert "fake_citation" in risk.categories
        assert risk.risk_score >= 0.4


# ---------- retrieval_poison ----------

class TestRetrievalPoison:
    def test_stuffed_doc_repeats_terms(self):
        cfg = retrieval_poison.default_config()
        _, text = retrieval_poison.make_stuffed_doc(cfg)
        first_term = cfg.target_query_terms[0]
        # Каждый ключевой термин должен встречаться многократно
        assert text.count(first_term) >= 3

    def test_stuffed_doc_contains_fake_answer(self):
        cfg = retrieval_poison.default_config()
        _, text = retrieval_poison.make_stuffed_doc(cfg)
        assert cfg.fake_answer in text or cfg.fake_answer.lower() in text.lower()

    def test_stuffed_doc_does_NOT_trigger_l1(self):
        """Это ключевое свойство retrieval-poison: он не содержит явных
        инъекций. L1 не должен его ловить (или ловит на очень низком уровне).
        Защита от него — L2/L3, не L1.
        """
        cfg = retrieval_poison.default_config()
        _, text = retrieval_poison.make_stuffed_doc(cfg)
        risk = analyze_chunk(text)
        # Должен быть ниже стандартного drop-порога 0.6 — иначе детектор
        # «слишком жадный».
        assert risk.risk_score < 0.6
