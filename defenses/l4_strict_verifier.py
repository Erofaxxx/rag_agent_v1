"""L4: расширение существующего verify_answer.

Текущий llm/verifier.py проверяет «утверждения в ответе vs найденные чанки»,
но не смотрит, есть ли в самих cited chunks инъекция. Это критическая дыра:
если триггер активировался и LLM послушно вывела «по протоколу 7-альфа лимит
снят» — verifier увидит, что эта фраза есть в чанке (формально подтверждено),
и ответит verified=True.

Этот модуль добавляет вторую проверку: пропускает каждый cited chunk через
L1-детектор и, если в нём найден injection-паттерн, помечает ответ как
suspicious независимо от результата фактической сверки.

Никакого LLM-вызова L4 не делает (зависит только от L1 регексов) — это
дёшево и быстро.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from defenses.l1_sanitize import analyze_chunk


@dataclass
class StrictReport:
    suspicious: bool
    findings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"suspicious": self.suspicious, "findings": self.findings}


def inspect_cited_chunks(
    cited_chunks: list[dict[str, Any]],
    *,
    risk_threshold: float = 0.4,
) -> StrictReport:
    """Проверяет каждый cited chunk на injection-паттерны.

    cited_chunks — формат из llm/agent.py::_collect_cited:
        {"chunk_id", "document_id", "filename", "snippet", ...}

    Возвращает suspicious=True если хотя бы один chunk выше порога.
    Threshold по умолчанию ниже, чем для L1-drop (0.5 vs 0.6) — тут нам важна
    чувствительность, не precision.
    """
    findings: list[dict] = []
    for c in cited_chunks:
        snippet = c.get("snippet") or ""
        if not snippet:
            continue
        risk = analyze_chunk(snippet)
        if risk.risk_score >= risk_threshold:
            findings.append({
                "chunk_id": c.get("chunk_id"),
                "filename": c.get("filename"),
                "risk_score": round(risk.risk_score, 3),
                "categories": risk.categories,
                "matched_phrases": risk.matched_phrases[:5],
            })

    return StrictReport(suspicious=bool(findings), findings=findings)


def build_warning(report: StrictReport) -> str:
    """Текст предупреждения, который добавляется в конец ответа агента.

    Стиль повторяет append_verification_warning из llm/verifier.py для
    консистентности UX.
    """
    if not report.suspicious:
        return ""
    files = sorted({f["filename"] for f in report.findings if f.get("filename")})
    files_str = ", ".join(files) if files else "источники"
    return (
        "\n\n⚠️ **Предупреждение безопасности (L4):** обнаружены подозрительные "
        f"паттерны в источниках ({files_str}) — возможная prompt-injection. "
        f"Найдено: {len(report.findings)} чанк(а/ов). Проверьте содержимое "
        "цитируемых документов перед использованием ответа."
    )
