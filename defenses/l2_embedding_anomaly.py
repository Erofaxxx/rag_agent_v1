"""L2: embedding-space anomaly detection (per-document).

Идея напрямую из темы 4 курса (occurrence в feature-space):
для документа считаем центроид всех его эмбеддингов и расстояние каждого
чанка до центроида. Чанки, которые выпадают из «облака темы» документа —
кандидаты на инъекцию (отравленный фрагмент обычно семантически отличается
от остального содержания файла).

Используем cosine distance на L2-нормализованных векторах:
    dist_i = 1 - <v_i, c> / (||v_i|| ||c||)

Затем z-score:
    z_i = (dist_i - mean(dist)) / std(dist)

Чанк помечается как аномалия, если z_i > z_threshold (по умолчанию 2.5).

Edge cases:
- Если в документе ≤ 2 чанка: статистика бессмысленна, помечаем все как
  not-anomaly (slipping defense, but no data to claim otherwise).
- Если std == 0: все чанки идентичны, ничего не помечаем.

Метод НЕ трогает FAISS-индекс: он выдаёт только флаги, а решение
(drop/warn) принимает caller (в нашем случае — api/documents.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnomalyReport:
    n_chunks: int
    n_flagged: int
    flags: list[bool]                # длиной n_chunks
    z_scores: list[float]            # длиной n_chunks
    threshold: float
    reason: str = ""                 # 'too_few_chunks' / 'zero_variance' / 'computed'

    def to_dict(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "n_flagged": self.n_flagged,
            "z_scores": [round(z, 3) for z in self.z_scores],
            "flags": self.flags,
            "threshold": self.threshold,
            "reason": self.reason,
        }


def detect_anomalies(
    vectors: np.ndarray,
    *,
    z_threshold: float = 2.5,
) -> AnomalyReport:
    """Считает аномалии для одного документа.

    Args:
        vectors: матрица (n_chunks, dim). Векторы ОЖИДАЮТСЯ нормализованными
                 на L2 (так делает embedding_service в этом проекте — см.
                 yandex.py / bge_m3.py). Если нет — нормализуем на лету.
        z_threshold: z-score, выше которого чанк считается аномалией.

    Returns:
        AnomalyReport.
    """
    if vectors is None or len(vectors) == 0:
        return AnomalyReport(
            n_chunks=0, n_flagged=0, flags=[], z_scores=[],
            threshold=z_threshold, reason="empty",
        )

    n = len(vectors)
    if n <= 2:
        return AnomalyReport(
            n_chunks=n,
            n_flagged=0,
            flags=[False] * n,
            z_scores=[0.0] * n,
            threshold=z_threshold,
            reason="too_few_chunks",
        )

    V = np.asarray(vectors, dtype=np.float32)

    # На всякий случай нормализуем (в проекте уже нормализованы embedding-сервисом,
    # но если кто-то прокинет сырые векторы — сами защитимся).
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    V_norm = V / norms

    centroid = V_norm.mean(axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm == 0:
        return AnomalyReport(
            n_chunks=n,
            n_flagged=0,
            flags=[False] * n,
            z_scores=[0.0] * n,
            threshold=z_threshold,
            reason="zero_centroid",
        )
    centroid = centroid / c_norm

    # cosine distance = 1 - cosine similarity
    sims = V_norm @ centroid
    dists = 1.0 - sims

    mean_d = float(dists.mean())
    std_d = float(dists.std())
    if std_d == 0.0:
        return AnomalyReport(
            n_chunks=n,
            n_flagged=0,
            flags=[False] * n,
            z_scores=[0.0] * n,
            threshold=z_threshold,
            reason="zero_variance",
        )

    z_scores = (dists - mean_d) / std_d
    flags = (z_scores > z_threshold).tolist()

    return AnomalyReport(
        n_chunks=n,
        n_flagged=int(sum(flags)),
        flags=flags,
        z_scores=z_scores.astype(float).tolist(),
        threshold=z_threshold,
        reason="computed",
    )


def short_summary(report: AnomalyReport) -> str:
    if report.reason != "computed":
        return f"L2: skipped ({report.reason}, n={report.n_chunks})"
    return f"L2: {report.n_flagged}/{report.n_chunks} flagged (z>{report.threshold})"
