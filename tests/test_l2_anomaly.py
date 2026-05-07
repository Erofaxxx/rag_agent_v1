"""Юнит-тесты для defenses/l2_embedding_anomaly.py.

Идея: подаём синтетические векторы — кластер похожих + явный outlier.
Проверяем, что outlier помечается флагом, а внутрикластерные точки — нет.

Никакого embedding-сервиса не нужно: работаем с numpy напрямую.
"""
from __future__ import annotations

import numpy as np
import pytest

from defenses.l2_embedding_anomaly import detect_anomalies


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def _make_cluster(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Кластер из N точек вокруг общего центра — имитация чанков одного
    документа на одну тему."""
    rng = np.random.default_rng(seed)
    center = rng.normal(0, 1, dim)
    noise = rng.normal(0, 0.05, (n, dim))   # small intra-cluster spread
    return _l2_normalize(center + noise)


def _make_outlier(dim: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Берём явно другое направление
    return _l2_normalize(rng.normal(0, 1, (1, dim)))


class TestDetectAnomalies:
    def test_too_few_chunks_returns_no_flags(self):
        v = np.random.randn(2, 8).astype(np.float32)
        report = detect_anomalies(v)
        assert report.reason == "too_few_chunks"
        assert report.n_flagged == 0
        assert report.flags == [False, False]

    def test_empty_input(self):
        v = np.zeros((0, 8))
        report = detect_anomalies(v)
        assert report.reason == "empty"
        assert report.n_chunks == 0

    def test_homogeneous_cluster_no_flags(self):
        # 10 почти-одинаковых векторов: std ≈ 0 → ничего не помечаем.
        cluster = _make_cluster(n=10, dim=64, seed=42)
        report = detect_anomalies(cluster, z_threshold=2.5)
        # либо zero_variance, либо computed с n_flagged=0
        assert report.n_flagged == 0

    def test_clear_outlier_is_flagged(self):
        cluster = _make_cluster(n=15, dim=64, seed=42)
        outlier = _make_outlier(dim=64, seed=99)
        all_vecs = np.vstack([cluster, outlier])

        report = detect_anomalies(all_vecs, z_threshold=2.0)
        assert report.reason == "computed"
        # Outlier — последний по индексу
        assert report.flags[-1] is True
        # Внутрикластерные не должны быть помечены
        assert sum(report.flags[:-1]) == 0

    def test_threshold_controls_sensitivity(self):
        cluster = _make_cluster(n=15, dim=64, seed=42)
        outlier = _make_outlier(dim=64, seed=99)
        all_vecs = np.vstack([cluster, outlier])

        loose = detect_anomalies(all_vecs, z_threshold=10.0)
        # Слишком высокий порог → ничего не ловим
        assert loose.n_flagged == 0

        strict = detect_anomalies(all_vecs, z_threshold=1.0)
        # Очень низкий порог → outlier и потенциально ещё что-то
        assert strict.n_flagged >= 1

    def test_multiple_outliers(self):
        cluster = _make_cluster(n=12, dim=64, seed=42)
        out1 = _make_outlier(dim=64, seed=99)
        out2 = _make_outlier(dim=64, seed=123)
        out3 = _make_outlier(dim=64, seed=200)
        all_vecs = np.vstack([cluster, out1, out2, out3])

        report = detect_anomalies(all_vecs, z_threshold=1.5)
        # Хотя бы один outlier должен сработать
        assert report.n_flagged >= 1

    def test_z_scores_consistent_with_flags(self):
        cluster = _make_cluster(n=15, dim=64, seed=42)
        outlier = _make_outlier(dim=64, seed=99)
        all_vecs = np.vstack([cluster, outlier])

        report = detect_anomalies(all_vecs, z_threshold=2.0)
        # Где flag=True, z_score должен быть > порога
        for f, z in zip(report.flags, report.z_scores):
            if f:
                assert z > 2.0
