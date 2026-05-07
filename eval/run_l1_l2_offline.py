"""Офлайн-проверка L1+L2 на сгенерированном корпусе.

Не требует ни сети, ни Yandex/OpenRouter API ключей. Эмбеддинги для L2
делаются через локальную псевдо-модель (TF-IDF + случайная проекция),
этого достаточно, чтобы убедиться: outlier-документы выпадают из
тематического кластера. ДЛЯ ПРОДАКШЕНА — использовать настоящие
эмбеддинги (BGE-M3 / Yandex), здесь — только для unit-eval.

Запускать:
    python eval/generate_corpus.py
    python eval/run_l1_l2_offline.py

Что выводит:
- Per-document отчёт L1 (категории, risk, action)
- Per-document отчёт L2 (n_flagged / n_chunks)
- Summary: TPR/FPR детекторов на текущем корпусе
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np

from defenses.l1_sanitize import sanitize_chunks
from defenses.l2_embedding_anomaly import detect_anomalies


# ----- Простой chunker без зависимости от config/pydantic -----
# Для offline-eval нам не нужен production-чанкер; режем по двойному переводу
# строки или 800 символов — что наступит раньше.

def simple_chunk(text: str, max_chars: int = 800) -> list[dict]:
    if not text:
        return []
    chunks: list[dict] = []
    paragraphs = re.split(r"\n\s*\n", text)
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append({"text": buf})
            if len(p) > max_chars:
                # ну ладно, режем грубо
                for i in range(0, len(p), max_chars):
                    chunks.append({"text": p[i : i + max_chars]})
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append({"text": buf})
    return chunks


# ----- Псевдо-эмбеддинг (без сети) -----
# Для нормальной демонстрации L2 используйте реальные эмбеддинги; здесь
# простой character-ngram → hash → проекция, достаточный, чтобы тема
# документа была отделима от прямой инъекции.

def _pseudo_embed(text: str, dim: int = 128, seed: int = 42) -> np.ndarray:
    """Char-ngram bag-of-words с фиксированным хеш-вокабом + L2 нормировка.
    Это «хорошие-плохие тексты различаются» — достаточно для офлайн-демо."""
    rng = np.random.default_rng(seed)
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    text = text.lower()
    # Биграммы из слов
    tokens = re.findall(r"\w+", text)
    for i, tok in enumerate(tokens):
        h = hash(tok) % dim
        vec[h] += 1.0
        if i + 1 < len(tokens):
            h2 = hash(tok + tokens[i + 1]) % dim
            vec[h2] += 0.5
    # Симметризация шумом, чтобы хеш-коллизии не доминировали
    vec += rng.normal(0, 0.01, dim).astype(np.float32)
    n = float(np.linalg.norm(vec))
    return (vec / n) if n > 0 else vec


def _embed_chunks(chunks: list[dict]) -> np.ndarray:
    return np.vstack([_pseudo_embed(c["text"]) for c in chunks])


# ----- Eval -----

def _is_poisoned(doc_path: Path) -> bool:
    return "poisoned_corpus" in doc_path.parts or doc_path.name.startswith("poison_") \
        or "spoofed" in doc_path.name or "stuffed" in doc_path.name \
        or doc_path.name == "regulation_financial_limits_v2.md"


def evaluate_doc(
    doc_path: Path,
    *,
    l1_mode: str = "warn",
    l1_threshold: float = 0.6,
    l2_threshold: float = 2.5,
) -> dict:
    text = doc_path.read_text(encoding="utf-8")
    chunks = simple_chunk(text)

    # L1
    kept, l1_report = sanitize_chunks(chunks, mode=l1_mode, threshold=l1_threshold)
    l1_max_risk = max((r["risk_score"] for r in l1_report), default=0.0)
    l1_warn = sum(1 for r in l1_report if r["action"] == "warn")
    l1_drop = sum(1 for r in l1_report if r["action"] == "drop")

    # L2
    vectors = _embed_chunks(chunks)
    l2_report = detect_anomalies(vectors, z_threshold=l2_threshold)

    poisoned = _is_poisoned(doc_path)
    detected = (l1_max_risk >= l1_threshold) or (l2_report.n_flagged > 0)

    return {
        "file": doc_path.name,
        "is_poisoned": poisoned,
        "n_chunks": len(chunks),
        "l1_max_risk": round(l1_max_risk, 3),
        "l1_warn": l1_warn,
        "l1_drop": l1_drop,
        "l1_categories": sorted({c for r in l1_report for c in r.get("categories", [])}),
        "l2_n_flagged": l2_report.n_flagged,
        "l2_reason": l2_report.reason,
        "detected": detected,
    }


def evaluate_corpus(corpus: Path, **kwargs) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(corpus.glob("*.md")):
        rows.append(evaluate_doc(p, **kwargs))
    return rows


def print_report(rows: list[dict], threshold: float) -> None:
    width_name = max((len(r["file"]) for r in rows), default=20)
    print(f"\n{'='*100}")
    print(f"{'file'.ljust(width_name)}  poisoned  L1_risk  L1w/d  L2flag  detected  categories")
    print("-" * 100)
    for r in rows:
        cats = ",".join(r["l1_categories"]) or "-"
        print(
            f"{r['file'].ljust(width_name)}  "
            f"{'YES' if r['is_poisoned'] else 'no':>8}  "
            f"{r['l1_max_risk']:>6.2f}  "
            f"{r['l1_warn']}/{r['l1_drop']:<3}  "
            f"{r['l2_n_flagged']:>5}  "
            f"{'YES' if r['detected'] else 'no':>8}  "
            f"{cats}"
        )

    # Confusion matrix
    tp = sum(1 for r in rows if r["is_poisoned"] and r["detected"])
    fn = sum(1 for r in rows if r["is_poisoned"] and not r["detected"])
    fp = sum(1 for r in rows if not r["is_poisoned"] and r["detected"])
    tn = sum(1 for r in rows if not r["is_poisoned"] and not r["detected"])

    n_pos = tp + fn
    n_neg = fp + tn
    tpr = (tp / n_pos) if n_pos else 0.0
    fpr = (fp / n_neg) if n_neg else 0.0

    print("-" * 100)
    print(f"L1 threshold: {threshold}")
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"TPR (Recall на отравленных) = {tpr:.2%}")
    print(f"FPR (False positive rate)   = {fpr:.2%}")
    print("=" * 100 + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="benchmarks", help="benchmarks dir")
    p.add_argument("--l1-threshold", type=float, default=0.6)
    p.add_argument("--l2-threshold", type=float, default=2.5)
    args = p.parse_args()

    base = Path(args.corpus)
    if not base.exists():
        raise SystemExit(f"Корпус не найден: {base}. Запустите generate_corpus.py.")

    rows = []
    for sub in ("clean_corpus", "poisoned_corpus"):
        d = base / sub
        if d.exists():
            rows.extend(evaluate_corpus(
                d,
                l1_mode="warn",
                l1_threshold=args.l1_threshold,
                l2_threshold=args.l2_threshold,
            ))

    print_report(rows, args.l1_threshold)


if __name__ == "__main__":
    main()
