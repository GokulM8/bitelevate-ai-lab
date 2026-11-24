#!/usr/bin/env python3
"""Benchmark the keyword baseline on labeled historical tickets.

Usage:
    python scripts/eval_baseline.py [path/to/tickets.csv] [--report BASELINE_REPORT.md]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.rule_classifier import (  # noqa: E402
    CATEGORIES, CONFIDENCE_THRESHOLD, NEEDS_AI_INFERENCE, classify_ticket,
)

CANDIDATES = [
    "data/historical_tickets.csv", "data/tickets.csv", "data/historical_tickets.json",
    "tests/data/historical_tickets.csv", "scripts/historical_tickets.csv",
]
LABEL_ALIASES = {"AUTHENTICATION": "AUTH", "LOGIN": "AUTH", "PAYMENT": "BILLING"}


def find_dataset(arg):
    if arg:
        return Path(arg)
    for candidate in CANDIDATES:
        if (ROOT / candidate).exists():
            return ROOT / candidate
    sys.exit("No dataset found. Pass a CSV/JSON path: python scripts/eval_baseline.py <file>")


def load_tickets(path):
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    tickets = []
    for row in rows:
        text = row.get("text") or " ".join(filter(None, [row.get("subject"), row.get("content")]))
        label = str(row.get("category") or row.get("label") or "").strip().upper()
        label = LABEL_ALIASES.get(label, label)
        if text.strip() and label:
            tickets.append((text, label))
    return tickets


def evaluate(tickets, threshold):
    confusion = {}
    for text, true in tickets:
        pred = classify_ticket(text, threshold=threshold).category
        confusion.setdefault(true, Counter())[pred] += 1

    def cell(true, pred):
        return confusion.get(true, Counter())[pred]

    n = len(tickets)
    correct = sum(cell(c, c) for c in CATEGORIES)
    fallbacks = sum(cell(t, NEEDS_AI_INFERENCE) for t in confusion)
    covered = n - fallbacks

    per_class = {}
    for c in CATEGORIES:
        tp = cell(c, c)
        predicted = sum(cell(t, c) for t in confusion)
        actual = sum(confusion.get(c, Counter()).values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = (precision, recall, f1, actual)

    return {
        "n": n, "correct": correct, "accuracy": correct / n if n else 0.0,
        "coverage": covered / n if n else 0.0,
        "covered_accuracy": correct / covered if covered else 0.0,
        "fallbacks": fallbacks, "per_class": per_class,
        "confusion": confusion, "cell": cell,
    }


def to_markdown(m, dataset_name, threshold):
    cols = list(CATEGORIES) + [NEEDS_AI_INFERENCE]
    extra_true = sorted(set(m["confusion"]) - set(CATEGORIES))
    rows = list(CATEGORIES) + extra_true
    macro_p = sum(v[0] for v in m["per_class"].values()) / len(CATEGORIES)
    macro_r = sum(v[1] for v in m["per_class"].values()) / len(CATEGORIES)

    out = [
        "# Baseline Report: Keyword Rule Classifier", "",
        f"- Dataset: `{dataset_name}` ({m['n']} tickets)",
        f"- Confidence threshold: {threshold:.2f} (below this -> `{NEEDS_AI_INFERENCE}`)", "",
        "## Overall metrics", "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy (fallbacks count as wrong) | {m['accuracy']:.1%} ({m['correct']}/{m['n']}) |",
        f"| Coverage (routed without AI) | {m['coverage']:.1%} |",
        f"| Accuracy on routed tickets | {m['covered_accuracy']:.1%} |",
        f"| Sent to AI (fallback) | {m['fallbacks']} tickets |",
        f"| Macro precision / recall | {macro_p:.1%} / {macro_r:.1%} |", "",
        "## Per-category metrics", "",
        "| Category | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for c, (p, r, f1, support) in m["per_class"].items():
        out.append(f"| {c} | {p:.1%} | {r:.1%} | {f1:.1%} | {support} |")
    out += ["", "## Confusion matrix (rows = true, columns = predicted)", "",
            "| true \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for t in rows:
        out.append(f"| {t} | " + " | ".join(str(m["cell"](t, c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--report", help="write the markdown report to this path")
    args = parser.parse_args()

    path = find_dataset(args.dataset)
    tickets = load_tickets(path)
    if not tickets:
        sys.exit(f"No labeled tickets found in {path}")
    metrics = evaluate(tickets, args.threshold)
    report = to_markdown(metrics, path.name, args.threshold)
    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
