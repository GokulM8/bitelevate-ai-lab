# Baseline Report: Keyword Rule Classifier

- Dataset: `historical_tickets.csv` (100 tickets)
- Confidence threshold: 0.60 (below this -> `NEEDS_AI_INFERENCE`)

## Overall metrics

| Metric | Value |
|---|---|
| Accuracy (fallbacks count as wrong) | 78.0% (78/100) |
| Coverage (routed without AI) | 79.0% |
| Accuracy on routed tickets | 98.7% |
| Sent to AI (fallback) | 21 tickets |
| Macro precision / recall | 98.9% / 78.0% |

## Per-category metrics

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BILLING | 95.5% | 84.0% | 89.4% | 25 |
| AUTH | 100.0% | 80.0% | 88.9% | 25 |
| TECHNICAL | 100.0% | 72.0% | 83.7% | 25 |
| HARDWARE | 100.0% | 76.0% | 86.4% | 25 |

## Confusion matrix (rows = true, columns = predicted)

| true \ pred | BILLING | AUTH | TECHNICAL | HARDWARE | NEEDS_AI_INFERENCE |
|---|---|---|---|---|---|
| BILLING | 21 | 0 | 0 | 0 | 4 |
| AUTH | 0 | 20 | 0 | 0 | 5 |
| TECHNICAL | 0 | 0 | 18 | 0 | 7 |
| HARDWARE | 1 | 0 | 0 | 19 | 5 |
