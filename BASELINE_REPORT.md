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

## Analysis

**Where the rules work.** When the rules commit to a category they are almost always right: 98.7% accuracy on the 79 routed tickets, and 100% precision for AUTH, TECHNICAL and HARDWARE. Tickets with strong keywords (invoice, password, crash, printer) are routed instantly at zero cost.

**Where they fail.** There were 22 errors out of 100 tickets:
- 20 tickets contained no keyword at all (confidence 0.0) and were correctly sent to the AI instead of being guessed. They use everyday phrasing, e.g. "Notifications are not working" (TECHNICAL), "Cannot access my account" (AUTH), "Money was taken from my account twice" (BILLING), "Touchpad is not responding" (HARDWARE).
- 1 ticket had mixed signals and fell below the threshold: "Receipt printer jams on every second print" (0.53; "receipt" points to BILLING, "printer" to HARDWARE).
- 1 ticket was misrouted with high confidence: "Tablet will not charge" (HARDWARE) was routed to BILLING at 0.80 because "charge" is a billing keyword. Keyword rules cannot use context, and a confident wrong answer is worse than a fallback.
- Word forms are not matched: "Upload keeps failing" was missed because the map contains "fails" and "failed" but not "failing" (no stemming).

**Fallback economics.** 21% of tickets need the AI model; 79% are routed in about 1 ms with no model call. Counting fallbacks as wrong (78% accuracy) understates the baseline and ignoring them (98.7%) overstates it; the useful numbers are the pair: 79% coverage at 98.7% accuracy.

**Expected AI performance (target, not measured).** The ML/LLM model should handle the 21 fallback tickets, especially TECHNICAL and HARDWARE phrasing without keywords, and resolve context errors like "charge" in a hardware sentence. A reasonable goal is to beat this baseline's 78% overall accuracy on the same 100-ticket set while sending fewer tickets to manual review.

**Next steps (not applied, to keep this baseline honest).** Add stemming or phrase rules ("not charging", "cannot access"), add missing device and phrasing keywords, and add a guard so "charge" is ignored when device words appear. Any tuning should be evaluated on a separate held-out set, because tuning on this set would inflate its numbers. Then compare the embedding-based classifier on the same dataset.

## Dataset caveat

The repo did not include historical tickets, so the 100 tickets in `data/historical_tickets.csv` are a synthetic, hand-written set (25 per category, including deliberately ambiguous and keyword-free tickets). It was written by the same people who wrote the keyword maps, so these numbers are optimistic. Real tickets would likely score lower. Use this report to compare the AI model against the baseline on the same set, not as a prediction of production accuracy.
