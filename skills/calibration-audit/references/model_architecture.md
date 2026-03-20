# Complaints Multi-Task Model Architecture

## Overview

The model is a multi-task transformer using DeBERTa-v3-base as the shared encoder backbone. All heads share the same [CLS] token representation.

## Architecture

```
Input Text → DeBERTa-v3-base Encoder → [CLS] pooling → Dropout(0.1)
                                                          │
                                    ┌──────────────────────┼──────────────────────┐
                                    │                      │                      │
                              Binary Heads           Multiclass Heads      Regression Head
                              (Linear→1)             (Linear→N)           (Linear→1)
                                    │                      │                      │
                           BCEWithLogitsLoss         CrossEntropyLoss       SmoothL1Loss
```

## Heads

### Binary Heads (sigmoid output)
- `is_complaint` — trained with `pos_weight` (class imbalance correction)
- `test1pass` through `test5pass` — five regulatory test outcomes, trained with per-class `pos_weight` tensor
- `vulnerability_detected` — trained with `pos_weight`

`pos_weight` inflates the loss contribution of positive examples to handle class imbalance. This means the model's raw sigmoid outputs are **not** well-calibrated probabilities — they're shifted toward predicting positive more aggressively.

### Multiclass Heads (softmax output)
- `complaint_outcome` — e.g., "upheld", "rejected", "referred"
- `sentiment` — e.g., "negative", "neutral", "positive"
- `tone` — e.g., "frustrated", "neutral", "threatening"
- `severity_level` — e.g., "low", "medium", "high", "critical"

### Regression Head
- `conduct_score` — continuous 0–100 score, normalized to 0–1 during training, scaled back at eval

## Training Details
- Optimizer: AdamW
- Scheduler: linear warmup
- Max sequence length: 512 tokens
- Loss: unweighted sum of all head losses
- Threshold: 0.5 for all binary heads (never tuned per head)

## Things Worth Thinking About

- The loss for each binary head is an unweighted sum — how might that affect heads with different class balances?
- `pos_weight` changes the loss gradient. What effect might that have on the model's output distribution?
- The threshold was set to 0.5 across all heads. Is there any reason to believe 0.5 is the right value for all of them?
