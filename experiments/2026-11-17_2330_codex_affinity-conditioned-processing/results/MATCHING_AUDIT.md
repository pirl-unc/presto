# Matching audit before model training

Both-class audit of immutable existing candidate pools. All ranks are BA percentile points.

| Split | Class | Rule | Retained | BA-only AUROC | BA pair win | Median positive nM |
|---|---|---|---:|---:|---:|---:|
| validation | I | legacy_nm | 340 | 0.4993 | 0.4912 | 6158.3 |
| validation | II | legacy_nm | 158 | 0.5015 | 0.5443 | 611.8 |
| validation | I | rank_0.5_all | 306 | 0.5565 | 0.6438 | 316.2 |
| validation | II | rank_0.5_all | 140 | 0.5023 | 0.5571 | 551.0 |
| validation | I | rank_hybrid_all | 209 | 0.5033 | 0.5359 | 923.0 |
| validation | II | rank_hybrid_all | 137 | 0.5016 | 0.5474 | 577.8 |
| train | I | legacy_nm | 1197 | 0.5011 | 0.5313 | 3100.7 |
| train | II | legacy_nm | 658 | 0.5015 | 0.5479 | 507.5 |
| train | I | rank_0.5_all | 1713 | 0.5714 | 0.6562 | 174.2 |
| train | II | rank_0.5_all | 559 | 0.5045 | 0.5716 | 237.0 |
| train | I | rank_hybrid_all | 1087 | 0.5086 | 0.5547 | 578.4 |
| train | II | rank_hybrid_all | 526 | 0.5021 | 0.5542 | 311.7 |

Primary choice: hybrid all-candidate rule, chosen using TRAIN balance and coverage. Residual within-pair BA imbalance remains. Pooled AUROC near.5 alone does not establish perfect affinity balance. No new model predictive gains have been measured.

Full native training census:
- I: 209695 positives / 159 studies; 170339 with both source flanks known; 160372 monoallelic.
- II: 178179 positives / 42 studies; 149939 with both source flanks known; 177734 monoallelic.
