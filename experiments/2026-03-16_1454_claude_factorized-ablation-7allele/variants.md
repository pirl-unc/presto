# Factorized Ablation — 8 Conditions

## C1: d128 A07(fac+seg) full pretrain=yes enc=mhcflurry

- App: `ap-h9LFGNDBJ0lz8PocdofQXX`
- Run: `fac-ablation-c1-20260316a`
- Pretrain: True
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c1-20260316a.log`

## C2: d128 A03(seg only) full pretrain=yes enc=mhcflurry

- App: `ap-aY3L1fCTFCM7QWT0FuBbeR`
- Run: `fac-ablation-c2-20260316a`
- Pretrain: True
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c2-20260316a.log`

## C3: d128 A07(fac+seg) full pretrain=no enc=mhcflurry

- App: `ap-9MPHXii2oOLS8g1ADW7JOO`
- Run: `fac-ablation-c3-20260316a`
- Pretrain: False
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c3-20260316a.log`

## C4: d128 A07(fac+seg) assay_heads_only pretrain=yes enc=mhcflurry

- App: `ap-1Q0hGfWwPkoP6qbMSWVM9h`
- Run: `fac-ablation-c4-20260316a`
- Pretrain: True
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c4-20260316a.log`

## C5: d128 A07(fac+seg) full pretrain=yes enc=log10

- App: `ap-23QS1X68qbuhPccPxEqKPT`
- Run: `fac-ablation-c5-20260316a`
- Pretrain: True
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c5-20260316a.log`

## C6: d32 A07(fac+seg) full pretrain=no enc=mhcflurry

- App: `ap-WSESPoYSsI5CzkBYPfcfC7`
- Run: `fac-ablation-c6-20260316a`
- Pretrain: False
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c6-20260316a.log`

## C7: d32 A03(seg only) full pretrain=no enc=mhcflurry

- App: `ap-SbPEzBsRLyUGZzw7H86p7o`
- Run: `fac-ablation-c7-20260316a`
- Pretrain: False
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c7-20260316a.log`

## C8: d128 pooled_single full pretrain=yes enc=mhcflurry [NEG CTRL]

- App: `ap-cZo5a7JuZOV872jiAh4XaV`
- Run: `fac-ablation-c8-20260316a`
- Pretrain: True
- Launch log: `/Users/iskander/code/presto/experiments/2026-03-16_1454_claude_factorized-ablation-7allele/launch_logs/fac-ablation-c8-20260316a.log`

## Pairwise Comparisons

| Question | Comparison | Expected |
|----------|------------|----------|
| Factorized helps? | C1 vs C2, C6 vs C7 | Modest improvement |
| Pretraining helps? | C1 vs C3 | Yes, ~0.01-0.03 Spearman |
| d=128 vs d=32? | C1 vs C6 | d=128 wins |
| full vs assay_heads_only? | C1 vs C4 | full slightly better |
| mhcflurry vs log10? | C1 vs C5 | mhcflurry slightly better |
| pooled collapses? | C8 vs all | Yes, Spearman ~0.02 |
