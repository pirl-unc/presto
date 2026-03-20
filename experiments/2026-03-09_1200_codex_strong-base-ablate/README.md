# Strong Base Ablate

- Agent/model: `codex/gpt-5`
- Source artifacts: [`modal_runs/strong_base_ablate`](/Users/iskander/code/presto/modal_runs/strong_base_ablate)
- Dataset / curation:
  - 7-allele class-I panel
  - exact IC50
  - qualifier_filter=exact
  - warm-start mhc-pretrain-20260308b
- Conditions tested:
  - legacy_baseline
  - score_context
  - legacy_peprank
  - legacy_m1
  - legacy_m1_peprank
- Takeaway: legacy_m1 is the strongest exact-IC50 class-I baseline.
- Local summary files:
  - [options_vs_perf.md](/Users/iskander/code/presto/experiments/2026-03-09_1200_codex_strong-base-ablate/options_vs_perf.md)
