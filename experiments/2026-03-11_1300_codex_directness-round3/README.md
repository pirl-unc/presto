# Directness Round 3

- Agent/model: `codex/gpt-5`
- Source artifacts: [`modal_runs/directness_bakeoff_round3`](/Users/iskander/code/presto/modal_runs/directness_bakeoff_round3)
- Dataset / curation:
  - 7-allele class-I panel
  - numeric_no_qualitative
  - qualifier_filter=all
  - warm-start mhc-pretrain-20260308b
- Conditions tested:
  - Q00..Q08 positional sweep
  - E00..E03 encoding sweep
- Takeaway: Position and target encoding both materially affect broad-contract performance.
- Local summary files:
  - [options_vs_perf.md](/Users/iskander/code/presto/experiments/2026-03-11_1300_codex_directness-round3/options_vs_perf.md)
  - [options_vs_perf.json](/Users/iskander/code/presto/experiments/2026-03-11_1300_codex_directness-round3/options_vs_perf.json)
  - [manifest.json](/Users/iskander/code/presto/experiments/2026-03-11_1300_codex_directness-round3/manifest.json)
  - [variants.md](/Users/iskander/code/presto/experiments/2026-03-11_1300_codex_directness-round3/variants.md)
