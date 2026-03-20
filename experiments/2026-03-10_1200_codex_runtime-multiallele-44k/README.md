# Runtime Multiallele 44k

- Agent/model: `codex/gpt-5`
- Source artifacts: [`modal_runs/runtime_multiallele_44k`](/Users/iskander/code/presto/modal_runs/runtime_multiallele_44k)
- Dataset / curation:
  - 7-allele class-I panel
  - all binding rows
  - qualifier_filter=all
  - ranking always on
- Conditions tested:
  - R00..R15 runtime knob sweep
- Takeaway: compute dominates; TF32 helps; worker-heavy variants hurt.
- Local summary files:
  - [options_vs_perf_from_logs.md](/Users/iskander/code/presto/experiments/2026-03-10_1200_codex_runtime-multiallele-44k/options_vs_perf_from_logs.md)
  - [collected_from_app_logs.json](/Users/iskander/code/presto/experiments/2026-03-10_1200_codex_runtime-multiallele-44k/collected_from_app_logs.json)
  - [manifest.json](/Users/iskander/code/presto/experiments/2026-03-10_1200_codex_runtime-multiallele-44k/manifest.json)
