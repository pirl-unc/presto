# Historical evidence storage

The original source commit and every previously tracked file are preserved in
`reproduce/history.tar.gz`, with a SHA-256 inventory in `reproduce/history.json`.
Verbose diagnostics and repeated frozen runtimes are stored only in that archive.
From the repository root, restore them without replacing changed files:

```bash
python scripts/restore_experiment.py experiments/2026-11-17_2330_codex_affinity-conditioned-processing/reproduce/history.json
```

The experiment README and decisions retain their original contents. Raw model
tensors and untracked per-example dumps remain at their documented artifact paths.
Current work uses the installed package, not these frozen runtimes.
