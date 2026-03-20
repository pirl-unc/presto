# Reproducibility Bundle

- Source script: `experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py`
- Launch command: `launch.sh`
- Launch metadata: `launch.json`
- Launcher snapshot: `source/` (if the source script existed at bundle creation time)
- Local resume command: `local_resume.sh`

These files freeze the launch invocation and environment for this experiment family so later agents can rerun or extend it without relying on current launcher defaults.

`launch.sh` preserves the original Modal launch contract. `local_resume.sh` is the portable non-Modal resume wrapper for CPU / Apple Silicon execution on another machine.

For safety, `local_resume.sh` defaults to `cpu`. Override `PRESTO_LOCAL_DEVICE=mps` if you want to try Apple Silicon acceleration explicitly.
