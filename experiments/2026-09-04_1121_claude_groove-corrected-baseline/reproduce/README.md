# Reproducibility bundle

- `launch.sh` is the intended launcher invocation.
- `launch.json` records the prepared invocation and current git state.
- `source/launch.py` is the frozen launcher snapshot prepared with this remediation.

The launcher itself performs the mandatory exact preflight. The bundle will be
regenerated with the actual environment and git state by the experiment
registry helper if the family is explicitly launched.
