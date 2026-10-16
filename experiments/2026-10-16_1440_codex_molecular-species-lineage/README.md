# Molecular MHC species and native role audit

Planned data-only audit by Codex / GPT-6 of the unchanged 9,053 selected observations.
The executed recipe will be frozen before use. Inputs are hash-pinned native training
parquet, frozen membership and published IC50/species inspection tables from the
0.1.16 fit. No model, MHC sequence reconstruction, training, new predictions or new
full-input qualification is included. Historical results remain unchanged.

Verify independent native source roles, canonical molecular auxiliary labels and
all 2,614 raw/effective IC50 targets, operators and partitions. Publish all selected
source/context rows, changed labels, citations, policy, counts, code and hashes
through datacache. Processing conditioning, antigen attribution and failed model
quality remain explicit follow-ups. See tasks/molecular-species-lineage-spec.md.

Run reproduce/launch.sh from an isolated environment with the pinned checkout
installed. CPU only; no cloud or GPU. The runtime receipt will record actual
versions, code/data hashes and resource usage.
