# PRIME 2.1 through mhctools — local software smoke test

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: source frozen; reproducible smoke execution pending

## Objective

Verify that the already licensed local PRIME 2.1 distribution can be invoked
through `mhctools.PRIME` with the existing isolated MixMHCpred 3.0 runtime. This
is a two-peptide software/integration smoke test, not an immunogenicity benchmark
and not evidence about model accuracy.

## Contract

- peptides: `NLVPMVATV`, `GILGFVFTL`;
- allele: `HLA-A*02:01`;
- PRIME: `/Users/iskander/code/PRIME/PRIME`, academic/non-commercial license;
- MixMHCpred: `/Users/iskander/code/weirdo/tmp/MixMHCpred-3.0/MixMHCpred`;
- isolated backend: `/Users/iskander/code/weirdo/tmp/prime-env/bin/python`;
- interface: `mhctools.PRIME` from the editable shared OpenVax environment;
- local CPU only; requested/observed GPU: none;
- no model training, loss, validation, TEST, locked cohort, or outcome labels.

The shared environment's installed distribution metadata currently says
`mhctools 4.5.1`, while its editable source checkout describes itself as
`v4.9.4`; the receipt records both rather than silently conflating them.

## Result

Pending the frozen rerun. The earlier exploratory call already resolved
MixMHCpred 3.0 and returned predictions, but only the frozen execution below is
the canonical result.

## Reproduce

Run `reproduce/launch.sh`. PRIME writes transient nested MixMHCpred files inside
its licensed installation, so a sandboxed execution may require explicit write
approval for that existing directory. The runner deletes its ordinary mhctools
temporary directory and writes only the compact JSON result under `results/`.

There is deliberately no test prediction dump: this family contains no held-out
dataset and measures integration behavior only.
