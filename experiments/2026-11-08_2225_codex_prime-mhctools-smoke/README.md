# PRIME 2.1 through mhctools — local software smoke test

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: complete

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

The frozen run passed in 1.19 seconds on local CPU. `mhctools.PRIME` resolved
MixMHCpred 3.0 through the isolated Python 3.12.6 backend and returned one
HLA-A*02:01 immunogenicity prediction per peptide, preserving input order:

| Peptide | PRIME score | PRIME percentile rank |
|---|---:|---:|
| NLVPMVATV | 0.190731 | 0.027 |
| GILGFVFTL | 0.277195 | 0.004 |

The PRIME wrapper SHA-256 is
`4db2323413bdd0f0cffffd9321b9439cd7c4569716aaa342a49a0b9b18a9a598`;
the MixMHCpred wrapper SHA-256 is
`f2db4120eddf4f17daec1183c9a466fa105ed97915c8fb724bc9c6e06405498b`.
The complete result is `results/smoke.json` (3,785 bytes, SHA-256
`f3e9411b81a7c55b3bf5b99395957d18da5b3820b5a8bb82ded3ec4ef65aad59`).

This closes the software-integration question: PRIME can be run through
mhctools here. It does not authorize use outside the recorded license scope and
does not replace the still-needed newly disjoint class-I T-cell cohort.

## Reproduce

Run `reproduce/launch.sh`. PRIME writes transient nested MixMHCpred files inside
its licensed installation, so a sandboxed execution may require explicit write
approval for that existing directory. The runner deletes its ordinary mhctools
temporary directory and writes only the compact JSON result under `results/`.

There is deliberately no test prediction dump: this family contains no held-out
dataset and measures integration behavior only.
