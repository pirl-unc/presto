# Reviews and external sources now coexist without changing targets

Completed locally on October 23, 2026 by Codex / GPT-6. Hitlist source commit
`0219827e08d2d8a400c1091ad770e87d62562b7a` (clean, unreleased 2.0.5); Presto registration
`7d4c7a5fb8d665cf74cf654069f44506468f4b5a` with the exact dirty-state receipt in `reproduce/launch.json`.
[Hitlist issue771](https://github.com/pirl-unc/hitlist/issues/771) documents the
schema collision. The fix is pushed to [draft767](https://github.com/pirl-unc/hitlist/pull/767).
Review, final hosted CI, merge and release are separate gates.

The schema-9 generation preserves the newer nine multimer adjudications alongside
all existing MAPP and MS-calibration evidence. The independent verifier checked
**183 files, all 126 predecessor files with distinct inodes, 56 appended-source
files, and 1,853,840 rows across 32 canonical query relations**. Those rows have
different grains; they are not that many independent assays. All four recipe
contents and partitions are unchanged. Their source provenance correctly points
to the new combined package.

## Data and curation contract

The input is the original source-reviewed schema-8 generation at
`/private/tmp/hitlist-v2-adjudication-204-reviewed` (manifest `8f35dc5529b376f1b2e3b25f91dd2f2dc57d0a53c4d8a76c094294193708c9b6`). The nine annotations
remain source-linked interpretations; they do not change an assay result or admit
previously excluded multimer examples. Its 578,529 T-cell assay IDs and 55,844
training examples remain unchanged by byte conservation.

Both external adapters use exactly the originals in
`/private/tmp/hitlist-v2-calibration-203-canonical` (manifest `bce500f34f111aaf3b41573e96fa630252204a3403d5a2e93e9c50caa97b8552`). All 33 PXD009092
source/reference files and all 23 PXD004886 source/calibration files are identical
to that generation. MAPP preserves captured endogenous products, nonbinding
fragments, source confidence/missingness, reference alternatives and procedural
controls. It is not an MHC ligand or free-concentration assay. Raw S4 preserves
positive-response selection and unresolved nominal/reporter claims; it does not
provide calibrated transfer to MAPP or intracellular concentrations. The existing
source and recipe policies are untouched; no new censoring, filtering, relabeling
or normalization policy was introduced.

| Preserved recipe | Existing admitted scope | Changed target / split files |
|---|---|---:|
| MAPP minimum three peptides/parent | 544 peptides, 65 parents, 130 condition sets | 0 |
| MAPP minimum two peptides/parent | 616 peptides, 101 parents, 202 condition sets | 0 |
| MAPP complete runs | 369 peptides, 71 parents, 3,696 source run quantities | 0 |
| Raw-S4 nominal response | 1,974 positive measurements, 224 summaries, 22 peptides | 0 |

## Tested conditions and independent checks

- Reference/predecessor fixtures, both historical schema-8 families, and the new
  schema-9 successor; unknown/ambiguous envelopes fail explicitly. New readers
  open the actual historical adjudication and external packages without changing
  them or advertising absent evidence.
- Actual new package `/private/tmp/hitlist-v2-combined-205-canonical`, manifest `42317a5efb7d642be5b8265131e6a6f7963e469adda946b780dac4d6aebb05b0`:
  85 materialized evidence relations; the multimer review source and both appended
  sources coexist in its canonical registry. Every existing baseline audit matches.
- Independent standard-library hashing and direct Parquet reading, with **no
  Hitlist imports**, check the inventory, source equivalence, exact nine review
  rows, query hashes and all four recipe outputs. Each target, contributor,
  exclusion, parent and split payload is byte-identical to the last canonical
  output. MAPP payloads also match the older standalone outputs.

The first independent check stopped because the initial config selected old
standalone MAPP exports, which predate `coverage.json`. This was a baseline
format error. `reproduce/verification-amendment.md` records the correction to the
latest canonical exports plus the additional original-payload check. Initial
config, verifier and failed log are retained. No source producer, target or model
was changed or rerun to obtain a pass.

## Runtime, software and reproduction

Local macOS 26.5 arm64, Python 3.12.6, PyArrow 26.0.0; CPU only, no Modal/GPU.
No environment installs/changes. Thread overrides and exact checkout/path hashes
are frozen. Run `bash reproduce/launch.sh` from a fresh destination as configured
in `reproduce/launch.json`. Both current and original verifier scripts are preserved
under `reproduce/source/`, as are the ten relevant Hitlist modules and full spec.

| Stage | Seconds | Peak process RSS |
|---|---:|---:|
| Source build and internal verification | 72.22 | included below |
| Open/catalog/baseline audit | 24.24 | included below |
| Stream all source/review query rows | 9.07 | included below |
| Four canonical materializations and verification | 9.74 | included below |
| Read both historical schema-8 families | 32.72 | included below |
| Complete producer | 148.01 | 3.28 GiB |
| Independent verifier | 12.83 | 0.64 GiB |

Normal local `./test.sh` passes **3,317 tests / 10 warnings** in 321.51 seconds
(325.53 seconds wrapper; 0.88 GiB peak process RSS). The unchanged memory guard
admitted one worker at 4.50 GiB free. The first sandbox attempt could not read
memory capacity and was retried outside the sandbox without bypassing the guard.
Seventy focused checks, format, lint and strict docs pass. These are local tests;
indexed-corpus and multi-Python hosted CI are reported separately.

## Artifacts and decision

`results/` contains the complete package manifest/catalog/audit, all query digests,
recipe coverage, historical-reader results, timings and independent verification.
`results/recipe-outputs.tar.gz` contains all four newly generated portable exports;
large original package bytes remain at the explicitly pinned paths. `launch_logs/`
retains the initial failure, successful amended verification and final normal-test
output. No validation/test prediction dump is produced: **no model is trained**,
no pretraining/synthetic data/loss terms or assay-to-output mapping change, and no
new predictive split exists in this conservation task. IC50 and binding-threshold
metrics do not apply. Existing memberships were checked, not re-evaluated.

Preferred integration: schema9 with the reviewed predecessor. Biological model
quality is unchanged: prior MAPP fits still fail uniform, and concentration,
precursor and observation-response limits remain. Next: shared admission of the
recovered cellular controls under Hitlist759, followed by a prospective local
signal test and exact-versus-sampled fit. The availability goal stays active.
