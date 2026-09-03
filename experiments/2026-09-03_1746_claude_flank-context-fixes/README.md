# Flank context wired end to end, on corrected mapping data

**Experiment id:** `2026-09-03_1746_claude_flank-context-fixes`
**Run dates:** 2026-09-03
**Agent/model:** Claude / Opus 5
**Git state:** `6f32d46126a73665e8927d85fc7a7983b2f05c09`, branch `claude/flank-context-and-data-fixes`, dirty: yes
**Status:** running

## Question and decision

Two questions, one paired design:

1. **Does the legacy-vs-masked conclusion survive on corrected code?** The
   2026-09-02 family answered "use `mask_unresolved`, defer candidate
   marginalization" on a codebase carrying the defects listed below.
2. **What happens once a protein terminus actually reaches the model?** It
   never had. `flank_n_is_terminus` was declared on three record dataclasses,
   moved to the device by `PrestoBatch.to`, and branched on by
   `Presto._pad_for_side` -- but no `PrestoSample(...)` construction in
   `data/loaders.py` passed it, so every row arrived `False` and the `X`
   boundary pad was unreachable. Measured on this corpus, **3.2% of binding
   rows carry an N-terminus and 2.9% a C-terminus**; all of them previously
   padded with `?` ("unknown") instead of `X` ("nothing is here"). On the C
   side that is the sharper error: a peptide ending at its protein's own
   C-terminus required no proteasomal cut, and the excision head was scoring a
   cleavage that never had to happen from a token asserting ignorance.

**Decision: pending run completion.**

## What changed since 2026-09-02_1720

Corrected in `claude/flank-context-and-data-fixes`:

- Terminus flags wired through binding, processing and elution records.
  `bulk_ms` carries no flanks, so its `False` default was already right.
- Flank context has one representation per layer: `""` in the frame, `None`
  on the sample, a pad token in the tensor. `?` no longer appears in any
  sequence string, so `is_encodable_sequence` rejects it plainly instead of
  `flank_context` special-casing one exact marker past the validator.
- The candidate collapse could let a left-join miss outrank the real mapping
  (empty identifiers sort first) and still stamp the row `single`/resolved.
- `_canonical_mask` read a float-encoded flag as False, disabling the
  canonical tie-break and masking the bug above.
- `_flank_coverage` gated on `flank_context_resolved`, so it reported
  resolution, not coverage. See the table below for why that mattered.
- Three drifted copies of "which categories are resolved" collapsed into one
  table, with the partition asserted.
- The three-way split reused the two-way helper's `seed + 53` peptide order
  and filled *test* from the front where two-way fills *validation* from the
  front, so at a given seed every held-out test peptide had already been used
  for model selection. Separate stream now.

**Confounding, stated explicitly.** Within this experiment, legacy vs masked
remains a clean paired comparison: both arms get the same terminus wiring, the
same corrected mapping data, and the same split. Comparing these numbers
against the 2026-09-02 table is *not* clean -- the terminus wiring and the
split seed both moved. Mapping categories are byte-identical between the two
families seed for seed, so the data-correctness fixes are defensive on this
corpus rather than corpus-changing; the differences that remain are the
terminus wiring and the new test split.

## The coverage stat now separates the arms

The previous family printed `flank coverage [binding]: 60.0%` for **both**
arms, because the stat gated on resolution -- which masking does not change.
Split into two stats, the arms are distinguishable:

| Arm | coverage [binding] | resolution [binding] | coverage [elution] | resolution [elution] |
|---|---|---|---|---|
| `legacy_global_canonical` | 100.0% | 60.0% | 100.0% | 86.8--87.3% |
| `mask_unresolved` | 60.0% | 60.0% | 86.8--87.3% | 86.8--87.3% |

Legacy keeps every flank while only 60% of binding junctions are genuinely
resolved. Masking clears exactly that 40% gap, so its coverage collapses onto
its resolution. That 60.0% is precisely the number the old single stat
reported for both arms.

So the masked arm trains **40% of binding rows and ~13% of elution rows** on an
explicitly unknown junction rather than an arbitrarily chosen one.

## Dataset and curation contract

- Source: frozen hitlist 1.55.8 curated indexes on the Modal `presto-data`
  volume, hash-verified before the GPU is touched. Note the local
  `~/code/hitlist` checkout has since moved to 1.56.0; the remote contract is
  unaffected and remains 1.55.8.
- Allele `HLA-A*02:01`; 16,721 binding, 20,000 elution, 1,000 stability and
  2 kinetics records per seed (37,723 total). Binding uncapped; capped
  modalities use reservoir sampling with sampling seed `training_seed + 17`.
- Split: peptide-disjoint 80/10/10, roughly 30,178 / 3,772 / 3,772 rows and
  ~23,200 / ~2,870 / ~2,900 peptides.
- Mapping strata: `single`, `flanks_agree`, `within_gene_canonical`,
  `cross_gene_unresolved`, `within_gene_unresolved`, `unmapped`.
- Supervision parity between the two policies verified for all four evidence
  families in `results/data_audit.json`.

Frozen artifact hashes are recorded in `reproduce/` and re-verified per run in
`results/runs/*/data_contract.json`.

## Training contract

Presto expanded topology, d_model 128 / 2 layers / 4 heads, AdamW
lr 2.8e-4 weight_decay 0.01, batch 256, 10 epochs, bf16 autocast.
Synthetic negatives, MHC augmentation and probe tracking all disabled.
Requested GPU `H100!`; observed hardware recorded per run in
`results/runs/*/hardware.json`.

## Conditions

| Condition | Policy | Seeds |
|---|---|---|
| legacy | `legacy_global_canonical` | 42, 43, 44 |
| masked | `mask_unresolved` | 42, 43, 44 |

## Results

Pending run completion. Held-out validation and test metrics, per-example
prediction dumps, paired per-seed differences and the mapping-stratified
breakdown will be written to `results/` by `code/aggregate.py`.

## Reproduce

```bash
modal run <exp>/code/launch.py --preflight-only      # CPU: hashes + pipeline
modal run --detach <exp>/code/launch.py              # 6 x H100!
python scripts/fetch_experiment_modal_runs.py --experiment-dir <exp> --wait
python <exp>/code/aggregate.py
```

Bundle: [`reproduce/`](./reproduce/)
