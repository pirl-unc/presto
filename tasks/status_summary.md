# Presto Status Summary — 2026-03-06

## Test Suite: 701 passed, 29 failed

---

## Active Bugs

### BUG-1: Groove extraction crashes on test fixture sequences (25 of 29 failures)

Most test failures share one root cause: `prepare_mhc_input()` in `data/groove.py:737` raises `ValueError` when it cannot extract groove halves from short/synthetic MHC sequences used in test fixtures. Before the groove-native path landed, loaders and predictor accepted raw amino acid strings directly. Now they route through groove extraction, which fails on:
- **`no_cys_pairs`**: synthetic/stub MHC sequences in test fixtures lack the conserved Cys disulfide pairs needed by the Cys-anchor heuristic.
- **`too_short`**: short test sequences (e.g. `"ACDEFGHIKLMNPQRSTVWY"`) are below the groove extraction minimum.

**Affected tests**: 11 in `test_loaders.py`, 13 in `test_predictor.py`, 1 in `test_train_iedb.py`.

**Fix**: Either (a) update test fixtures to use real groove-bearing sequences or sequences that pass extraction, or (b) add a fallback in `prepare_mhc_input()` that returns raw truncated sequence when extraction fails on user-provided direct sequences (not allele names). Option (a) is cleaner.

### BUG-2: Two-field alias resolution returns wrong resolution level (1 failure)

`test_resolve_alleles_uses_aliases_from_index` expects `HLA-A*02:01:01` but gets `HLA-A*02:01`. The protein-resolution normalization contract now collapses to two-field names, but this test still expects three-field output.

**Fix**: Update the test expectation to match the current normalization contract, or if the intended behavior is three-field resolution, fix the alias lookup to return the full index key rather than the normalized alias.

### BUG-3: Synthetic binding negatives crash on groove extraction (1 failure)

`test_synthetic_binding_negatives_stay_in_parent_binding_task_group` — same root cause as BUG-1. Synthetic negative generation uses a stub MHC sequence that fails groove extraction.

### BUG-4: Predictor checkpoint loading (2 failures)

`test_from_checkpoint` and `test_from_checkpoint_uses_config` — likely downstream of the same groove extraction issue when the predictor tries to resolve test alleles against fixtures that lack real sequences.

---

## Dead Code (525,322 parameters, 2.4% of model)

These modules are defined in `__init__` but never called in `forward()`:

| Module | Params | Location |
|--------|--------|----------|
| `immunogenicity_cd8_mlp` | 262,656 | `models/presto.py:477-481` |
| `immunogenicity_cd4_mlp` | 262,656 | `models/presto.py:482-486` |
| `PresentationBottleneck` | 4 | `models/presto.py:521` |
| `w_presentation_class1_latent` | 1 | `models/presto.py:512` |
| `w_presentation_class2_latent` | 1 | `models/presto.py:513` |
| `w_class1_presentation_stability` | 1 | `models/presto.py:522` |
| `w_class1_presentation_class` | 1 | `models/presto.py:524` |
| `w_class2_presentation_stability` | 1 | `models/presto.py:523` |
| `w_class2_presentation_class` | 1 | `models/presto.py:525` |

These are vestiges from the pre-latent-DAG immunogenicity and presentation pathways. They waste GPU memory and get serialized into every checkpoint. Should be deleted (with a `_load_from_state_dict` shim for old checkpoint compatibility).

---

## Legacy TCR Infrastructure (still in codebase, marked non-canonical)

The TCR tower and matcher remain in the codebase behind `enable_tcr=False`:
- `TCREncoder` + `TCRpMHCMatcher` instantiated in `__init__` (`models/presto.py:536-537`)
- `encode_tcr()` method (`models/presto.py:2158`)
- `predict_chain_attributes()` method (`models/presto.py:2174`)
- `tcr_a_tok`/`tcr_b_tok` forward path gated by `enable_tcr` (`models/presto.py:1611-1735`)
- `models/tcr.py` (full TCR encoder + matcher module)

This is documented as "future/non-canonical" but still contributes significant parameter count when `enable_tcr=True`. Not urgent to remove, but should be cleaned up when receptor removal (Phase 3-5 of the receptor removal plan) lands.

---

## Incomplete Implementation Phases

### Groove-native runtime (Phase 3-6 of integrated plan)

The groove extraction algorithm and augmented index are built and tested. The runtime contract rewrite — making the actual training/inference path consume groove halves instead of full chains — is partially landed but causing the 25+ test failures above. Remaining work:

- [ ] **Phase 3**: Unified sample/batch contract rewrite — switch `PrestoSample`/`PrestoBatch` MHC inputs to groove halves, add `tcr_evidence` targets, remove receptor chain tensors.
- [ ] **Phase 4**: Model surgery — groove-half positional/segment embeddings, delete `groove_bias` path, TCR tower removal, add `tcr_evidence` head.
- [ ] **Phase 5**: Training/predictor/CLI cleanup for the new contract.
- [ ] **Phase 6**: Local sanity training + 1-epoch Modal diagnostic.

### Receptor removal (Phase 3-5 of receptor removal plan)

- [ ] Remove receptor fields from `PrestoSample`/`PrestoBatch`.
- [ ] Remove TCR tower, matcher, chain classification from model.
- [ ] Remove legacy TCR training tasks; replace with pMHC-only `tcr_evidence` outputs.
- [ ] Update docs/tests.

### Learning refactor end-to-end verification

- [ ] Full 1-epoch Modal training on the refactored stack.
- [ ] Analyze losses, probes, gradients, allele separation.
- [ ] Longer full training run if 1-epoch looks healthy.

### mhcgnomes failing cases audit

- [ ] Enumerate every allele-bearing source parseable by `mhcgnomes`.
- [ ] Run full parse audit, cluster failures by pattern.
- [ ] Distinguish true parser gaps from bad data tokens.

---

## Data Quality Items (from `tasks/data_and_scripts_audit.md`)

### Trivial (do immediately)
- **DQ1**: Fix dog B2M duplicated residue in `data/b2m_sequences.csv`.
- **DQ5**: Strip trailing `X` from IMGT sequences during index build.
- **SH2**: Fix wrong repo URLs in `scripts/train_modal.py` and `scripts/sanity_check_modal.py` (still point to `escalante-bio/presto` instead of `pirl-unc/presto`).
- **SH5**: Add clarifying comment for `kd_log10` vs `kd_nM`.

### Small fixes
- **DQ2+DQ3**: Add `is_null`/`is_questionable` flags for N-suffix (858) and Q-suffix (420) alleles; exclude from binding training by default.
- **B2M2**: Fix inconsistent fallback in `_generate_mhc_only_samples` (can produce `None` for `mhc_b` in Class I).
- **SH1**: Rename `train_iedb.py` to `train_unified.py` across all references.
- **DQ4**: Flag `H2-K*q` as partial (328 aa, missing alpha-1 groove residues).

### Medium effort
- **MR1**: Domain-aware positional encoding (auxiliary task, ~40 lines).
- **MR2**: Pseudosequence extraction as learned sub-task (~50 lines + data file).

---

## Probe Discrimination Status

After the learning refactor and training fixes, one-epoch Modal training shows:
- `SLLQHLIGL + HLA-A*02:01`: KD ~841 nM, binding ~0.34
- `SLLQHLIGL + HLA-A*24:02`: KD ~856 nM, binding ~0.34
- Delta: binding +0.005 (correct direction but weak)

The model learns monotone loss reduction and correct A*02:01 > A*24:02 ordering, but allele discrimination is still weak after one epoch. This is expected to improve with longer training and groove-native MHC representation.

---

## Recommended Priority Order

1. **Fix the 29 test failures** — most are BUG-1 (groove extraction on test fixtures). This is blocking CI/confidence in the codebase.
2. **Delete dead code** (525K params) — straightforward cleanup, reduces checkpoint bloat.
3. **Complete groove-native runtime** (Phases 3-6) — the highest-impact architectural change remaining.
4. **Receptor removal** — coordinate with groove runtime since both hit the same interfaces.
5. **Trivial data/script fixes** (DQ1, DQ5, SH2, SH5) — quick wins.
6. **Null/questionable allele filtering** (DQ2+DQ3) — improves training data quality.
7. **Full Modal training run** — only meaningful after contract stabilizes.
8. **mhcgnomes audit** — important for data coverage but not blocking architecture work.
