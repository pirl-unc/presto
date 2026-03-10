# Data, Scripts & MHC Representation Audit: TODO

Consolidated findings from the MHC data quality audit, script hygiene review, train_iedb.py analysis, and MHC representation research. Items are organized by category, then by priority within each category.

---

## Category 1: MHC Data Quality

### DQ1: Fix dog B2M sequence (duplicated residue)
**File:** `data/b2m_sequences.csv`, entry `dog` / `E2RN10`
**Issue:** CSV has 126 aa; UniProt has 125. Extra `E` at ~position 60.
**Fix:** Re-fetch from UniProt E2RN10, replace the row.
**Priority:** Trivial.

### DQ2: Filter null alleles from training
**File:** `data/mhc_index.py` (build), `data/mhc_index.csv` (output)
**Issue:** ~1,567 N-suffix alleles encode frameshifted proteins that never reach the cell surface. Training binding predictions against these teaches noise.
**Fix:** During `build_mhc_index()`, detect N-suffix alleles and add `is_null=True` column. During allele resolution for training, exclude null alleles by default. Keep in index for completeness.
**Priority:** High.

### DQ3: Filter questionable expression alleles
**File:** Same as DQ2.
**Issue:** ~420 Q-suffix alleles may not produce functional cell-surface MHC.
**Fix:** Add `is_questionable=True` flag, same approach as DQ2.
**Priority:** Medium.

### DQ4: Flag H2-K*q as partial
**File:** `data/ipd_mhc/mouse_uniprot_overlay.csv`
**Issue:** 328 aa vs 362-369 for other Class I entries. Missing alpha-1 groove residues (not just signal peptide). UniProt P14428 is incomplete.
**Fix:** Add `is_partial=True` flag. Exclude from binding training. Check IMGT for alternate source.
**Priority:** Medium.

### DQ5: Strip trailing X from IMGT sequences
**File:** `data/mhc_index.py` (build)
**Issue:** 2,638 sequences end with `X` (IMGT stop codon convention).
**Fix:** `seq = seq.rstrip("X")` during index build.
**Priority:** Trivial.

### DQ6: Map remaining IPD species prefix codes
**File:** `data/mhc_index.py`
**Issue:** 138 IPD-MHC entries have empty species because their prefix codes (Stbr-, Tutr-, Zica-, etc.) aren't mapped.
**Fix:** Add mappings for the unmapped IPD prefix codes. These are standardized.
**Priority:** Low.

### DQ7: Add is_partial flag for groove-only fragments
**File:** `data/mhc_index.py` (build)
**Issue:** ~1,983 entries from IPD-MHC are groove-only fragments (70-99 aa). Not wrong, but heterogeneous.
**Fix:** Add `is_partial=True` for sequences significantly shorter than expected full-length for their gene family. Document in index metadata.
**Priority:** Low.

### DQ8: Parse signal peptides and augment MHC index
**File:** `data/mhc_index.py`
**Issue:** Signal peptide presence is inconsistent across sequences. Some are full precursors, others are mature forms. The model should learn to ignore signal peptides.

**Fix — index augmentation:**
Add three columns to `mhc_index.csv`: `sequence` (full as-is), `signal_peptide` (SP portion if known), `mature_protein` (mature portion if known). Only fill SP/mature when confidently determined.

**SP identification approach:** Per-gene reference alignment. For each gene family (HLA-A, HLA-B, H2-K, etc.), take one reference allele with known SP cleavage site (from UniProt annotation). For all other alleles of that gene, align first ~40 residues and cut at the homologous position. MHC alleles within a gene family share >80% identity, so the cleavage site is at a fixed position.

Known cleavage positions (mature protein start):
- HLA-A/B/C: residue ~25 (after `...AALA↓GSHS...` or similar)
- HLA-DRA: residue ~26
- HLA-DRB1: residue ~30
- H2-K/D/L: residue ~22-24

**Fix — training-time augmentation:**
When a sample's MHC has known `mature_protein`, randomly choose between `sequence` (full with SP) and `mature_protein` (SP stripped) during training. This teaches the model to be invariant to SP presence. Probability split: 50/50, or biased toward mature (70/30) since inference inputs will usually be mature.

**Priority:** Medium.

---

## Category 2: B2M Handling

### B2M1: Log human B2M fallback frequency
**Files:** `data/loaders.py:1716-1725`, `scripts/train_iedb.py:484`
**Issue:** When species is unknown, human B2M is silently substituted. This is probably rare but should be measured.
**Fix:** Add `logger.warning()` when the fallback triggers. Count occurrences during data loading and report in training logs.
**Priority:** Low.

### B2M2: Fix inconsistent fallback in _generate_mhc_only_samples
**File:** `scripts/train_iedb.py:2347`
**Issue:** This code path does NOT fall back to HUMAN_B2M when species is unknown, unlike every other path. Can produce `None` for `mhc_b` in Class I training samples.
**Fix:** Use the same fallback pattern as `_default_class_i_beta2m()`.
**Priority:** Medium.

---

## Category 3: Script Hygiene

### SH1: Rename train_iedb.py to train_unified.py
**Files:** `scripts/train_iedb.py`, `cli/train.py`, `scripts/probe_training.py`, `scripts/train_modal.py`, `tests/test_train_iedb.py`
**Issue:** File is named `train_iedb` but the CLI exposes it as `presto train unified`. It loads from IEDB, VDJdb, 10x, CEDAR, and other sources. The name is misleading.
**Fix:** Rename file. Update all imports and references. Rename test file.
**Priority:** Medium.

### SH2: Fix wrong repo URLs
**Files:** `scripts/train_modal.py:29`, `scripts/sanity_check_modal.py:70`
**Issue:** Both have `https://github.com/escalante-bio/presto.git` but repo is at `pirl-unc/presto`.
**Fix:** Change to `https://github.com/pirl-unc/presto.git`.
**Priority:** Trivial.

### SH3: Archive benchmark_binding_latents.py
**File:** `scripts/benchmark_binding_latents.py`
**Issue:** 490-line one-off research script. No tests, no CLI integration, no cross-imports.
**Fix:** Delete or move to `scripts/archive/`. It's in git history.
**Priority:** Low.

### SH4: Factor train_unified.py into modules
**File:** `scripts/train_iedb.py` (5,240 lines, 198.5 KB)
**Issue:** Not dead code — all conditional paths are reachable. But it does too many things. 70 functions across 5,240 lines.

**Proposed factoring:**

| Lines | Proposed module | Purpose |
|-------|----------------|---------|
| 352-760 | `training/probes.py` | Probe affinity/motif tracking |
| 760-1349 | `training/diagnostics.py` | Latent space analysis, info flow |
| 1490-2150 | `data/synthetic_negatives.py` | Synthetic negative augmentation |
| 2150-2523 | `data/unified_loader.py` | Multi-source record loading |
| 2523-2926 | `data/mhc_audit.py` | MHC resolution auditing |
| 3189-3672 | `data/record_parser.py` | Merged TSV parsing |

Reduces train_unified.py from ~5,240 to ~2,330 lines. Each extracted module is independently testable.
**Priority:** Medium-low (doesn't affect correctness; do incrementally).

### SH5: kd_log10 vs kd_nM — add clarifying comment
**Issue:** Both are computed and stored in metrics. They ARE interconvertible but serve different purposes.
**Fix:** Add a one-line comment where both are computed. No code change.
**Priority:** Trivial.

---

## Category 4: MHC Representation Improvements

These are architectural enhancements to help the model learn better MHC representations without MSA inputs. Ordered by effort-to-benefit ratio.

### MR1: Domain-aware positional encoding (optional auxiliary task)
**File:** `models/presto.py` (encoder section)
**Issue:** Current MHC positional encoding is plain sequential (`nn.Embedding(400, d_model)`). Position 1 and position 182 have no encoded structural relationship, even though both are strand A of their respective groove domains.

**Fix:** Add a per-residue `domain_id` prediction as an **optional auxiliary task**, not a hard-coded lookup. The model learns to predict which structural domain each MHC residue belongs to. When ground truth is available (known gene family with IMGT G-DOMAIN boundaries), supervise the prediction. When the input is an arbitrary unknown sequence, the model still predicts domain labels — the auxiliary head acts as a structural prior without requiring the information at inference time.

Domain categories (5 classes):
- GROOVE_FLOOR (beta sheets facing peptide)
- GROOVE_HELIX (alpha helix forming groove wall)
- IG_DOMAIN (Ig-like fold, not peptide-contacting)
- SIGNAL_PEPTIDE (if present)
- OTHER (transmembrane, cytoplasmic)

Implementation:
```python
self.mhc_domain_head = nn.Linear(d_model, 5)  # per-residue domain classification
# Loss: CE(domain_pred, domain_label) when labels available, masked otherwise
```

The predicted domain logits can optionally be fed back as a soft embedding added to MHC token states (like a learned positional prior), but this is a secondary enhancement.

IMGT G-DOMAIN boundaries for reference alleles (used to generate training labels):
```
MHC-I alpha (mature): floor1=1-49, helix1=50-90, floor2=91-139, helix2=140-182, Ig=183+
MHC-II alpha: floor=1-49, helix=50-90, Ig=91+
MHC-II beta: floor=1-49, helix=50-90, Ig=91+
```

**Key design choice:** This works for arbitrary sequences because it's a prediction, not a hard-coded lookup. The model learns the structural pattern from supervised examples and generalizes.

**Question:** Is IMGT G-DOMAIN boundary data actually downloadable in machine-readable form? If not, the boundaries above are well-established enough to hard-code as reference labels for each gene family (HLA-A, HLA-B, HLA-C, HLA-DRB1, etc.) and derive labels by aligning other alleles to the reference.

**Effort:** ~40 lines for the head + label generation script.
**Priority:** Medium.

### MR2: Pseudosequence extraction as a learned sub-task
**File:** `models/presto.py` (new auxiliary head)
**Issue:** The ~34 groove contact residues (the NetMHCpan pseudosequence) are the most information-dense positions for binding prediction. Currently the model has no explicit signal about which positions matter most.

**Fix:** Instead of hard-coding contact positions as a fixed attention mask (which breaks for arbitrary unknown sequences), make pseudosequence extraction a **learned sub-task with optional supervision**:

```python
# Per-residue "contact importance" prediction
self.mhc_contact_head = nn.Linear(d_model, 1)  # sigmoid → per-residue contact probability

# When ground truth available (known allele with NetMHCpan pseudosequence positions):
#   loss = BCE(contact_pred, contact_label)  where contact_label is 1 at ~34 known positions
# When unknown:
#   no supervision, but the learned contact probabilities still modulate attention
```

The predicted contact probabilities can be used as soft attention weights in the binding cross-attention:
```python
# In binding latent cross-attention, modulate MHC key values:
contact_weights = sigmoid(self.mhc_contact_head(h_mhc))  # (B, L_mhc, 1)
h_mhc_weighted = h_mhc * (1.0 + contact_weights)  # soft boost at predicted contact sites
```

This way: (a) for known alleles with pseudosequence labels, the model learns which positions matter; (b) for novel sequences, it generalizes from the learned pattern. The contact head is an attention prior, not a hard gate.

**Source for training labels:** NetMHCpan pseudosequence position files (publicly available, ~34 positions in mature protein coordinates). MHCflurry uses 37 positions. These are per-gene-family, not per-allele — any allele of HLA-A uses the same contact positions.

**Effort:** ~50 lines + a small data file.
**Evidence:** "Do Pseudosequences Matter?" (bioRxiv 2025) found curated pseudosequences remain the most efficient MHC representation.
**Priority:** Medium-high.

### MR3: Class-conditional groove bias + MHC-II groove unification
**File:** `models/presto.py` (groove_bias_a/groove_bias_b, lines 469-474)
**Issue:** The current groove bias has fixed cutoffs (alpha at ~210, beta at ~120). But the groove architecture is fundamentally different between classes:
- **MHC-I:** Groove is entirely on the alpha chain (alpha-1 + alpha-2, ~182 residues). B2M contributes nothing to peptide binding — it's a structural scaffold.
- **MHC-II:** Groove is a heterodimer: alpha-1 (~90 residues) + beta-1 (~90 residues) form the groove together. Both chains are essential.

The current single-decay bias doesn't capture this.

**Fix — groove bias:** Compute class-specific groove biases and blend with `class_probs`:
```python
# Class I: alpha groove = positions 1-182, beta (B2M) = no groove role
# Class II: alpha groove = positions 1-90 only, beta groove = positions 1-90
groove_bias_a = class_probs[:,:1] * bias_a_classI + class_probs[:,1:2] * bias_a_classII
groove_bias_b = class_probs[:,:1] * bias_b_classI + class_probs[:,1:2] * bias_b_classII
```

**Longer-term consideration — MHC-II single-chain input:**
For MHC-II, the groove is formed by alpha-1 + beta-1 together. It may be beneficial to present these as a single concatenated "groove sequence" with a (G4S)x5 linker, rather than as two separate segments. This mirrors experimental single-chain MHC-II (scMHC-II) constructs, which are standard in structural biology. For MHC-I, the input would be the alpha chain only (B2M dropped or kept as a separate non-groove segment).

This is a bigger change (affects tokenization, segment structure, collation) so it should be evaluated separately. The class-conditional groove bias is the quick win; the single-chain representation is a future experiment.

**Note on (G4S)x5 linker:** This is biologically realistic — scMHC-II constructs routinely use flexible glycine-serine linkers of this length. The linker replaces the transmembrane anchor and lets the two groove domains fold into their native contact geometry. The model should learn to treat linker tokens as non-informative (similar to padding).

**Effort:** ~25 lines for the groove bias fix. Single-chain input is a separate larger task.
**Priority:** Medium (groove bias fix). Low (single-chain experiment).

### MR4: ESM-2 embedding initialization — DEFERRED
**File:** `models/presto.py` (aa_embedding)
**Issue:** Design doc specifies ESM-2 initialization but it's not implemented.

**Assessment:** ESM-2's per-token (non-contextualized) amino acid embeddings are just a 20x320 matrix of amino acid property vectors. The marginal benefit over random init + training is unclear for a model that trains on millions of examples. The contextualized embeddings from ESM-2 would be more useful, but those require running an ESM model (even a small 8M one), adding a runtime dependency.

**Decision:** Defer. If the model struggles with amino acid property encoding after the other MR fixes, revisit. Not worth the dependency for a marginal per-token init improvement.

**Priority:** Low (deferred).

---

## Implementation Order

### Phase A — Trivial fixes (do immediately):
- DQ1: Fix dog B2M
- DQ5: Strip trailing X
- SH2: Fix repo URLs
- SH5: Add kd comment

### Phase B — Small fixes:
- DQ2 + DQ3: Null/Q allele flags
- B2M2: Fix inconsistent fallback path
- SH1: Rename train_iedb -> train_unified
- SH3: Archive benchmark script
- DQ4: Flag H2-K*q as partial
- B2M1: Add fallback logging

### Phase C — Medium effort:
- DQ8: Signal peptide parsing + training augmentation
- MR3: Class-conditional groove bias
- MR1: Domain-aware positional encoding (auxiliary task)
- MR2: Pseudosequence extraction (learned sub-task)

### Phase D — Larger effort:
- SH4: Factor train_unified.py into modules
- MR3 (extended): MHC-II single-chain input experiment
- DQ6: Map remaining IPD species prefixes
- DQ7: Add is_partial flag for fragments

---

## References

- NetMHCpan-4.1: Reynisson et al., NAR 2020. Pseudosequence positions.
- NetMHCpan-4.2: Frontiers in Immunology 2025. Transfer learning + structural features.
- "Do Pseudosequences Matter?": bioRxiv 2025. Pseudosequences outperform ESM embeddings.
- MHCflurry 2.0: O'Donnell et al., Cell Systems 2020. 37-position pseudosequence.
- BigMHC: Albert et al., PLOS Comp Bio 2023. 30-position information-content pseudosequence.
- ESM-MHC: ACM ICBBT 2024. ESM-2 features for MHC prediction.
- ESMCBA: arXiv 2507.13077, 2025. Domain-adapted ESM for pMHC.
- RPEMHC: Bioinformatics 2024. Residue-residue pair encoding.
- TransPHLA: Nature Machine Intelligence 2022. Full-sequence transformer.
- IMGT G-DOMAIN: Lefranc et al., 2005. MHC groove domain numbering.
