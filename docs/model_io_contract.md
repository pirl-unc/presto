# Presto Model I/O Contract

Every input the model accepts, every output it produces, which path each input is
allowed to reach, and what supervises what.

Companion to [`design.md`](design.md) (architecture),
[`assay_modeling_contract.md`](assay_modeling_contract.md) (the normative input rule)
and [`assay_learning_scheme.md`](assay_learning_scheme.md) (data sources).

Status: **target contract**. Sections marked *(implemented)* describe the code today;
sections marked *(planned)* are the agreed design not yet built. The gaps are listed
explicitly in §8 rather than glossed.

---

## 1. What the model is

One model over one question, asked at several depths:

> Given a peptide in its protein context, an MHC molecule, and the biological and
> experimental state of the sample it came from — was this peptide produced, does it
> bind, is it presented, is it seen by T cells, and would we have observed it?

Everything below exists to keep those sub-questions separable, so that a gain in one
cannot be quietly borrowed from another.

## 2. Inputs, in five tiers

The tiers are a permission system, not a taxonomy. Each tier states **where its values
are allowed to reach**, and those limits are the model's main defense against learning
shortcuts instead of biology.

### Tier 1 — Sequence *(implemented)*

The only content the trunk encoder ever sees.

| input | shape | notes |
|---|---|---|
| `peptide` | tokens | the analyte |
| `nflank`, `cflank` | tokens | source-protein context; the junction substrate |
| `mhc_a`, `mhc_b` | tokens | full chains, groove-extracted |

**Rule:** the trunk is a function of these and nothing else. Every other tier acts on
routing, output heads, or observation-model offsets.

### Tier 2 — Provenance selectors *(implemented)*

Protocol facts that decide **which sub-model applies**, never features fed to a
predictor that must generalize.

| input | domain | meaning |
|---|---|---|
| `peptide_source` | `{mhc, protein}` | what was captured: an MHC ligand, or a protein digested in vitro |
| `enzymatic_digest` | `{none, trypsin, chymotrypsin, lysc, gluc}` | post-capture step; non-`none` only when `peptide_source = protein` |

These are nested, not independent: the digest is a step *after* capture, and it exists
on only one arm of the fork.

```
cells → lyse
   ├─ peptide_source = mhc:      immunoprecipitate → acid-elute → (no digest)
   └─ peptide_source = protein:  whole lysate → denature → digest(enzyme)
```

**Rule:** a selector may gate which branch computes the termini. It may **not** be
concatenated into any predictor. `peptide_source` separates the two corpora perfectly,
so as a feature it is a free shortcut ("protein-sourced ⇒ never presented"); as a gate
it is just the truth about which process cut the peptide.

### Tier 3 — Cellular state *(implemented)*

Causal on **where the peptide was cut**. Meaningful only when `peptide_source = mhc`;
for a digested protein the donor cell's machinery is irrelevant, because the protein was
extracted intact and denatured before any enzyme touched it.

| input | domain | source |
|---|---|---|
| `host_species` | ~50 values | `host` (100% populated) |
| `mhc_species` | 21 values | `mhc_species` (100%) — differs from `host_species` for transfectants |
| `stimulus` | `{none, ifn_gamma, ifn_type1, tnf_alpha, tlr, cell_activation, cytokine_unspecified}` | `condition_category`; **default `none`** — read as "not known to be stimulated", which conflates an untreated sample with an unrecorded one. `cytokine_unspecified` is the separate case where a cytokine *was* applied but the deposit does not name it. |
| `apm_perturbation` | see below | `apm_*` / `condition_category` |

`apm_perturbation` is grouped by **mechanism**, not by gene. Per-gene flags exist
upstream but are too thin to learn from individually (ERAP1 25 samples, TAP1 16, B2M 12,
and ~12 each for TAP2/TAPBP/ERAP2/PDIA3/CALR/IRF2/GANAB/SPPL3), and a single boolean
would make biologically opposite interventions identical:

| group | genes | effect on the termini this model predicts |
|---|---|---|
| `none` | — | wild type |
| `peptide_supply` | TAP1/2, PSMB5/8/9/10 | changes which peptides reach the ER at all |
| `n_term_trimming` | ERAP1, ERAP2 | shifts the N-terminus specifically |
| `loading_complex` | TAPBP, CALR, CANX, PDIA3 | destabilizes loading; does not change cleavage |
| `mhc_null` | B2M | abolishes class I presentation outright |
| `class_ii_loading` | HLA-DM/DO, CD74, CIITA | class II editing and register |

**Rule:** may condition the processing/termini path only. Must not reach binding,
presentation or recognition.

### Tier 4 — Expression context *(planned)*

Causal on **whether the source protein was present**, not on where it was cut.

| input | resolution | coverage |
|---|---|---|
| `expression_context` | backoff `cell_line_name` → `cell_type` → `source_tissue` | 58% → 69% → 97% |
| `source_protein` | UniProt / Ensembl id | for the per-protein abundance offset |

Backoff rather than one field with a catch-all: the three are nested with inverted
coverage, so descending the chain reaches near-total coverage instead of dumping 30% of
rows into "other".

**Rule:** enters the **observation model as an offset** — `abundance(protein, sample)` —
and nowhere else. Fed to the processing latent it would let the model memorize
"B-cell sample ⇒ these peptides", which is identity, not biology.

### Tier 5 — Output-side descriptors *(implemented)*

Never touch the trunk; they parameterize readout heads. This is the sanctioned mechanism
in `assay_modeling_contract.md`.

| group | axes |
|---|---|
| assay descriptor | type · method · prep · geometry · readout |
| acquisition | instrument · fragmentation · acquisition mode · labeling · FDR · fractionation depth |
| T-cell context | method · readout · APC · culture · stimulation · peptide format *(legacy; more conditioned than the contract allows)* |

### Optional overrides *(implemented)*

`mhc_class`, `mhc_species`, `species_of_origin` — constrained priors on values otherwise
inferred from sequence, per the contract's override clause.

## 3. Latents *(implemented)*

Twelve, per `design.md` S7.1, with `--latent-topology expanded`. Ten are
cross-attention; immunogenicity is an MLP over its dependencies.

| level | latents | token access |
|---|---|---|
| 0 | `processing_class1`, `processing_class2` | nflank + peptide + cflank |
| 0 | `species_of_origin`, `ms_detectability` | peptide only |
| 1 | `binding_affinity`, `binding_stability` | peptide + mhc_a + mhc_b |
| 2 | `presentation_class1`, `presentation_class2` | **none** — pure bottleneck over upstream latents |
| 2.5 | `recognition_cd8`, `recognition_cd4` | peptide + foreignness |
| 3 | `immunogenicity_cd8`, `immunogenicity_cd4` | none — MLP over binding + recognition |

Verified behaviorally, not just declared: peptide-only latents are *exactly* invariant to
MHC and flank substitution; processing moves only via the sanctioned `context_vec`
channel, three orders of magnitude less than binding does.

## 4. Outputs

123 tensors. By family:

| family | key outputs |
|---|---|
| **processing / excision** | `processing_class{1,2}_logit`, `processing_logit`, `excision_logit`, `excision_s_n`, `excision_s_c`, `excision_s_internal`, `excision_s_len` |
| **binding** | `assays.{KD_nM, IC50_nM, EC50_nM, kon, koff, t_half, Tm}`, `binding_affinity_score`, `binding_affinity_probe_kd` |
| **core / PFR** | `core_start_logit`, `core_length`, `core_membership_prob`, `npfr_length`, `cpfr_length` |
| **presentation** | `presentation_class{1,2}_logit`, `presentation_logit` |
| **elution / MS** | `elution_logit`, `ms_logit` (same tensor), `ms_detectability_logit` |
| **recognition** | `recognition_cd{8,4}_logit` |
| **immunogenicity** | `immunogenicity_cd{8,4}_logit`, `immunogenicity_logit` |
| **T cell / TCR** | `tcell_logit`, `tcr_evidence_logit`, `tcr_evidence_method_logits` |
| **identity** | `mhc_class_logits`, `mhc_species_logits`, `mhc_{a,b}_type_logits`, `chain_compat_logit`, `species_of_origin_logits`, `foreignness_logit` |
| **vectors** | `pep_vec`, `pmhc_vec`, `mhc_{a,b}_vec`, `groove_vec`, `latent_vecs` |

Every `*_logit` has a paired `*_prob`.

### The one composed output

```
elution_logit = softplus(w_p)·presentation_logit
              + softplus(w_d)·ms_detectability_logit
              + bias
```

This is the only place two heads are summed into a third, and it is the crux of the
identification argument in §6.

## 5. Observation model

Both corpora factorize identically:

```
logit P(observed) = excision(peptide, protein | source, digest, conditions)
                  + [ MHC cascade ]          # peptide_source = mhc only
                  + detect(peptide | acquisition)
                  + abundance(protein | sample)
```

```
source=protein:  excision(digest) ────────────────────────────→ detect → observed
source=mhc:      excision(conditions)   → TAP → bind → present → detect → observed
                                                                   ↑
                                                    shared, and identifiable only
                                                    because the top branch has no
                                                    MHC term
```

## 6. Supervision map

| output | supervised by | loss | negatives |
|---|---|---|---|
| binding affinity | IEDB/CEDAR IC50/KD/EC50 | censored regression, 6 assay families | synthetic |
| stability, kinetics | half-life, Tm, on/off rate | censored regression | none |
| processing / excision | in-vitro digests *(today)*; APM conditions *(planned)* | BCE | rule-violating enzyme relabels **(real)** |
| elution / presentation | MHC ligand MS | BCE, Noisy-OR bags for multi-allele | synthetic |
| MS detectability | shotgun depth ladder | BCE, graded targets | in-silico digest *(planned)* |
| T cell, immunogenicity | IEDB/CEDAR T-cell | BCE + context CE | synthetic cascade |
| TCR evidence | VDJdb / 10x / McPAS | BCE, down-weighted 0.05 | none (positive-only) |
| identity auxiliaries | MHC sequence | CE, weight 0.1 | n/a |

The shotgun branch is the only source of **real** negatives anywhere in the model.

## 7. Invariants

Testable statements, not aspirations. Those marked ✓ have tests today.

1. ✓ The trunk is a function of Tier 1 only.
2. ✓ Peptide-only latents are exactly invariant to MHC and flank substitution.
3. ✓ Presentation has no token access and always has dependencies (else the empty-KV
   fallback would expose it to token 0).
4. ✓ Assay descriptors reach output heads only; `binding_context` is first read after
   the latent DAG.
5. *(planned)* `peptide_source` is unreachable from binding, presentation and recognition.
6. *(planned)* Tier 3 conditions reach the processing path only.
7. *(planned)* Tier 4 expression enters as an observation offset only.
8. ✓ A row missing a label contributes zero gradient to that task.

## 8. Known gaps

Recorded rather than glossed. Each is tracked in `tasks/todo.md` (repository, outside the docs tree).

**Closed**

1. ~~Validation is not peptide-disjoint.~~ `peptide_grouped_split_indices` groups by
   peptide alone; the trainer asserts disjointness after splitting. `--split-mode
   random_rows` reproduces the old behavior and says so loudly.

3. ~~Tiers 2–4 are one flat `machinery` axis.~~ Split into `peptide_source`,
   `enzymatic_digest`, `processing_stimulus` and `apm_perturbation`, with the source
   acting as a soft gate rather than a feature.
4. ~~`s_len` conflates two mechanisms.~~ Length and missed-cleavage terms are gated to
   the protein branch; the in-vivo branch contributes neither, so the protease is no
   longer credited for MHC groove selection.
6. ~~Detectability is validated out of domain.~~ `dual_corpus_transfer_set` builds the
   24,125-peptide in-domain evaluation set. Measuring it is an experiment, not code.

**Open**

2. **The in-vivo processing path still receives no gradient.** The factorization made
   the conditioning *expressible* and Tier 3 state now flows from hitlist (187
   `n_term_trimming` rows, 187 `peptide_supply`, 88 IFN-γ in a representative slice),
   but excision labels come only from `data/bulk_ms.py`, whose rows are all
   `peptide_source="protein"`. Every supervised row therefore has `is_protein == 1`,
   and `(1 - is_protein)` zeroes the in-vivo terms: ~482 parameters train to nothing.
   Verified through the real pipeline and pinned by
   `test_real_pipeline_still_starves_the_in_vivo_path`.

   Closing it needs excision labels on MHC rows. An observed ligand is evidence the
   in-vivo machinery produced it, so elution positives are natural excision positives —
   but the negatives are the same unsolved problem as detectability negatives. The
   alternative is routing Tier 3 conditions into the *processing latent* rather than
   only the excision readout, so ERAP-KO-vs-WT elution contrasts supervise them through
   the existing elution loss. That is a contract decision, not a patch.

5. **`processing` and `excision` remain parallel scores of one question.** Merging them
   changes the semantics of an existing supervised task, so it wants the Stage 4 arm-C
   result first rather than a unilateral edit.
7. **T-cell context conditioning is legacy** and more input-conditioned than the
   contract allows. Removing it materially changes T-cell predictions, so it needs its
   own before/after rather than being folded into an unrelated refactor.
