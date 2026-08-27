# The Assay Learning Scheme

Every source of supervision Presto learns from, what metadata distinguishes it, which
output it drives, and how it is scored.

This document is **descriptive** — it records what the code does and what the data
contains. The **normative** rule lives in
[`assay_modeling_contract.md`](assay_modeling_contract.md): assay identity may choose
supervision targets and may parameterize output heads, but must never enter the
predictive input path. Architecture is in [`design.md`](design.md); training mechanics
in [`training_spec.md`](training_spec.md).

Status markers: **[impl]** in the canonical path today, **[partial]** present but
incompletely wired, **[planned]** designed, not built.

---

## 1. Why a single scheme

Presto is one model over many measurement types that disagree with each other. An IC50
from a competitive radioactivity assay and a KD from direct fluorescence measure related
but non-identical quantities on different scales. A half-life from radioactive
dissociation and one from fluorescence polarization differ by a method offset. A
mass-spec non-detection can mean "not presented" or "not detectable". Pooling any of
these without modeling the difference silently trains the model to average across
incompatible scales.

The scheme has one shape everywhere:

```
shared latent  ──►  family-specific output head  ──►  loss on the rows that carry that label
     ▲                        ▲
sequence only        assay descriptor parameterizes
                     the head (never the input)
```

## 2. The generative factorization

Both MS corpora, and every binding measurement, decompose the same way:

```
logit P(observed) = excision(peptide, protein, machinery)   # was it produced?
                  + [ MHC cascade ]                            # MHC branch only
                  + detect(peptide, acquisition)               # would it fly?
                  + abundance(protein, sample)                 # was the source there?
```

`excision` and `detect` are both functions of peptide sequence, so an MHC-only corpus
cannot separate them. The non-MHC shotgun branch (§3.9) is what breaks the confound.
See `tasks/protease_detectability_spec.md` in the repository.

`excision` is scored by `ExcisionHead` (`models/heads.py`) as
`n_terminus_score + c_terminus_score + missed_cleavage_score + length_score_value` — the two termini, a missed-cleavage penalty applied
only to machinery with a hard rule, and a length term — where the machinery
**indexes the readout** and never conditions
the trunk. In-vitro protease profiles are pinned to their exact P1 rules by default, which makes
them constraints rather than findings — set `pin_profiles=False` for the genuine
known-answer calibration, where the rule has to survive training. The proteasome is a
learned convex mixture over those same profiles, matching its beta1/beta2/beta5
catalytic specificities.

## 3. Source inventory

Counts are as-loaded, not as-trained; caps and splits apply downstream. Merged-TSV
figures come from `data/merged_deduped_funnel.tsv`; hitlist figures from the built
indexes as of 2026-08-26.

### 3.1 Binding affinity **[impl]**

| | |
|---|---|
| Sources | IEDB, CEDAR (via merged TSV or `hitlist` binding index) |
| Rows | 251,418 merged / 29,470 unexploded for a single allele slice in hitlist |
| Record | `BindingRecord` (`data/loaders.py`) |
| Families | `IC50`, `EC50`, `KD`, `KD (~IC50)`, `KD (~EC50)` |
| Head | `assays.KD_nM` via `AssayHeads` (`models/heads.py`) |
| Losses | `binding`, `binding_kd`, `binding_ic50`, `binding_ec50` — all `censor`, weight 1.0 |
| Censoring | Full. `qualifier ∈ {-1, 0, 1}` → `censor_aware_loss` (`training/losses.py`) |
| Fan-out | `_categorize_binding_assay_type` (`data/collate.py`) emits six mutually exclusive masked tasks from one value |

This family is the template. Everything below is measured against how completely it
models its own heterogeneity.

### 3.2 Stability **[partial]**

| | |
|---|---|
| Rows | `t_half` 11,375 merged / 10,013 hitlist · `Tm` 1,250 merged / 178 hitlist |
| Record | `StabilityRecord` |
| Head | `assays.t_half`, `assays.Tm` |
| Losses | `t_half`, `tm` — `censor`, weight 1.0 |
| Censoring | Honored. `t_half_qual` / `tm_qual` reach the batch and both losses are censor-aware, so a ">2h" row is no longer trained toward exactly 2h |

`half life` is **not one assay**. Corpus-wide it spans six methods:

| method | rows |
|---|---:|
| purified MHC/direct/radioactivity | 6,629 |
| purified MHC/direct/fluorescence | 3,188 |
| cellular MHC/direct/fluorescence | 155 |
| binding assay | 32 |
| cellular MHC/direct/radioactivity | 5 |
| lysate MHC/direct/radioactivity | 4 |

A radioactive dissociation half-life and a fluorescence-polarization half-life are
offset from each other exactly the way IC50 and KD are. Binding models that offset with
six family heads; **`t_half` pools all six methods into one MSE task.** The method is now
captured (`StabilityRecord.assay_method`, `PrestoSample.stability_assay_method`) and
recorded in run stats, so the offset can be modeled — that head work is queued in
`tasks/todo.md`.

Coverage note: hitlist carries fewer stability rows than the merged TSV (Tm 178 vs
1,250). The migration must union the two sources, not replace one with the other.

### 3.3 Kinetics **[partial]**

| | |
|---|---|
| Rows | `koff` 75 merged / 29 hitlist · `kon` 40 merged / 25 hitlist |
| Losses | `koff`, `kon` — `mse`, weight 1.0 |
| Censoring | Fields exist, unused |

Tiny. Present for completeness; not expected to carry gradient signal on its own.
`association constant KA` (2 rows) is parsed by hitlist but not routed by Presto.

### 3.4 Elution / mass spec **[impl]**

| | |
|---|---|
| Rows | 2,781,068 merged · 4,053,693 observations / 1,285,987 peptides in hitlist |
| Record | `ElutionRecord` |
| Heads | `elution_logit`, `presentation_logit`, `ms_logit` (`ms_logit` is the same tensor as `elution_logit` per design S9.3) |
| Losses | `elution`, `presentation` — `bce`, weight 1.0, both keyed on the same `elution` target |
| Multi-allele | Noisy-OR MIL bags when a sample has several alleles; `elution`/`presentation` table entries are skipped in favour of the bag loss when MIL is active |
| Negatives | Synthetic only — see §4 |

The elution logit is `softplus(w_p)·presentation_logit + softplus(w_d)·ms_detectability_logit + bias`
(`ElutionHead`, `models/heads.py`). The detectability half of that sum currently has no
supervision of its own.

### 3.5 Processing **[partial]**

| | |
|---|---|
| Record | `ProcessingRecord` — the only record type that has always had flank fields |
| Head | `processing_logit` (class-probability mixture of `processing_class1/2_logit`) |
| Loss | `processing` — `bce`, weight 1.0 |

Two structural gaps. First, the output is a **single scalar per MHC class**: there is no
separable N-terminal versus C-terminal cleavage score, even though the two termini are
produced by different biology (proteasome C-term, ERAP N-term trimming). Second, on the
merged-TSV path the flanks were never populated — `ProcessingRecord` was constructed
without them, so the model saw one `<MISSING>` token in both flank segments. Fixed by the
hitlist migration, which supplies `n_flank`/`c_flank` at ~98%.

### 3.6 T-cell response **[impl, legacy conditioning]**

| | |
|---|---|
| Rows | 212,190 merged |
| Heads | `tcell_logit`, `immunogenicity_logit`, plus six context CE heads |
| Losses | `tcell`, `immunogenicity` (`bce`); `tcell_assay_method`, `tcell_assay_readout`, `tcell_apc_type`, `tcell_culture_context`, `tcell_stim_context`, `tcell_peptide_format` (`ce`) |

`TCellAssayHead` consumes a `tcell_context` dict of six categorical embeddings and uses
it for a bias, a multiplicative gate, and mixing gates. This is **more context-conditioned
than the outputs-only contract allows** and is flagged as legacy in the head's own
docstring, in the contract, and in `design.md`. Do not treat it as precedent.

### 3.7 TCR evidence **[impl]**

| | |
|---|---|
| Rows | 166,321 merged (VDJdb, 10x, McPAS) |
| Losses | `tcr_evidence` (`bce`, weight **0.05**), `tcr_evidence_method` (`bce`, weight **0.02**) |

Deliberately down-weighted: positive-only evidence with no matched negatives.

### 3.8 MHC identity auxiliaries **[impl]**

`mhc_class`, `mhc_species`, `mhc_a_fine_type`, `mhc_b_fine_type` — all `ce`, weight
**0.1**, targets derived in-code rather than collated. `species_of_origin` (`ce`) and
`foreignness` (`bce`) at weight 1.0.

### 3.9 Bulk non-MHC proteomics **[impl]**

The new branch. Shotgun whole-cell MS, deliberately kept in a separate hitlist index so
it can never be confused with immunopeptidomics.

| | |
|---|---|
| Rows | 2,047,003 peptide-level (661,142 unique peptides); 222,285 protein-level |
| Cell lines | A549, HCT116, HEK293, HeLa, Jurkat, K562, MCF7, MDA-MB-231, THP-1 |
| Coordinates | `uniprot_acc`, `start_position`, `end_position` all 100% → junction context with zero mapping ambiguity |
| Label | Graded: `n_fractions_in_run` ladder (12/14/39/46/50/70) and `n_replicates_detected` (1/2/3), both 100% populated |
| Negatives | Excision negatives are real (mismatched-enzyme relabeling); detectability negatives need the in-silico digest (hitlist#361) |
| Record | `BulkMSRecord` (`data/bulk_ms.py`) |
| Heads | `ms_detectability_logit`, `excision_logit` (+ `excision_n_terminus_score` / `excision_c_terminus_score` / `excision_length_score`) |
| Losses | `ms_detectability` (`bce`, soft targets, weight 0.5), `excision` (`bce`, weight 1.0) |

Four proteases with machine-readable cleavage rules, all on one cell line, one
instrument, one search engine:

| enzyme | rule | rows | proteasome analog |
|---|---|---:|---|
| Trypsin/P | C-term K/R, not before P | 1,962,502 | beta2 trypsin-like |
| Chymotrypsin | C-term F/W/Y/L/M, not before P | 109,315 | beta5 chymotrypsin-like |
| LysC | C-term K, P allowed | 101,785 | beta2 trypsin-like |
| GluC | C-term E (and D in bicarbonate), not before P | 95,686 | beta1 caspase-like |

The three proteasome catalytic specificities each have a pure single-enzyme analog here,
which is why the machinery axis is chemistry transfer rather than only a control.

### 3.10 Antigen-processing perturbations **[planned]**

The only *interventional* data in the corpus: 242 of 688 curated `ms_samples` carry an
APM flag, 125 have a per-sample perturbation, spanning 29 PMIDs / 1.34M observations.
Named panels include the HAP1 12-gene KO panel, Guasp 2019 ERAP1/ERAP2/double KO, and
Koumantou 2019 ERAP1 inhibition.

Consumer trap, worth repeating wherever these columns are used: `apm_<gene>_perturbed` is
ORed with the **parent study's** perturbation list, so a WT control inside a KO study
inherits the flag. `condition_category` is the per-sample truth; `study_apm_perturbed` is
the study-level roll-up. Featurizing the wrong one inverts the control arm.

## 4. Negatives, and where they come from

| family | negative source | kind |
|---|---|---|
| binding | synthetic (`peptide_scramble`, `peptide_random`, `mhc_scramble`, `mhc_random`, `no_mhc_alpha`, `no_mhc_beta`) | synthetic |
| elution / MS | synthetic pairings + data-conditional hard allele-mismatch pairs | synthetic |
| processing | `flank_shuffle`, `peptide_scramble` | synthetic |
| downstream | cascade projection of binding negatives | synthetic |
| **bulk MS** | **in-silico digest minus observed** | **real** |

The bulk branch is the only place Presto has genuine negatives. Three conditions make an
absence informative, all derivable from recorded metadata: the candidate must be inside
the arm's search space (outside it means *unsearchable*, not undetected), its parent
protein must have been observed (removing the abundance confound), and the arm must be
deep enough for absence to mean anything.

## 5. The assay descriptor

One factorization, shared across every quantitative family, because
`purified MHC/direct/fluorescence` means the same thing whether it produced an IC50 or a
half-life.

| axis | vocabulary | values |
|---|---|---|
| type | `BINDING_ASSAY_TYPES` | unknown, KD, KD_PROXY_IC50, KD_PROXY_EC50, IC50, EC50, OTHER |
| method | `BINDING_ASSAY_METHODS` | 9 composite methods + unknown/OTHER |
| prep | `BINDING_ASSAY_PREP` | PURIFIED, CELLULAR, LYSATE, BINDING_ASSAY |
| geometry | `BINDING_ASSAY_GEOMETRY` | COMPETITIVE, DIRECT, T_CELL_INHIBITION |
| readout | `BINDING_ASSAY_READOUT` | RADIOACTIVITY, FLUORESCENCE |

Parsed by `_factorize_binding_assay_method` (`data/collate.py`), embedded by
`AffinityPredictor`'s factorized tables (`models/presto_modules.py`), consumed **only**
inside `AssayHeads` residual heads — never by the trunk. Stability and kinetics rows now
fall back to their own recorded method, so they carry real prep/geometry/readout instead
of `unknown`.

The type axis carries `T_HALF`, `TM`, `KOFF` and `KON` alongside the affinity families,
so a stability row is distinguishable from an unrecognized binding assay. Entries were
appended, keeping existing indices stable; `Presto._grow_appended_embeddings` extends
older checkpoints on load.

Upstream vocabulary: hitlist's `response_measured` is the same controlled vocabulary the
merged TSV carries as `value_type`, so it maps 1:1 with no translation layer. Units are
consistent per response family and are asserted at ingest — a unit change upstream fails
loudly rather than training at the wrong scale.

## 6. Honest list of non-comparabilities

Things currently pooled that should not be, ordered by how much data they affect:

1. **`t_half` across six methods** (§3.2) — ~10K rows, two methods at comparable volume.
   (Censoring is now honored; the per-method offset is the part still outstanding.)
2. **Elution positives across acquisition arms** — instrument is 1.4% populated on the MS
   side, so the acquisition covariate story only works from the bulk direction today.
3. **`Tm` across sources** — merged and hitlist disagree by 7x on row count; unresolved.
4. **Qualitative binding is dropped entirely** — rows with no numeric value are skipped,
   matching the merged-TSV contract. hitlist's binding index holds 895,785 rows including
   qualitative tiers; `MSEWithInequalities`-style inequality targets would recover them.
5. **`presentation` and `elution` share one target** with two heads and two losses.

## 7. Where this lives in code

| concern | location |
|---|---|
| record types | `data/loaders.py` |
| sample/batch contract | `data/collate.py` (`PrestoSample`, `PrestoBatch`) |
| target collation | `TARGET_SPECS` (`data/collate.py`) |
| assay vocabularies | `data/vocab.py` |
| descriptor factorization | `_factorize_binding_assay_method`, `_collate_binding_context` |
| loss registry | `LOSS_TASK_SPECS` (`scripts/train_synthetic.py`) |
| censored regression | `training/losses.py` |
| output heads | `models/heads.py`, `models/presto_modules.py` |
| hitlist adapter | `data/hitlist_source.py` |
| held-out per-task metrics | `training/holdout_eval.py` |
| excision head | `ExcisionHead` (`models/heads.py`) |
| machinery vocabulary + rules | `data/vocab.py` (`EXCISION_*`) |
| bulk corpus | `data/bulk_ms.py` |
| latent topology | `Presto.EXPANDED_LATENT_*` (`models/presto.py`) |
| contrast co-batching | `BalancedMiniBatchSampler._build_contrast_index` (`data/loaders.py`) |
| upstream curation | `hitlist` (`generate_training_table`, `bulk_proteomics`, `apm`) |
