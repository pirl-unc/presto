# Presto TODO / Implementation Status Audit

This file is the implementation-status matrix against:
- `docs/design.md`
- `docs/training_spec.md`
- `docs/tcr_spec.md`

Legend:
- `Implemented`: in canonical code path now.
- `Partial`: present but not fully wired in canonical unified path.
- `Planned`: not implemented in canonical path.

Audit date: 2026-08-26

## 1) Data Sources Matrix

| Data source | Registry/download | Parsed | Used in canonical unified training | Status | Notes |
|---|---|---|---|---|---|
| Cross-source merged TSV (`merged_deduped.tsv`) | Produced by `presto data merge` | Yes | Yes (default required input) | Implemented | canonical training input; raw fallback only when explicitly enabled |
| IEDB MHC ligand | Yes | Yes | Yes | Implemented | binding/kinetics/stability/processing/elution |
| IEDB T-cell | Yes | Yes | Yes | Implemented | includes assay context fields |
| CEDAR MHC ligand | Yes | Yes | Yes | Implemented | merged with IEDB when available |
| CEDAR T-cell | Yes | Yes | Yes | Implemented | merged with IEDB when available |
| VDJdb | Yes | Yes | Yes | Implemented | positive TCR:pMHC tuples |
| 10x VDJ | Yes | Yes | Yes | Implemented | chain attribute supervision |
| IMGT/HLA | Yes | Yes | Indirect | Implemented | sequence resolution + class/species support |
| IPD-MHC | Yes | Yes | Indirect | Implemented | non-human MHC resolution |
| IMGT TRAV/TRBV refs | Yes | Registry/download only | No | Partial | available as auxiliary references |
| IEDB B-cell | Yes | Yes | No | Partial | parser exists; not canonical supervision |
| CEDAR B-cell | Yes | Yes | No | Partial | parser exists; not canonical supervision |
| McPAS-TCR | Yes | Yes | Yes | Implemented | included in merged `tcr_pmhc` supervision |
| PIRD | Registry only (current URL unstable/404) | No canonical parser | No | Partial | harmonization pending once stable machine-readable export is available |
| STCRDab | Yes | File downloaded; no canonical assay parser | No | Partial | structural resource; adapter pending |
| IMGT/LIGM-DB | No | No | No | Planned | not integrated |
| IMGT/3Dstructure-DB | No | No | No | Planned | not integrated |
| hitlist curated MS + binding indexes | Yes | Yes | Yes (`--data-source hitlist`) | Implemented | supplies `n_flank`/`c_flank`, which the merged TSV has never had |
| Bulk non-MHC shotgun proteomics (Bekker-Jensen, CCLE) | Yes | Yes | Yes (`--bulk-ms`) | Implemented | four protease arms; identifies MS detectability |

## 2) Losses / Priors Matrix

| Loss or prior | Status | Notes |
|---|---|---|
| Balanced mini-batch sampler across assay/source/label/allele/synthetic-kind | Implemented | default in unified training (`--balanced-batches`) |
| Binding censor-aware regression | Implemented | KD supervision path |
| KD/IC50/EC50 assay losses | Implemented | shared KD calibration path |
| Kinetics/stability losses (`kon`, `koff`, `t_half`, `Tm`) | Implemented | active when labels present; `t_half`/`Tm` are censor-aware |
| Processing BCE | Implemented | sparse-label aware |
| Elution BCE | Implemented | canonical instance path |
| MS BCE | Implemented | presentation-linked |
| T-cell BCE | Implemented | `tcell_logit` and immunogenicity path |
| T-cell context CE heads | Implemented | method/readout/APC/culture/stim |
| MHC class/species CE | Implemented | inferred from MHC sequences |
| Chain attribute CE | Implemented | chain species/type/phenotype |
| Synthetic binding negatives | Implemented | includes no-beta2m negatives |
| Synthetic elution negatives + hard pair negatives | Implemented | allele mismatch hard negatives included |
| Synthetic processing negatives | Implemented | peptide+flank negatives |
| Cascaded downstream negatives | Implemented | binding negatives propagated to downstream tasks |
| Consistency priors (cascade/assay/no-beta2m/tcell context/upstream) | Implemented | weighted regularizers |
| Unified smooth ramp scheduling | Implemented | canonical production choice |
| Multi-allele bag (MIL/Noisy-OR) training loss | Implemented | elution/presentation/MS bag loss wired via collator + unified loss path |
| Multi-TCR bag training loss | Planned | not first-class yet |
| Contrastive InfoNCE in canonical unified path | Partial | utilities exist; not always central |
| Similarity-mined hard negatives in canonical unified path | Partial | implemented in utilities, partial canonical use |

## 3) Heads / Latents / Outputs Matrix

| Requirement | Status | Notes |
|---|---|---|
| Single stream `Nflank|peptide|Cflank|MHC_a|MHC_b` | Implemented | canonical forward in `models/presto.py` |
| Segmented latent-query DAG (`processing_class1/processing_class2/binding_affinity/binding_stability/presentation_class1/presentation_class2/recognition_cd8/recognition_cd4/immunogenicity_cd8/immunogenicity_cd4`) | Implemented | Two topologies. `--latent-topology expanded` gives each design-doc latent its own query and segment scope (`Presto.EXPANDED_LATENT_*`); the historical default `collapsed` folds them into five cross-attention latents plus projections and MLPs, exposing the same twelve names downstream. Segment access is asserted behaviorally in `tests/test_latent_topology.py`. |
| Machinery-conditioned excision head (`s_N`/`s_C`/`s_len`) | Implemented | `ExcisionHead`; in-vitro protease P1 rules pinned, proteasome a learned convex mixture over them |
| MS detectability supervision | Implemented | `ms_detectability` loss from the non-MHC shotgun corpus; previously the latent had no loss at all |
| Held-out per-task metrics in the canonical trainer | Implemented | `training/holdout_eval.py`; metric family derived from `TaskLossSpec.loss_type`, so a new task is scored without extra wiring. `presto train unified` now writes `summary.json` and `val_predictions.csv`. |
| Full-sequence MHC encoding | Implemented | canonical |
| Core/PFR decomposition outputs (`core_start_*`, `core_length`, `npfr_length`, `cpfr_length`) | Implemented | exported from canonical path |
| `*_vec` outputs (`pmhc_vec`, `tcr_vec`, `mhc_a_vec`, `mhc_b_vec`, `latent_vecs`) | Implemented | canonical naming convention |
| `*_logit` / `*_prob` paired outputs for core heads | Implemented | processing, binding, presentation, recognition, immunogenicity, ms/elution, tcell |
| Independent processing pathways | Implemented | class I and class II |
| Class-symmetric binding path | Implemented | shared binding latent module + class-probability-calibrated class1/class2 binding logits |
| Class-specific presentation heads | Implemented | additive logit composition |
| CD8/CD4 recognition heads | Implemented | separate latent readouts |
| CD8/CD4 immunogenicity heads + mixture | Implemented | class-gated composition |
| Optional TCR matcher path | Implemented | `match_logit` and `match_prob` |
| Multi-allele Noisy-OR inference | Implemented | predictor-level bag aggregation |
| Multi-allele first-class bag training | Implemented | dataset/collator/train loop support bag-level Noisy-OR MIL |
| Multi-TCR bag inference/training | Planned | tracked backlog |

## 4) Priority Backlog

1. Run the Stage 4 factorial (arms A-E) to decide whether detectability/excision
   supervision earns its place; see `tasks/todo.md`.
2. Per-method output structure for `t_half` (six non-comparable methods currently pooled)
   and censor-aware stability loss.
3. Append `T_HALF`/`TM`/`KOFF`/`KON` to `BINDING_ASSAY_TYPES` with the checkpoint pass.
4. In-silico digest negatives for detectability (hitlist#361).
5. Add first-class multi-TCR bag training/inference wiring.
6. Promote optional sources (McPAS, PIRD, STCRDab, IMGT refs) to canonical only after schema/dedup audit.
7. Expand canonical bag-level supervision beyond elution/presentation/MS where reliable multi-instance labels exist.

## 5) Canonical Doc Set

Normative docs:
- `docs/design.md` — architecture specification
- `docs/training_spec.md` — training specification
- `docs/tcr_spec.md` — TCR encoder specification

Supporting notes:
- `docs/notes/`
