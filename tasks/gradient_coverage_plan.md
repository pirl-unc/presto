# Plan: full gradient coverage, clean curation, brev end-to-end

Goal (2026-08-29): data cleanly curated; tasks well organized; **every submodule
receives gradient, training signal, or a documented pretraining step**; presto
trains end to end on brev.

## Measured starting point

With every modality present in one batch (elution class I + class II, binding
class I + class II, stability, kinetics, processing, T-cell, TCR evidence,
bulk MS), **100,863 of 512,162 parameters (19.7%) receive no gradient.**

Reproduce: `scratchpad/audit.py`.

## Triage — dead parameters by cause

| cause | params | examples | disposition |
|---|---|---|---|
| **A. Alternate config branch never selected** | ~76k | `pep_pos_concat_*`, `pep_abs_pos`, `groove_1_abs_pos`, `presentation_class*_mlp`, `core_window_score_class*`, `binding_direct_segment_gate`, `groove_frac_mlp` | Do not allocate what the active config cannot reach. Allocating an unreachable branch makes a model look larger than it is and silently ships untrained weights. |
| **B. Needs a modality absent from the batch** | ~5k | `class2_pfr_score`, `immunogenicity_cd4_latent_head`, `tcr_evidence_method_head` | Verify each is reachable with the right record; pin it with a test that supplies that record. |
| **C. Output head with no loss term** | ~500 | `recognition_cd8_head`, `recognition_cd4_head`, `foreignness_head`, `species_of_origin_head` | These are *scored* in the held-out pass but never trained — a metric on an untrained head is noise. Either supervise or stop reporting. |
| **D. Structurally masked by design** | ~200 | `excision_head.p1_profile_c` under `pinned_mask`, `mixture_logits` | Legitimate. Document and pin so it stays a known, deliberate gap. |

## Tasks

- [x] 1. Curation: map hitlist condition categories on evidence (infection ->
      type I IFN / TLR, activation its own token). `ifn_type1` went 0 -> 21,394
      rows; stimulated rows 71,040 -> 164,610.
- [x] 2. Curation: every hitlist condition category reviewed, so the
      unmapped-category signal means "genuinely new" and nothing else.
- [x] 3. Classified with evidence (each verified by driving the real batch).
- [x] 4. Category A closed: `POSITION_MODE_COMPONENTS` drives both what the
      composer reads and what the constructor allocates; collapsed-topology,
      class-specific-core and direct-segment modules are allocated only when
      their mode selects them. **19.7% -> 3.1% dead.**
- [x] 5. Category B: each confirmed reachable by supplying the right record
      (a class II T-cell record revived `immunogenicity_cd4_latent_head`; a
      class II processing record revived `class2_processing_predictor`).
      Remaining B entries are listed with the data they need.
- [ ] 6. Category C — **needs a decision, see below.** Not made unilaterally:
      wiring these changes what the model computes.
- [x] 7. Category D pinned as deliberate.
- [x] 8. `tests/test_gradient_coverage.py` asserts the dead set equals a
      categorized allowlist. A new untrained parameter fails; fixing one
      requires deleting its entry, so the fix shows in the diff.
- [x] 9. All 31 LOSS_TASK_SPECS have a head; `foreignness` and
      `species_of_origin` do have losses (an earlier reading that they did not
      was wrong -- the audit batch simply lacked their targets).
- [ ] 10. Brev end-to-end run on a single cheap GPU, held-out pass completing,
      artifacts written per the experiment contract.

## Rules for this work (learned the hard way on this branch)

- Prove a parameter is trained by driving records -> dataset -> collator ->
  `compute_loss` -> `backward()` and reading `p.grad`. Never infer from code.
- Adding a field/tensor is not the same as populating it. Verify it arrives.
- Never delete a test that pins a known gap in order to close the gap.


## Open decision: category C, computed but never consumed

Four groups are computed every forward, published in `outputs`, and read by no
loss. They are not merely untrained -- they are **intended features that were
never connected**:

| what | params | intent |
|---|---|---|
| `assay_{type,prep,geometry,readout}_embed` + `factorized_proj` | ~1.2k | absorb assay bias so affinity is comparable across assays |
| `sequence_summary_proj` | ~4.1k | direct pep/MHC summary into the binding path |
| `binding_stability_score_head` | ~0.6k | a stability readout, wired to no stability loss |
| `recognition_cd{8,4}_head` | 66 | publish CD8/CD4 recognition probabilities |

The last is the most user-visible: `recognition_cd8_prob` is emitted as a
probability from an untrained projection.

Wiring any of them changes what the model computes and how it is calibrated,
so it is a modelling decision, not a cleanup. Options per group: connect to an
existing loss, add supervision, or stop publishing the output.
