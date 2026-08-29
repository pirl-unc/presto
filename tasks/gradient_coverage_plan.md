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
- [ ] 3. Classify each dead parameter into A/B/C/D **with evidence**, not by
      reading names. A parameter is only "config-dead" once the config that
      would reach it is identified.
- [ ] 4. Category A: stop allocating unreachable branches under the default
      config, or make the selecting flag explicit.
- [ ] 5. Category B: add the record types that reach them; pin with tests.
- [ ] 6. Category C: decide supervise-or-remove per head. Do not leave a head
      that is reported but untrained.
- [ ] 7. Category D: pin as deliberate.
- [ ] 8. A standing test asserting the dead-parameter set equals the documented
      D set — so this cannot regress silently the way gap 2 did.
- [ ] 9. Tasks/organization: confirm every LOSS_TASK_SPEC has data and a head.
- [ ] 10. Brev end-to-end run on a single cheap GPU, held-out pass completing,
      artifacts written per the experiment contract.

## Rules for this work (learned the hard way on this branch)

- Prove a parameter is trained by driving records -> dataset -> collator ->
  `compute_loss` -> `backward()` and reading `p.grad`. Never infer from code.
- Adding a field/tensor is not the same as populating it. Verify it arrives.
- Never delete a test that pins a known gap in order to close the gap.
