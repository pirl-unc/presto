# Presto — active work

Working file. One screen, current only. History lives in
`archive/todo_history_through_2026-08.md` (9,039 lines, append-only through
2026-08-29); detailed specs live in the per-topic files listed at the bottom.

Last reorganized: 2026-08-29.

---

## In flight

- [x] **Brev end-to-end run.** Done — `experiments/2026-08-29_1953_claude_brev-e2e-class1`.
      Binding Spearman 0.796 (n=17,885) on a peptide-disjoint split.
- [ ] **Full-modality brev run.** The completed run trained no immunogenicity,
      recognition or TCR heads: hitlist has no T-cell evidence. The merged TSV
      is staged at `/root/presto_data/merged_deduped.tsv` and
      `reproduce/launch_full_modality.sh` is ready.
- [ ] **Synthetic negatives are too easy.** elution/presentation AUPRC 1.0000
      and excision 0.9998 with *zero* pos/neg overlap: the metric measures
      "real peptide + real MHC", not presentation. Either score on real
      negatives only, or make the synthetic ones hard.
- [ ] **`ms_detectability` learned nothing.** AUPRC 0.9071 against a 0.9042
      base rate — lift +0.003. The arm the excision/detectability design exists
      for. Needs `dual_corpus_transfer_set` (24,125 peptides).
- [x] **Category C** — closed. All four groups wired; the category is empty.
- [x] **Many-output refactor** — assay and condition axes are output tracks,
      not inputs. Invariance verified in `tests/test_many_output_contract.py`.
- [ ] **Cell line / tissue are not represented at all.** `ElutionRecord`
      carries them; they never reach `PrestoSample`. 172 distinct lines, and
      the design's intended proxy for expression profile. Neither input nor
      output track — a gap, not a deviation.
- [ ] **Re-run the brev baseline.** The 2026-08-29 numbers predate the
      recognition→immunogenicity edge, the assay panels, and the removal of
      all condition inputs. Not comparable.

## Open — model contract (`docs/model_io_contract.md` S8)

- [ ] **Gap 5** — merge `processing` and `excision`. Changes an existing task's
      semantics; wants Stage 4 arm C first.
- [x] **Gap 7** — closed. The seven context arguments are gone from
      `TCellAssayHead.forward`; predictions are context-free marginals and the
      panel carries per-condition structure. The before/after measurement is
      still owed — this changes T-cell predictions by construction.

Gaps 1, 2, 3, 4, 6 are closed. Gap 2 took three attempts and two of those were
wrongly reported as closed — see `lessons.md`.

## Open — experiments

- [ ] Stage 4 factorial (arms A–E) at GPU scale. Never run.
- [ ] Stages 0a/0b/1 at scale.
- [ ] Measure detectability transfer on `dual_corpus_transfer_set`
      (24,125 peptides). Builder exists; the measurement is the experiment.
- [ ] Binding Spearman at real scale. A toy run showed −0.656 at n=77, which is
      too small to mean anything; the question is open, not the answer.
- [ ] 24 MB of per-example dumps in
      `experiments/2026-03-28_claude_class1-best-hits/results/` are untracked.
      AGENTS.md wants them in the experiment dir; committing 24 MB to history
      is a call for a human.

## Open — data and modelling

- [ ] `t_half` per-method output structure. Censoring is done; the six-method
      offset is not.
- [ ] In-silico digest negatives (hitlist#361).
- [ ] Triage the 362 unchecked boxes in the archive. Most are stale — boxes
      never ticked on finished work — but they have not been audited, and
      saying otherwise would be a guess.

## Recently closed (2026-08-27 → 08-29)

- Gap 2 closed for real: in-vivo excision → class I presentation edge; all 501
  parameters take gradient from elution labels.
- Gradient coverage 19.7% → 3.1%; `tests/test_gradient_coverage.py` pins the
  remainder against a categorized allowlist.
- hitlist condition categories mapped on evidence: `ifn_type1` 0 → 21,394 rows,
  stimulated rows 71,040 → 164,610.
- `inducer` → `stimulus`, `basal` → `none`, `ifn_ab` → `ifn_type1`.
- Sequence validation at every ingest path, against the tokenizer vocab.
- Held-out pass now scores the same function training optimizes.
- MIL instance-cap crash; curriculum unfreezing; zero-init clobber.

---

## Where things live

| file | what |
|---|---|
| `gradient_coverage_plan.md` | gradient audit, categories, open decision |
| `lessons.md` | corrections and the rules they produced |
| `protease_detectability_spec.md` | excision / detectability design |
| `learning_refactor_plan.md`, `learning_refactor.md` | learning-scheme refactor |
| `affinity_followup_plan.md`, `focused_affinity_improvement_plan.md` | affinity work |
| `class2_register_design_benchmark.md` | class II register |
| `mhc_processing_plan.md`, `tcr_evidence_spec.md`, `receptor_removal_plan.md` | per-topic specs |
| `runtime_speedup_plan.md` | performance |
| `status_summary.md`, `training_analysis.md`, `data_and_scripts_audit.md` | analyses |
| `archive/todo_history_through_2026-08.md` | everything before this reorganization |
