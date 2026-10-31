# Continue actual MHC-II comparisons after publication

Codex / GPT-6,2026-10-27. Full goal and `mhc2_improvement_spec.md` unchanged.
Previous goal turn is **progress**: published e04dfcf, recovered native101,
closed a primary-source audit and filed/read back Hitlist858. Current start
confirmed live handles96044/12902/29950; final-head CI40432992416 remains active.

## Authoritative scheduling evidence

The remaining-arms controller96044 reached its30minute admission deadline with
no launched action. Its final admission sample has20,031,225,856available bytes,
below its added20GiB continuous60second scheduling threshold. Preserve that
terminal state and all samples. Original action limits were not reached because
no fit ran. Do not retry a native fit automatically, lower its10GiB runtime RAM
floor, alter its4GiB RSS ceiling, or disturb the older convergence jobs.

The original2036comparison independently specifies binding-only, sequence and
combined logistic controls. These need only the frozen IC50 parent and immutable
MS cohort, not successful native102/null fits. Move this independent comparator
stage ahead of the deferred native fits. This changes scheduling, not labels,
splits, model inputs, coefficients, optimization, selection, tests or gates.
The original logistic script remains byte-identical except for the previously
registered serialization-transition wrapper. No held-out result changes this
order or its fitting recipe.

Run the controls once under their **original** action contract: localCPU,2threads/
1interop,13GiB available RAM and15GiB free disk to start,10GiB RAM/disk runtime
floors,4GiB RSS and2h/action. The extra20GiB/60second scheduling threshold belonged
to the now-failed all-arm continuation; this comparator-only action does not
reuse that controller or claim to satisfy that extra threshold. Preserve this
explicit scheduling deviation. No new native/censor fit overlaps the controls.
Stop on any feature/fit failure; no automatic retries or installs/cloud use.
Before each comparator action, record at most15minutes of admission samples and
require the original13GiB RAM startup reserve continuously for30seconds. This
does not lower the action-time RAM floor; if the start reserve is unavailable,
finish the wait with no launch. Disk start/floor checks stay in the action guard.

Freeze a new comparator-only invocation and controller receipt before launch.
Require missing comparator output directory, unchanged source/runtime hashes,
fixed parent checksum and original per-example source IDs. Preserve full train/
validation/test/external predictions and training-only standardization. Verify
these saved controls before their use; do not promote the first native seed or
select a replacement based on held-out results. The fixed two-seed ensemble,
permutation diagnostic and full prespecified bootstrap remain mandatory.

## Independent work and closure

- Follow old seed44/permutation and final-head CI through terminal handles.
- Complete and independently replay the frozen comparator stage.
- Resume remaining native arms only by a separately recorded action after
  suitable headroom exists; preserve all prior admission/fitting failures.
- Recover the150step censored-binding snapshot in its unchanged prior runtime
  when it can run without overlapping a new native fit.
- Qualify actual condition sources and fit/ablate the supported factors; source
  issues alone do not complete that track. Status reads853/816 found no new data
  or upstream completion; they are not a reason to repeat exhausted retrievals.
- Keep registry, source-linked predictions, snapshots and published evidence
  current. Do not mark the goal complete or blocked while progress remains.
