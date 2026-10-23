# Real-sequence variable-span computational comparison

Registered October22,2026; Codex / GPT-6. This is a forward/backward computation
and estimator-equivalence study. It does not fit biological measurements or
claim concentration recovery. The pending shared recipe is not bypassed.

Use the verified full matched-parent sequence files from the MAPP human2026_03
and PXD057588 mouse2026_03 reference audits. Freeze their hashes before selection.
From each source, select one distinct full sequence in each length interval
[64,128),[256,512),[1024,2048),[2048,4096), choosing the lexicographically smallest
sequence SHA256 then accession. Assert every band is represented; do not alter
bands after timing. Eight real full proteins; no truncation, concatenation or
sequence alteration. Selection uses no assay values. Save selected records.

Policy: fixed one-hot/prefix-sum features, start and conditional-length tanh MLPs,
32hidden units, no final output bias, context_size2 with fixed synthetic vector
[0.25,-0.5]. These numbers do not stand for species, cytokines or gene loss.
Seed42 weights held unchanged. Length supports1–40 and1–53, including short and
nonbinding candidates; both policy shapes share the same parameter dimensions.
CPU, four intra-op threads, one inter-op thread, float32. No GPU/Modal.

Conditions for each of16parent/support cases: exact expected cost; dense
REINFORCE8/64; lazy REINFORCE8/64. Synthetic energy is a fixed function of span
composition, normalized position and length. Exact loss is sum q*(logq+energy),
equivalent to reverse KL to normalized exp(-energy) up to a parameter-independent
constant. Sampled loss uses fresh draws and the leave-one-out baseline of
logq+energy. No clipping, standardization, sample reuse or reference KL.
This mathematical energy is not a biological oracle or assay label.

Measure prepare+forward+cost+backward+gradient clearing, with no optimizer update,
so weights remain matched. Per case/condition use4warmups then12measured calls.
Seeded random interleaving of condition order across repetitions; fresh identical
per-count RNG seeds for dense/lazy comparisons. Synchronize only if required by
the local CPU backend. Report all raw timings, medians, start-network rows,
conditional-network rows including masked cells, valid spans and cost queries.
Forward hooks check actual network rows against reported work. Record process
peakRSS as a family-wide measure, not a per-condition allocation comparison.

Before timing, independently enumerate support and verify dense/lazy selected
log probabilities and all parameter gradients using identical fixed spans.
Record actual seed-matched sample equality separately. For the longest-band
human/mouse proteins at max_length40, collect128independent fresh gradient
estimates per sample count. Compare their mean, coordinate SE, relative L2 error
and cosine to the exact gradient. These probes are diagnostics, not a required
claim that every noisy estimate is close. A separate exact probability check
must pass; do not tune model/selection on timing or gradient-error outcomes.

Preserve source/model code snapshots, selected inputs, config/checkpoint hashes,
per-call times and sample traces, gradient probe vectors and summaries, runtime
and environment. Independently replay measured work, source selection, gradient
and timing summaries. No predictive train/validation/test split or per-assay
prediction dumps apply because this is not a fitted predictive experiment.
Keep prior exact/REINFORCE experiment records unchanged. Publish whether lazy
scoring actually saves CPU time and conditional network work, and preserve the
remaining O(L) start-scoring and autograd-memory limitations.
