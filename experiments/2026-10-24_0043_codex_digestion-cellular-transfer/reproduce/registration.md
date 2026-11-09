# Registered transfer qualification and bounded comparison

Registered before biological target arithmetic or numerical probes. The copied
plan defines physics, dataset, fitting and scope. Additional frozen choices:

- Float64 NumPy mathematical reference and Torch analytical fitter, local CPU one
  thread, unchanged shared environment. All three exact arms plus one named
  wider-context hybrid; uniform observed composition remains a no-fit baseline.
- Independent branching-generator resolvents and numerical quadrature for every
  span in lengths 2/4/6 at delta 1e-4/.01/1/100. Value rtol1e-8/atol1e-10;
  log-delta central finite differences h1e-5, rtol1e-6/atol1e-8;
  residue-weighted mass identity rtol1e-9. No output floors.
- Enumerated length6 complete-partition age mixture: normalized proposal,
  importance/Rao-Blackwell gradient expectation absolute error <=1e-10.
  Fixed 65,536 conditional draws, seed42, binomial count and conditional-age mean
  checks <=6 standard errors (bin-count variance adds1 to avoid division by zero).
- K512 is per training parent/condition record, not per target peptide; 16 fresh
  groups at log(.01) and +/-1. Within each example center dlogF/dlogdelta by its
  arithmetic span mean and add the exact coefficient-sum baseline contribution.
  For multiple peptide occurrences, distribute the candidate gradient coefficient
  proportional to occurrence F; preserve source candidate labels only once.
  The proposal is .9 normalized absolute coefficient*centered-derivative +.1 uniform.
- Relative mean gradient error <=.2, relative RMS <=.5. If |exact gradient|<=1e-6,
  instead require absolute mean error<=2e-7 and RMS<=5e-7. No post-probe denominator
  adjustment. A failure blocks only sampled training; keep all exact fits.
- Target-independent preflight uses centered predicted logF squared as a synthetic
  cost-only residual; never treated as a biological label or reported skill.
  Three full training-support gradient/sampler repetitions per arm; projected
  time = max repetition *2000 *3 +60. Exact caps max(120,ceil(projected)) <=900s;
  hybrid max(600,ceil(projected)) <=7200s. An over-budget arm is not launched.
  Qualification cap900s; all runs4GiB. Freeze measured caps before real fitting.
- No optimizer-state adaptation, clipping, restart, extension or checkpoint
  selection; all2000 scalar Adam states plus RNG state before steps1/100/1000/2000
  saved. Independent final NumPy replay, all scalar optimizer arithmetic and the
  four fixed gradient/RNG states. Training/validation metrics and prediction rows,
  per-parent and exposure-stratified metrics; test parents excluded.

Feature/exposure audit runs before any target arithmetic. No source filters or
curation changes. Six training and three development digestion parents reported
separately for exact identity/construct containment/shared measured peptides.
No feature-space support flag is claimed to establish biological independence.
