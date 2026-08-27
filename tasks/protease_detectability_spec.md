# Protease-Conditioned Excision + MS Detectability

Spec for reorganizing Presto around a shared peptide-excision abstraction and an
identifiable MS-detectability term, supervised by non-MHC shotgun proteomics.

- Status: `draft` — awaiting approval before implementation
- Author: claude
- Date: 2026-08-26
- Related: `docs/design.md` S7.4 (`ms_detectability`), S9.3 (elution head),
  `experiments/model_to_beat.md`

---

## 1. Problem

Presto's `ms_detectability` latent (latent #12, `models/presto.py:170`,
`models/heads.py:1309-1336`) is declared, implemented, and wired into the elution
head as `pres_logit + softplus(w) * ms_detectability_logit` — but **no loss touches
it**. Trained only on immunopeptidomics, the model cannot separate

- `excision` — was this peptide produced at all? — from
- `detect` — would this peptide fly in the mass spectrometer?

Both are functions of peptide sequence, and no MHC-only corpus contains an instance
where one varies and the other does not. The detectability latent is therefore a
free bottleneck that silently absorbs whatever the presentation pathway cannot
explain, and the processing latent has no ground truth at either terminus.

## 2. Generative model

Every MS observation, in either corpus, factorizes the same way:

```
logit P(observed) = excision(peptide, protein, M)   # was it produced?
                  + [ MHC cascade ]                    # MHC branch only
                  + detect(peptide, acquisition)       # would it fly?
                  + abundance(protein, sample)         # was the source there?
```

`M` is the **excision machinery**: proteasome (class I), endo/lysosomal cathepsins
(class II), or an in-vitro protease (trypsin, chymotrypsin, LysC, GluC).

The two corpora are **sibling branches under a shared excision abstraction**, not
stacked layers. Nothing digests an eluted MHC ligand, and the in-vitro protease is
not upstream of MHC — it replaces the entire biological path:

```
shotgun:  L(M = trypsin | chymo | LysC | GluC) ─────────────────────→ D → obs
MHC-I:    L(M = proteasome) → TAP → ERAP trim → bind → present ────→ D → obs
                                                                      ↑
                                                      shared; identifiable only
                                                      because the top branch has
                                                      no MHC term
```

### 2.1 Why this identifies `detect`

A clean 2x2 over the Bekker-Jensen 2017 design matrix:

- **Vary `M`, hold peptide chemistry and sample fixed.** Same cell line, same protein
  pool, same instrument, same lab, same search engine — only the cleavage rule
  changes. Differences between protease arms are attributable to `excision` alone.
- **Vary peptide, hold `M` fixed.** Within one arm, `excision ~ 1` by construction
  for any fully-cleaved in-silico candidate. Differences between observed and
  not-observed candidates are attributable to `detect` alone.

### 2.2 Excision representation

Do **not** one-hot the enzyme. Represent every machinery `M` in one space — a P1
residue-preference vector, a P1' block, and a length/processivity term:

```
excision(peptide, protein, M) = n_terminus_score(N-junction ctx, M)      # N-terminal junction
                                + c_terminus_score(C-junction ctx, M)      # C-terminal junction
                                + missed_cleavage_score(peptide, M)      # missed-cleavage penalty
                                + length_score_value(len, M)
```

**Which steps the score covers.** `c_terminus_score` is endoproteolytic cleavage in both branches.
`n_terminus_score` is cleavage for an in-vitro protease but ERAP *trimming* in vivo — both are
peptide-bond hydrolysis, so they share a head, but they are not the same process.
`missed_cleavage_score` is the missed-cleavage constraint: a peptide can match an enzyme's rule at
both ends and still be an implausible product if it carries internal sites the enzyme
would also have cut (59.4% of observed tryptic peptides carry zero internal K/R, 29.7%
one, 8.6% two). It applies only to machinery with a hard rule — the proteasome is
processive and does not cut at every available site, so penalizing its internal
residues would be wrong biology.

**Explicitly out of scope**, and belonging to downstream latents instead: TAP transport,
MHC binding, surface display.

**A known conflation.** `length_score_value` means different things per branch. For an in-vitro
digest, length is a property of cleavage-site spacing. For class I, the 8-11mer
distribution is set mostly by the MHC groove and TAP, not by the proteasome — so the
in-vivo length term risks attributing MHC selection to the protease. Consider dropping
`length_score_value` for in-vivo machinery once the length preference is carried by the presentation
path.

In-vitro enzymes are pinned (or strongly regularized) to their known profiles from
`hitlist/data/bulk_proteomics/sources.yaml`; the proteasome is a **learned mixture
over the same basis**, initialized near a beta1/beta2/beta5 blend:

| proteasome site | specificity | in-vitro analog in corpus | rows |
|---|---|---|---:|
| beta5 chymotrypsin-like | hydrophobic / aromatic | Chymotrypsin (F/W/Y/L/M) | 109,315 |
| beta2 trypsin-like | basic | Trypsin/P, LysC (K/R) | 2,064,287 |
| beta1 caspase-like | acidic | GluC (E/D) | 95,686 |

beta5 chymotryptic-like activity is why MHC-I ligands have hydrophobic and aromatic
C-termini, so this is chemistry transfer, not just a control. Two properties follow:

- **A known-answer calibration for the processing head**, which has never had one. Be
  precise about the two settings: with `pin_profiles=True` (default) the in-vitro P1
  preferences are *constraints*, so asserting that trypsin prefers K/R checks the
  constraint, not a finding. The honest version of the control is
  `pin_profiles=False` — seed the table with the rule, let training move it, and check
  whether the rule survives contact with data. Run that before trusting the in-vivo
  readout. What is learnable either way: the proteasome mixture weights, the
  per-machinery context corrections, and the whole N-terminal table.
- **Both termini get supervised.** For an in-vitro protease both termini follow one
  rule, so the four-enzyme panel supervises `c_terminus_score`. For class I the C-term is
  proteasome-determined but the N-term is ERAP-trimmed — a different process,
  supervised by the ERAP1/ERAP2 KO panels (section 5). Today neither terminus is
  identifiable.

### 2.3 `M` parameterizes the readout, it does not condition the latent

The obvious implementation — inject `M` as an extra KV token on the `processing`
latent, reusing the existing `extra_tokens` mechanism (`models/presto.py:2556-2562`,
which already routes `apc_cell_type_context` to `{"processing", "pmhc_interaction"}`)
— is **wrong**, for two independent reasons.

**Reason 1: transitive leakage.** Presentation consumes the processing vector
directly (`models/presto.py:2650-2664`):

```python
processing_class1_vec = self.processing_class1_proj(processing_vec)
presentation_class1_vec = self.presentation_class1_vec_norm(
    self.presentation_class1_mlp(torch.cat([processing_class1_vec, interaction_vec], -1)))
```

`M` perfectly separates the two corpora — every MHC row is `M=proteasome`, every
shotgun row is one of the four enzymes. Conditioning the latent therefore puts a
corpus indicator one linear layer away from presentation, where the shortcut
"`M=trypsin` implies never presented" is trivially true and teaches nothing. A
dependency mask cannot prevent this; the leak is through the vector, not the
attention.

**Reason 2: `docs/assay_modeling_contract.md` is normative and already rules on
this.** Its closing line: "elution/MS should follow the same outputs-only assay rule;
if assay/platform structure is modeled later, it must remain output-side rather than
input-conditioned."

So `M` goes on the **output side**, which is the sanctioned mechanism ("assay/task
descriptors may parameterize output heads") and the same shape as the existing
factorized assay embeddings in `AssayHeads` (`models/heads.py:824-899`):

```
c_terminus_score(peptide, protein, M) = head_M(processing_vec)      # M indexes the readout
n_terminus_score(peptide, protein, M) = head_M(processing_vec)
```

The trunk stays strictly sequence-only and learns a machinery-agnostic
representation of junction context; the machinery-specific part lives entirely in the
readout. That is also the better inductive bias: the latent answers "what does this
junction look like", the head asks "would enzyme `M` cut here". Enzyme heads are
pinned to their known profiles; the proteasome head is a learned convex combination
of them.

**Consequence: no contract amendment is required, for `M` or for acquisition.**
Platform/acquisition structure becomes output-side readouts on
`ms_detectability_vec` — which is exactly what `docs/design.md:897-903` already
specifies (`orbitrap_bias`, `tof_bias`) and never implemented.

## 3. What is needed from hitlist

**Most of it already exists.** Verified against the built parquets in `~/.hitlist/`
and hitlist `main` (2026-08-26).

### 3.1 Already available — no work required

| Capability | Where | State |
|---|---|---|
| In-silico digest for all 4 enzymes, canonical rules | `hitlist.proteome.digest(seq, enzyme, min_len, max_len, max_missed)` (#104) | Landed. Enzyme strings match `sources.yaml::digestion_enzyme`; docstring documents the exact positives/negatives recipe |
| Observed bulk peptides with filters | `hitlist.bulk_proteomics.load_bulk_peptides(cell_line, gene_name, uniprot_acc, digestion_enzyme, n_fractions_in_run, fractionation_ph, enrichment, ...)` | Landed |
| Protein-level abundance (the `abundance` term) | `load_bulk_proteomics()` — CCLE + Bekker-Jensen, 222,285 rows | Landed |
| Bulk peptide -> protein coordinates | `bulk_proteomics.parquet`: `uniprot_acc` 100%, `start_position`/`end_position` 100% | Landed. Junction context is derivable with zero mapping ambiguity |
| MHC peptide flanks | `peptide_mappings.parquet` (2.77M rows): `n_flank`/`c_flank`/`position`/`transcript_id` all 100% populated | Landed (#141 closed) |
| Flank-aware unified export | `generate_training_table(map_source_proteins=True)` — docstring: "Suitable for flank-aware model pipelines such as Presto" | Landed |
| Graded detectability signal | `n_fractions_in_run` and `n_replicates_detected` (1/2/3), both 100% populated | Landed |
| Harmonized acquisition covariates | `instrument, instrument_type, fragmentation, acquisition_mode, labeling, search_engine, fdr` — same names in both indexes, by design | Landed |

### 3.2 Requested — one new function

**R1. Detectability training-set builder.** `digest()` operates on a single protein
sequence. What is missing is the proteome-scale join that turns it into a labeled
dataset with the three validity conditions applied:

```python
hitlist.bulk_proteomics.build_detectability_training_set(
    cell_line="HeLa",
    digestion_enzyme="Trypsin/P (cleaves K/R except before P)",
    n_fractions_in_run=46,        # depth arm; absence is only informative when deep
    max_missed=2, length=(7, 30), # MUST match the arm's MaxQuant search space
    require_protein_observed=True,
) -> pd.DataFrame
```

One row per in-silico candidate:

```
peptide, uniprot_acc, gene_symbol, start_position, end_position,
n_flank, c_flank,                       # junction context, from protein sequence
observed: bool,                          # the label
n_replicates_detected, first_seen_at_n_fractions,   # graded label
protein_observed: bool, protein_abundance_percentile,
digestion_enzyme, n_missed_cleavages, cell_line_name,
<acquisition columns>
```

The three conditions that make absence informative, all derivable from columns
hitlist already owns:

1. **Search-space membership.** A peptide outside the arm's MaxQuant enzyme spec /
   missed-cleavage allowance / length window was not "undetected", it was
   *unsearchable*. Must be excluded, not labelled negative.
2. **Parent protein observed.** Conditioning on the protein appearing in the
   protein-level table removes the abundance confound, which would otherwise
   dominate the label.
3. **Depth.** Absence only carries signal in a deep arm (dominant Bekker-Jensen arm
   is 46 fractions).

Filed as [hitlist#361](https://github.com/pirl-unc/hitlist/issues/361).

**Why hitlist and not Presto:** conditions (1) and (3) are source metadata that
hitlist curates, and (2) needs the protein-level index. Re-deriving them downstream
is exactly the "drift into subtle buffer/variant bugs" the `digest()` docstring warns
about. Estimated ~100 lines. Presto can prototype it locally to unblock, then upstream.

Scale check (HeLa, tryptic): 384,418 observed peptides over 15,981 proteins;
candidate set is order 10^6, so roughly 0.6-0.7M negatives for this arm alone.

### 3.3 Requested — nice to have, not blocking

**R2. Peptide-level intensity (hitlist #95, already filed).** `log2_intensity`,
`abundance_log2_normalized`, `abundance_percentile` are **0% populated** at peptide
level. With them, `detect` gets a continuous target; without them the
fraction-depth ladder plus `n_replicates_detected` gives an ordinal target, which is
sufficient for phase 1. Worth upvoting, not worth waiting for.

**R3. Per-observation sample anchor + APM columns** — filed as [hitlist#362](https://github.com/pirl-unc/hitlist/issues/362).
For the knockout axis (section 5). `mhc_allele_provenance == "sample_allele_match"`
covers **1,213,519 observations** that are *already* matched to a curated `ms_sample`
(`hitlist/curation.py:1417`, `scanner.py:574`) — hitlist uses the match to resolve the
allele and then discards the sample identity. The ask is to carry the matched
`sample_label` through onto `observations.parquet` and denormalize the
`apm_*_perturbed` / `condition_category` block.

Caveat that makes this more than plumbing: the existing resolver matches on
**allele**, and a KO arm shares its HLA genotype with its WT control by design. So
`sample_allele_match` is structurally blind on exactly the axis of interest.
Arm disambiguation needs the IEDB-side fields (`antigen_processing_comments`,
`cell_name`, assay IRI, submission grouping). Bounded scope: **29 PMIDs**.

Note: this gap is not covered by #140 (line-expression anchors) or #141 (closed).

## 4. Phase 1 — what to build now (needs nothing from hitlist except R1)

Everything required is built and populated today. Ordered by dependency:

1. **Corpus plumbing.** A `bulk_ms` corpus alongside the MHC corpus: peptide,
   junction context from `start/end + protein sequence`, machinery id, acquisition
   covariates, protein-abundance offset, detectability label. No MHC fields.
2. **`M` conditioning** on the excision/processing latent only, enforced in the
   dependency mask, with a test asserting `M` is unreachable from
   binding/presentation/recognition.
3. **`n_terminus_score` / `c_terminus_score` two-site factorization** of the processing output, replacing the
   single scalar logit; enzyme profiles pinned, proteasome learned as a mixture.
4. **Detectability loss** on `ms_detectability_logit`, ordinal over the depth ladder,
   with an IP-vs-shotgun branch offset (shared physics, different sample prep — do
   not assume zero shift).
5. **Batch construction.** The identification argument in S2.1 is a statement about
   *comparisons*, so the sampler has to guarantee those comparisons exist inside a
   batch — observed vs in-silico-negative from the same protein and enzyme arm, and
   the same protein across enzyme arms. Random sampling confounds them with per-protein
   abundance. This generalizes the existing same-peptide/different-allele binding
   pairing (`_build_binding_family_index`, `data/loaders.py:2695`) into a declarative
   contrast-group mechanism. See `tasks/todo.md` Stage 3c.
6. **Positive controls, run before anything else is believed:**
   - conditioned on `M=trypsin`, the C-terminal head recovers K/R;
   - conditioned on `M=chymotrypsin`, it recovers F/W/Y/L/M;
   - `M` ablation changes shotgun predictions and provably does not change any MHC
     prediction.
7. **The actual question**, as a factorial rather than an uninterpretable
   with/without — adding the bulk corpus adds data, a loss, and junction supervision
   simultaneously. Arms A (control) / B (`detect` only) / C (excision only) /
   D (both) / E (shuffled-label negative control), 3 seeds, primary metric held-out
   elution AUPRC on the MHC branch. Plus a mechanism check that is separate from the
   outcome check: does `ms_detectability_logit` actually correlate with held-out
   shotgun detection (expected ~0 in arm A, high in B/D)? Full table in
   `tasks/todo.md` Stage 4.

## 5. Phase 2 — the knockout axis (blocked on R3)

The KO panels are the only *interventional* data in the corpus and the natural test
of whether the processing latent is mechanistic: an ERAP1-KO peptidome should shift
N-terminal preferences in a predictable direction. This supervises `n_terminus_score`, the half
the protease panel cannot reach. Available once R3 lands: 242 APM-flagged samples,
125 with a per-sample perturbation, 29 studies / 1.34M observations / 638K peptides.

Use `condition_category` (per-sample truth), **not** the `apm_<gene>_perturbed`
flags: those are ORed with the parent study's perturbation list, so a WT control in a
KO study inherits `apm_erap1_perturbed=True`. Featurizing the wrong one inverts the
control arm.

## 6. Honest caveats

- **Denatured != processive.** In-vitro digestion acts on denatured protein; the
  proteasome degrades folded, ubiquitinated substrates in a confined chamber with its
  own length preferences. Transfer is at the level of residue-context cleavage
  chemistry only. Pin the enzyme profiles; keep the proteasome mixture learned.
- **Sample prep differs.** Immunopeptidomics has no in-vitro digest step, uses low
  input, and runs different gradients. Model `detect` as shared function + branch
  offset, not as identical.
- **C-terminal support.** Observed bulk C-termini: K 1.06M, R 773K, E 84.6K, L 49.7K,
  F 25.0K, Y 15.4K, M 9.4K, W 6.3K. The ~190K non-K/R peptides from the chymotrypsin
  and GluC arms are what make extrapolation into MHC-relevant chemistry defensible.
  A trypsin-only corpus would not have supported this.
- **The 8-15mer overlap is a bonus, not the mechanism.** 1,161,900 bulk peptide rows
  fall in 8-15 (median 13). Convenient, but the value is the identification design.
