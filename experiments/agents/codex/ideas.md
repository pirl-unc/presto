# Codex Ideas

Use this file for rough experimental hypotheses and brainstorms.

- Keep items short.
- Promote actionable items into `todo.md`.
- Promote non-trivial items into `plans/`.

## 2026-10-22 — quantitative availability source candidates

Goal-directed source reconnaissance, not source admission or a biological fit:

- [Wolf-Levy et al., Nature Biotechnology 2018](https://www.nature.com/articles/nbt.4279)
  describes MAPP capture of proteasome-associated endogenous fragments alongside
  standard proteomics. Supplementary Data 2 and 8 are peptide tables; PRIDE
  accessions include PXD009092, PXD009360, PXD007263, PXD009181, PXD009438,
  PXD009112 and PXD009508. Candidate for relative product-pool supervision;
  captured local fragments and measured intensities are not necessarily free
  cellular concentration. Audit the exact tables, paired sample identities,
  original values, precursor quantities and observation scope before fitting.
- [Stopfer et al., Nature Communications 2020](https://www.nature.com/articles/s41467-020-16588-9)
  / PXD017407 reports calibrated copies-per-cell for selected pMHCs and broader
  relative immunopeptidomic data. Useful possible downstream quantity check;
  the measured pool is MHC-bound, not all intracellular products. A small number
  of calibrated targets cannot by itself anchor an entire concentration model.

Neither source has been downloaded or schema-qualified in this program yet.
The earlier generic optimizer study does not establish any biological endpoint.


## October24: transfer a digestion-trained cleavage prior to cellular MAPP

The nested sequence-context comparison has a stronger digestion development gain
from wider cleavage context than from fragment-length dependence. Before adding
more mechanistic state, test whether frozen digestion-trained rates transfer to
the existing cellular captured-composition endpoint that defeated direct fitting.
This is a candidate next experiment, not a launched or qualified result.

- Reuse canonical MAPP min2 manifestdf78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80; parent and example files are JSONL. No new consumer curator.
- Audit cross-source parent/construct and peptide overlap before claiming transfer; exact separation is not homology/specimen separation.
- Compare frozen adjacent versus wider-context digestion rates, plus uniform-cut and uniform-composition controls. Fit at most one global effective exposure scalar initially; do not silently call it known intracellular digestion time. Enzyme-context reference and MS capture/response remain explicit assumptions.
- Preserve all-fragment physical support; normalize only for the observed-set MAPP evaluation. This endpoint still cannot identify total yield, clearance or concentration. Do not transfer instrument-specific MS coefficients without support.
- Use an observation-aligned centered-log10 loss, retain earlier KL-based results as different-objective references, and qualify a matched sampled-gradient counterpart for the same new loss. Freeze the full specification and verification scope before any new fit.
- Treat prior MAPP held-outs as previously exposed; original digestion test remains closed. Independent cellular concentration validation remains an unmet objective.


## October24: native intracellular peptidome perturbations

[Hitlist788](https://github.com/pirl-unc/hitlist/issues/788) now owns admission of
Dasgupta2014 (10.1371/journal.pone.0103604) and related Gelman2013
(10.1371/journal.pone.0053263) TMAB-labeled cellular peptide extracts. These are
a distinct candidate measured population beyond proteasome capture/MHC elution.
Primary articles describe public supplementary ratios, censoring and reused
epoxomicin/bortezomib measurements. No workbook has been schema-qualified or fitted
in this program; do not treat literature row counts as admitted examples.
Curation must preserve drug/vehicle/exposure, cell/species, extraction/labeling,
channel/control covariance, PTMs/charge/identification, original mappings and
cross-paper measurement lineage. No drug-to-KO inference or absolute-concentration
claim. Shared canonical admission must precede any consumer fitting.


## October24: APC-derived endolysosomal component measurements

Primary source lead: Öztemiz2026, PMID42846314, DOI10.3389/falgy.2026.1936335 ([paper](https://www.frontiersin.org/journals/allergy/articles/10.3389/falgy.2026.1936335/full)). Extracted dendritic-cell endolysosomal preparation and peptide time-course signals could help separate processing from MHC capture, but are not native intact-cell concentrations. Original quantitative files and dependencies remain unqualified. Public PeptideStudio sample_data at18dd99da4a619abd8a28c7622e0809eb2c03e0ff is explicitly synthetic; do not use it as biology. [Hitlist812](https://github.com/pirl-unc/hitlist/issues/812) filed and body read back, including source identity, activity/sample context, quantity/modification/detection semantics and demo exclusion. No numeric assets admitted, no experiment or training claim. Register an original-source audit before numeric inspection/admission.

## 2026-10-27 — Stronger cross-class sharing, after the frozen v4 benchmark

User welcomes simpler representations and additional data that make both classes
train the same machinery. These are prospective hypotheses, not demonstrated wins.
Do not add them to the already frozen ten-condition comparison after seeing results.

1. **Interaction before pooling.** Current v4 still pools independently encoded
   peptide and MHC sequences before fusion. Replace that bottleneck with one shared
   residue-pair interaction/cross-attention block, conditioned on both MHC grooves.
   Use the same scorer for every candidate and endpoint, with geometry carried by
   alignment/masks instead of class-specific parameters. Compare against v4 under
   a fresh fixed budget and held-out cross-class protocol.
2. **Shared pocket coordinates.** Align the binding grooves across classes using
   verified structural correspondence; do not assume equal sequence indices or
   pretend the two groove topologies are identical. Represent terminal constraints
   and peptide-to-pocket alignments explicitly. One differentiable alignment rule
   could permit internal bulges and open-ended overhangs with different allowed
   geometry. Validate recovered contacts/registers against independent structures.
3. **Structural auxiliary supervision.** Curate pMHC contacts/core annotations for
   both classes with structure/peptide/allele-family split controls. Train the same
   interaction representation on contacts and affinity, preserving assay-specific
   readouts. Structures are generally positive complexes and are not affinity
   labels or evidence of universal negative pairs. Structural target-class exposure
   means zero-shot affinity labels, not no exposure to that class.
4. **Unlabeled sequence pretraining.** Masked-residue objectives over both groove
   chains and peptides can initialize the same encoder; document exact corpora,
   near-duplicate exclusions and whether target-class sequences were seen.
5. **Empirical bridging.** Prioritize direct comparable binding assays containing
   overlapping peptide series across classes, keeping global peptide-family splits.
   Shared peptide does not mean equal affinity. Do not force cross-class equality,
   fabricate negative labels, or call teacher predictions independent measurements.

Primary-source leads (scope-limited evidence, not proof of cross-class transfer):
- pMHChat (class II residue contact/affinity modeling): https://pubmed.ncbi.nlm.nih.gov/40619811/
- MHCnuggets (transfer among alleles; separate class baselines): https://www.cs.jhu.edu/~rohit/research/mhcnuggets_paper.pdf
- MHCSeqNet2 (class I, structural embedding/low-data allele generalization): https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btad780/54913538/btad780.pdf

Sequence-only evidence from the current benchmark: zero exact training peptides
shared across classes, only22 shared9mer substrings. Direct comparable assays of
overlapping peptide series could provide a much stronger empirical bridge. About
85% of the original II test rows have a9mer seen in II training, so a prospective
peptide-family/core-disjoint benchmark belongs in the backlog before broad claims
of new-core generalization. Cross-class overlaps are only1/958 and8/1845 test rows.

## 2026-10-27 — Ranked transfer brainstorm after the fixed-budget comparison

User asks about MHC-I terminal overhangs, species differences, nonclassical
molecules, short/fixed-register class II, homogeneous flank summaries and class
classification. These are hypotheses for prospective tests, not changes to the
completed benchmark. The present human purified-IC50 runs do not qualify MS or
processing prediction. Both transfer directions are modest; same-class models
outperform the joint model at1000updates, with unequal per-class exposure.

Biological distinctions: a central peptide bulge, terminal overhang, transient
opening during loading and complete dissociation are different phenomena. Natural
class-I overhangs exist and vary by molecule; there is no justified single species
openness ranking. One dominant class-II binding register does not make the groove
closed, and enumerating registers does not assert physical sliding. Class-II
length preferences can reflect binding plus processing/selection in MS. Chicken
BL2*02 has a structurally supported10-residue core, illustrating why9 is a strong
human default rather than a universal law.

Ranked candidates:

1. **Residue interactions before pooling.** Replace independent peptide/MHC mean
   compression with one shared residue-to-groove interaction block and structurally
   corresponding groove coordinates. Current v4 still compresses these sequences
   before interaction. Contacts learned from either class should train the same
   local compatibility function. Geometry/masks preserve genuine differences.
   Test within-class and both transfer directions at matched capacity/exposure.
2. **One alignment and boundary representation.** Use one candidate-alignment
   interface for both classes, permitting terminally anchored, bulged and overhang
   configurations with MHC-conditioned costs. Do not force every class-I ligand
   into a nine-residue core. Each residue carries distance to peptide termini,
   candidate groove coordinate and contact/overhang role; shared weighted summaries
   cover N-overhang, groove contacts, internal insertion and C-overhang, with an
   explicit empty region. Preserve ordered nearby context. Typical class-I flanks
   are empty, but the same position/contact machinery is trained on all residues;
   auxiliary structures and actual extended class-I examples supply missing
   overhang supervision. Cropping must not inherit unchanged affinity labels.
3. **Measured peptide series and rank-aware supervision.** Prioritize existing
   matched truncation/extension/substitution binding series on the same molecule
   and assay. Fit measured absolute affinity and within-assay affinity differences
   with a generic ranking term; keep assay/censor uncertainty explicit. Series
   split together. No assumed monotone length effect or fabricated binding labels.
   This directly addresses Spearman and gives boundary/core effects empirical
   anchors. Kd, stability and IC50 remain distinct measured outputs.
4. **Structural and noncanonical bridge data.** Curate verified peptide contacts,
   terminal anchoring, occupied groove positions and backbone/register annotations
   across classical I/II, extended class-I examples, avian examples and peptide-
   binding HLA-E/HLA-G. Prefer complementary geometry over raw sequence count;
   structures are positive complexes, not affinity measurements. Supervise the
   shared representation, not a parallel predictor. PDZ/SH3 peptide-binding data
   are a lower-priority broader contact-pretraining ablation. CD1/MR1 require lipid/
   metabolite chemistry and are not a first-line peptide-affinity bridge; DM/DO
   editing measurements do not supply ordinary presentation-affinity labels.
   Target-class structural/pretraining exposure must be disclosed: this tests
   zero target-class affinity labels, not total class blindness.
5. **Balance optimization and supervise anatomy.** Compare equal total updates
   and equal per-class example exposure; use balanced batches and inspect gradient
   contributions before declaring negative transfer. Preserve regression calibration
   while measuring per-MHC/within-assay rank. An existing mhc_class auxiliary can
   be ablated at low weight, but class discrimination could encourage separation
   and is weaker supervision than contact/terminal-anchoring geometry. No class-
   specific expert or independently trained register head. Freeze objective weights
   on training/validation only and use fresh family-disjoint evaluation.

Recommended execution sequence: balanced-exposure control first to resolve the
current confound; residue-interaction representation next; add shared alignment/
boundary features and structural supervision as separate measured ablations.
Paired affinity-series data are the preferred empirical acquisition in parallel.

Primary literature, scope-limited:
- Natural class-I C-terminal extensions,2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC5960288/
- Songbird alternative N-terminal anchoring,2023: https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1209059/full
- Chicken class-II decamer core,2021: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001057
- Class-II measured elongation-series analysis: https://pmc.ncbi.nlm.nih.gov/articles/PMC2640366/
- HLA-E/Mamu-E/Qa-1 binding comparisons: https://pmc.ncbi.nlm.nih.gov/articles/PMC7653511/
- Joint structural/binding training with transfer to PDZ/SH3,2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC9992841/
- Historical latent-register/shared binding-model precedent: https://pubmed.ncbi.nlm.nih.gov/18771399/

The cited work supports structural/data precedents; it does not demonstrate that
these changes improve Presto. Do not mine exceptions for pathogen-specific design.


## 2026-10-27 — Follow-ups from the shared transfer comparison

The registered `2026-10-27_2100_codex_transfer-improvements` comparison supports
a modest fixed-budget joint-affinity gain; zero-shot transfer remains weak.
These follow-ups remain untested:
- Keep sampling and architecture effects separate; neither need be additive.
- Compare validation selection by pooled RMSE with a declared balanced group
  objective prospectively, holding training and checkpoint availability fixed.
- Audit cross-class structural coordinate correspondence. Shared token attention
  does not itself establish aligned MHC groove positions or physical contacts.
- Curate explicit panel/condition membership for real peptide series before
  constructing pairwise affinity losses. The training-only inventory identifies
  candidates, not validated paired measurements; no synthetic crop labels.
- If residue interactions are retained, batch their computation across observations
  in a separate software/performance study and verify checkpoint parity. Do not
  silently change the execution contract of a running predictive benchmark.

- A further shared-core hypothesis: register pooling currently has temperature1
  in standardized model units, so its effective temperature in log10(nM) is
  the source-training target standard deviation. Source-only and joint fits
  therefore use different physical-unit pooling strengths despite identical code.
  A prospectively fixed assay-unit temperature could make this more homogeneous.
  This is a mathematical observation, not an accuracy result or physical occupancy
  claim. Test it with frozen source-only checkpoints and then matched new fits
  before changing the common training or checkpoint contract.
