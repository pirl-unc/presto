# Broader MRM known-amount/absolute-response source admission

Prepared 2026-10-25 after family1423 verified conditional dose response and found
no benefit from learning extra slopes on nine selected standards. This is the
next source-admission step toward the existing availability objective. It is not
another nominal-dose fit, a model-promotion criterion, or a narrowed goal.

## Hypothesis and fixed candidate scope

The previously identified broader MRM deposits may retain separate heavy/light
peak areas with source-backed standard amounts and enough molecular diversity
to constrain sequence-dependent response under a specified protocol. Ratio-only
curves cannot identify a shared isotope-pair response coefficient. Test actual
deposited support before allocating another sequence-response model.

Primary candidate: Michaud et al.2018, DOI10.1038/s42003-018-0087-6,
PMC6123701, PASS01163 (validation experiments1–2) and PASS01200 (3–5).
Public CPTAC records linked from this publication can identify the specific
Panorama folder missed by the earlier broad screen. The observed public record
non-CPTAC-3588 is the initial route; it is not a performance-selected assay.

Fallback/comparator only if the first candidate lacks essential processed data:
Sobsey et al.2025, DOI10.1021/acs.jproteome.4c00576, PMC11812018,
PASS05890 and its publication-linked CPTAC assay IDs. Do not expand to unrelated
papers, donors, raw-MS reprocessing or private data in this family.

## Stage A: bounded public access and schema discovery

Register one timestamped family and freeze the fetch/audit code and exact
invocation before retrieval or numerical parsing. Start with known publication,
CPTAC and repository metadata URLs. Follow only recorded public links or documented
read-only APIs for these specific deposits. Preserve discovery provenance for
every additional URL: referring page/field and exact request parameters.

Retain response URL, redirect chain, HTTP status/type, timestamps, bytes and SHA256
for every fetched asset. Preserve failed attempts, HTML challenges and access
limits without calling them successful data or proof of global absence. Do not
use credentials, request access, contact authors or circumvent access controls.
Use an existing official alternate public API when a web parser is unsupported.

Metadata phase: at most20 initial HTML/XML/JSON/README responses,5MiB each,10s
connect and30s read timeouts,120s wall time per request. Write immutable originals
once under data/mrm_anchor_sources/FAMILY. Extract links/schema/field names only;
do not query response values or fit anything until the actual processed endpoint
and a bounded second-stage invocation are frozen in an append-only amendment.
Do not fetch proprietary raw, mzML, mzXML, mz5 or vendor-data directory contents.

Processed phase: download only publication supplements, explicit processed
tables, or a source-backed Skyline document/export when needed to expose values
and their definitions. Inspect declared sizes and inventory first. Limit a single
processed asset to256MiB and cumulative family retrieval to512MiB; no silent
budget increase. A larger resource requires an explicitly recorded new decision,
not a partially downloaded scientific success. Extract only necessary metadata
from archives, preserving original bytes and rejecting unsafe member paths.

## Stage B: semantic admission before implementation or fitting

Inventory actual source grain: sequence/form/charge/transition × acquisition ×
standard amount. Preserve both isotope channels separately, aggregate/summed
versus individual-transition definitions, normalizer type and raw/normalized
response. Do not infer absolute areas from ratios or assay-optimized transition
selection from a generic total-ion model. Preserve zero, absent, missing, bounded,
unintegrated, rejected and saturated states as distinct source assertions.

Recover intended queried panel and planned observations, preparation/lot and
physical replicate identity, assay selection, injection amount versus stock or
solution concentration, units and conversions. Amount assignment, purity and
uncertainty remain source-qualified; a stated purity bound is not a certified
exact molarity. Require explicit form/run/dose links, not arithmetic invented
from response magnitude or sequential column numbering.

Capture source species, plasma/cell/matrix role, digest/enzyme and spiking stage,
instrument/acquisition/LC, standard chemistry, sample preparation and validation
experiment. Synthetic post-digestion standards are not intracellular degradation
products. Protein inference from a tryptic surrogate is not native free-peptide
concentration. Selected targeted MRM support does not establish unqueried-peptide
absence or transfer to DDA/DIA/MHC elution.

Freeze a field-to-source semantic mapping and every eligibility/disposition rule
before any supervised fit. Reusable curation belongs in the existing Hitlist
evidence/catalog/query/material-context/calibration-applicability system. The
Presto family may audit original files for admission but must not train through
a private raw-data adapter. If a new shared capability is needed, extend existing
Hitlist#762 with exact assets, fields, verified counts and acceptance criteria.

Independent checks must reconcile the source inventory and important joins/values
using a second parser or independently written arithmetic where relevant. Preserve
formula versus cached values, source coordinates and round-trip literal equality.
Do not select peptides, protocol parameters or exclusions using a desired fit.

## Decision, scope and reproducibility

Outcomes: (1) candidate has source-backed separate absolute areas and amount joins,
with a qualified scoped recipe ready for shared implementation; (2) only ratios or
ambiguous amounts/identities exist in the retrieved scope, producing a precise
negative admission and upstream request; or (3) a specific access/capacity limit
remains, with the exact route and actual downloaded state recorded. A missing
HTTP/browser route is not evidence that the paper withheld the dataset.

This family performs no pretraining, model optimization, loss computation,
predictive validation/test split or exact/RL comparison. No prediction dumps or
binding-threshold metrics apply. All original native outer partitions remain
closed. Any subsequent model needs a separately registered contract specifying
peptide/protocol transfer, controls, fixed representation, response identifiability
and held-out support, followed by the native/full-candidate availability comparison.

Local CPU only, one BLAS thread. Parsing/audit stages keep20min,3GiB process-tree
RSS,1GiB disk reserve plus32MiB startup margin,50ms polling and OS child peak
checks; startup uses3GiB macOS free-plus-speculative, not inactive RAM. No Modal,
shared environment changes, process killing or weakened guards. Network stages
also enforce bounded streamed bytes and time. Store datasets once and reference
them; compact prediction artifacts in later models should join metadata by IDs.

Close the family with access receipts, schema/admission tables, metadata contract,
independent verification or explicit unfinished scope, README and canonical
experiment_log.md entry. Preserve immutable executed code/invocations and failed
attempts. Passing source admission does not achieve the full APC concentration goal.
