# Resolve native PSMD2 reference and quantity grain before canonical admission

This extends source audit 0344 and Hitlist #837. It is a source qualification,
not another fit on inspected data. The next biological endpoint remains the
same-parent relative-product contrast specified in
`native-psmd2-canonical-and-contrast.md`; the full goal remains precursor-relative
peptide availability in APCs. HEK293T evidence alone cannot meet that goal.

## Prospective scope

1. Register a new source family before retrieval. Reuse the verified 0344
   workbook, deposited inventory, acquisition design, search configuration and
   primary article by path/hash, without recopying them.
2. Retrieve the deposited `HUMAN_NS3_Cyclin.fasta` (33,003,906 bytes), requiring
   deposited SHA1 `ebfe21f8f804ceb7679d4c41c001fde25cea2a96`. Preserve all original
   bytes. Classify headers and references; restrict future biological predictors
   to endogenous human records. Reporter/nonhuman and decoy records remain
   source dispositions and do not become prediction or optimization targets.
3. Inspect the original central directory of `Hi20S_txt.zip` using HTTP byte
   ranges, requiring exact Content-Range/length for the known 167,389,796-byte
   object. Do not accept an ignored range as a full download. Record all members'
   names, methods, CRCs, offsets and sizes. Reject unsafe/encrypted/unsupported
   members. Retrieve only `peptides.txt`, `summary.txt`, and optionally small
   parameter tables needed for interpretation. No RAW/Andromeda downloads.
   Only inspect an evidence table if the admitted members cannot resolve a
   specific remaining quantity question and its size fits the unchanged cap.
4. Verify each retrieved member against its local and central headers, compressed
   and uncompressed lengths and CRC. Independently reopen a reconstructed small
   ZIP containing retained compressed members to check stdlib extraction equality.
   Partial archive verification does not verify the complete archive SHA1.
5. Reconcile every workbook peptide with original peptide-table rows and all
   corresponding quantity/identification fields, preserving missing, extra and
   conflicting cells. Respect literal Excel values and date-shaped gene defects.
   Examine reference accessions and all parent alternatives without choosing
   leading/razor proteins. Report mapping/support only, not condition means,
   effects or model scores; do not open a biological held-out partition.
6. Use this evidence to specify the smallest common Hitlist source adapter on
   authoritative current main, with shared metadata/material contracts. Canonical
   admission, release and a separately registered fit follow qualification.

## Bounds and failure handling

Local only, no dependency installs or shared-environment edits. Maximum retrieved
asset/member 64 MiB, family raw data 256 MiB, 30 seconds per network request,
1,200 seconds aggregate acquisition and 20 minutes per local stage. Preserve a
continuous 1 GiB disk reserve plus 32 MiB startup margin; 3 GiB RSS cap. Never
relax these bounds after observing failures. If a sequential ranged recovery is
needed, register its exact offsets against the same object first. No blind retries
or overwriting receipts. All full/partial data, commands, failures and independent
verification belong in the canonical family. Source qualification has no training,
validation or test predictions and cannot establish predictive quality.

## Closure

Record original and retained-source checksums, source/reproduction code hashes,
runtime, all discrepancies, reference/support counts and exact next admission
decision. Update the canonical registry and #837 with verified findings once
published. A successful source audit does not close #837 or the biological goal.
