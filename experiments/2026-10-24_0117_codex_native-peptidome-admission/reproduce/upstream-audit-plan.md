# Upstream reconciliation before training admission

Hitlist PR790 merged during the audit, at
`f4d3a6d0f1933162f9ac6eac42ac1b134c8a27dc`. The user checkout is unchanged;
an isolated detached checkout at `/private/tmp/hitlist-native-audit` pins this
implementation. Source admission will use its public package writer/EvidenceStore.
This supersedes the assumption that an importer still needs implementing.

Before execution:
- Independently decode all six retrieved OOXML workbooks using zip/XML, preserving
  every original cell coordinate, raw value/type/style and formula/cache; check
  decoded values against openpyxl. Preserve empty/formatted cells and sheet layout.
- For the four canonical assets, require exact equality to Hitlist's pinned hashes.
  Build and verify the canonical package without changing the shared environment.
- Compare every exported source row and ratio cell against independently decoded
  original values. Count source rows, ratio slots, identified versus unknown forms,
  explicit bounds/overlap/error and metadata assertions; these are not independent
  biological observations or admitted training-example counts.
- Inspect candidate cross-paper repeated measurements using source-independent
  keys, retaining both row locators. Matching values are evidence of possible reuse,
  not proof of identity; source-unique labels must not imply specimen independence.
- Check tentative sequence syntax, modifications, exposure/washout assertions and
  instrument provenance against source legends/Methods. Do not infer machinery
  activity or repair values locally.
- Preserve any upstream semantic errors as findings with reproducible examples;
  report them on the existing issue or a concrete nonduplicate bug. Source-byte
  replay success is separate from training eligibility.

All operations are CPU-only and have no model, training, pretraining, biological
simulation, predictive split or loss. Existing per-asset 120-second/2-GiB parsing
limits apply. Canonical package build/verification gets 180 seconds/2 GiB before
execution. Freeze code/invocation receipts; preserve failed attempts. Original
epoxomicin archive retrieval failed HTTP500 and the publisher page failed HTTP403;
those failures remain explicit rather than delaying the available four-asset
shared-package audit indefinitely. No author outreach or access circumvention.
