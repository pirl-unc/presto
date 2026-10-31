# Original APC-derived endolysosomal source audit

Status: source audit closed; no peptide-quantity cohort admitted. Codex / GPT-6, October26.
Plan: [prospective source audit](../agents/codex/plans/2026-10-26_endolysosomal-source-audit.md).
Existing scope: [source qualification](../agents/codex/plans/endolysosomal-source-qualification.md).

Question: does the cited Öztemiz2026 original publication provide measured peptide
time-course tables and sufficient source/context/replicate provenance? Its linked
software examples are explicitly synthetic and remain excluded. Isolated
endolysosomal preparation quantities are not native APC concentration targets.

No fit, pretraining, synthetic supervision, loss, validation/test split or
prediction dump applies to this source-only admission. No old biological held-out
is reopened. Local CPU; no GPU/Modal or dependency changes. Original assets will
remain under `data/endolysosomal_source_audit/2026-10-26_1642_codex_endolysosomal-source-audit/`.
The initial combined12-operation ceiling was explicitly extended to15 after
operation9, before the supplement API or PDFs were read: one API call and the
five listed files remained. The publisher's linked UI modules resolved the API.
The original plan and prospective amendment are both retained. Time/byte limits
were unchanged; no additional source search occurred.

## Original assets and admission decision

The [primary article](https://www.frontiersin.org/journals/allergy/articles/10.3389/falgy.2026.1936335/full)
describes isolated murine dendritic-cell endolysosomal preparations and two plant
allergens. MS signal is a relative observation, with search-length filtering and
preparation-dependent modified forms. These data would not directly establish
short-product coverage or absolute native APC concentrations.

All five publisher-listed version1 PDFs were retrieved, hashed, rendered and
viewed. Each has one page and no embedded file. The full inventory and unresolved
fields are in [evidence.json](results/evidence.json).

| Original supplement | Content | Admission |
| --- | --- | --- |
| 1 | Reporter-activity plot | Different measurement family; graph only. |
| 2 | Two protein-level half-life summaries | Eight aggregate cells; no individual peptide observations. |
| 3 | Bet v1 peptide map at nine times | Color-coded intensity, no original per-row numeric table. |
| 4 | Aes h1 peptide map at nine times | Color-coded intensity, no original per-row numeric table. |
| 5 | One peptide time-course plot | Graph only; original values and missingness remain unavailable. |

**Zero peptide-quantity rows are admitted.** Original intensity tables and
sample/acquisition/replicate lineage are still required. No graph was digitized,
color converted into a quantitative label, or aggregate expanded into replicates.
Protein-degradation half-life is not the pMHC dissociation endpoint. The earlier
synthetic-example exclusion remains; this audit did not revisit that repository.
This is a scoped source-access limit, not a claim that no original data exist.

## Verification, resources and reproduction

Starting commit7e0c8e9 plus registered plan/task notes; registration commit1db2705.
Exact state and original limits are in `reproduce/launch.json`. The ledger preserves
15 calls across tools, including the pre-registration issue read. Local HTTP bodies
total18,347,804bytes; the final download ended424.420s after registration. Individual
guarded HTTP calls total11.832s, with maximum polled RSS35,913,728bytes. Browser
network bytes/time are unmeasured; saved response size is not transferred bytes.

Poppler and pypdf agree on page/attachment counts and independently replay all
eight aggregate table cells. Poppler verifies both nine-time labels and log2
legends. Initial pypdf extraction hit its form limit on the maps, so it was not
used to assert text completeness. All rendered pages were inspected instead.
Independent replay takes2.006s,116,047,872bytes parent peak RSS and33,816,576bytes
maximum child RSS. These are post-execution peaks; Poppler calls have60s timeouts.
Initial ad hoc extraction runtime/RSS was unmeasured. No environment changes.

`bash reproduce/launch.sh` replays original hashes, publisher inventory and metadata
offline, requiring the raw assets at the ledger paths. The original article/PDF
bytes remain in the data directory, with precise public URLs and hashes retained
here. No predictive split or metrics apply because no cohort or model was admitted.

Status: closed negative admission. Next Step: continue qualified IC50 work;
this source needs original peptide tables and sample/run lineage before shared
ingestion or fitting. Open Questions: independent response scale, short products
and native APC transfer remain unresolved. No full-model qualification follows.
