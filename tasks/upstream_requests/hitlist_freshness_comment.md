Freshness audit for the current Presto IC50/TCR work, checked 2026-10-20:

- The failed IC50 run actually consumed `human-cohort-corrected-v1`, native manifest `b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`, produced by Hitlist 1.78.0. Its manifest records current indexes at export time. This remains the newest published corrected general human training bundle found in the complete release inventory.
- Current PyPI/main are 1.82.11. The local shared editable install is 1.82.2 and its separately cached observations fail `observations_cache_is_current()`. Neither package upgrades nor this local cache describe the immutable fit's input freshness.
- Atlas donor-lineage and bulk-count releases are separate evidence contracts, not replacements for the native binding bundle. Presto #127 still gates Atlas consumption.

Please track a fresh, versioned corpus/training export incorporating current released curation, with pinned upstream raw snapshot identity/date, build and software versions, input freshness receipt and a changed-row/field audit against corrected-v1. Keep original IDs and distinguish source acquisition updates from annotation-only changes. A refreshed export must not silently replace old frozen inputs or claim predictive improvement.

Concrete missing-evidence requests are now #721 (structured binding conditions and replicates), #722 (access-aware TCRvdb adapter), and #723 (measured functional/processing evidence). Existing #680/#705/#706 remain the lineage/independence work; no duplicate issue opened for those.
