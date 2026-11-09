# Export wrapper return-type correction

The first export invocation completed EvidenceStore.materialize and canonical
verify_recipe_package, then failed when its wrapper attempted to JSON-serialize
the returned Path as a manifest dictionary. This is a consumer API mistake, not a
source/data or model failure. Preserve the original export.py, frozen snapshot and
failure log. Do not overwrite the materialized recipe or change the cohort policy.

Resume from the existing recipe using a separate export_v2.py: compare its recorded
request and source binding with the frozen request, rerun canonical verification,
load the manifest JSON from the returned directory, then perform the not-yet-run
canonical relation queries. Record the continuation code/hash/command separately.
No model/held-out outcome has been evaluated. Source/package/recipe content and
selection/normalization assumptions remain unchanged.
