# Current-main verification amendment

After the successful registered admission, concurrent Hitlist PR840 merged as
`d2e0cb61005c33c8ad484b234670174346a529ee`. It adds Goldberg MAPP context semantics
and increments the shared catalog version, while retaining unreleased package
version 2.0.37. It does not change native-peptidomics inputs or the contrast code.

The original source release build40167325306 succeeded. Its exact distributions
were downloaded and verified against clean main e755be4 and the trusted Actions
artifact; they are retained, not uploaded. Current-main release40168710669 now
tests d2e0cb6. No live build was cancelled or restarted.

The recipe PR841 was rebased onto current main as90a7a62. Its six frozen executable
overlay files remain byte-identical. The current base adds unrelated catalog/source
changes and tests, so rerun the required format/lint/default suite. Once that
finishes, invoke the final public recipe verifier against the existing qualified
package under this rebased code, with the same resource caps. Do not rematerialize
inputs, choose another split, inspect held-out effect summaries or alter criteria.

This is a post-admission compatibility check, not a new blind experiment or
independent target replay. Preserve the original frozen launch, all original
artifacts and this separately registered invocation and its receipts. The first
biological fit still requires the tested source and recipe releases.
