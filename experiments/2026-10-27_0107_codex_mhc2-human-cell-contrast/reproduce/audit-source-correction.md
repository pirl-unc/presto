# Correct the frozen export identity before source inspection

The first audit stopped at its file hash assertion before selecting any rows:
0.082seconds/67,092,480bytesRSS. Its code paired the corrected-v4 bundle path
with the hash of the distinct older native export used in the DP/DQ fit. The
files are different artifacts and must not be conflated. Preserve that script,
launch and failed receipt unchanged.

The planned prior1803 inventory used the corrected-v4 `observations.parquet`,
145,685,701bytes, SHA256
`a665f37c0b5917fefb6d12eade530d7e2001de1f0ee73479b21d2c393d173ab4`.
That hash is independently pinned in the published1803 execution-inputs.json
and the downloaded bundle's SHA256SUMS. Use that exact export for this audit,
with its own observations_meta.json identity. Current2.0.42 curation remains a
separate pinned source. The older native-export hashf902341e... remains correct
for the frozen2036 fitting cohort and is unchanged there.

Create audit_v2.py and a new execution receipt; change only the expected source
hash/receipt/launch names and describe outputs as corrected-export rows. Verify
the three-way prior receipt/checksum manifest/file match before execution. All
selectors, conditions, missing-data handling, counts/replay and resource limits
remain unchanged. No predictive data or model outcome was used in this correction.
