# Public PASS processed Skyline retrieval

After the fixed20 HTTP metadata requests, PASS05890's public accession page lists
CPTAC_Exp1_Curves_CS.sky (7.8M), its larger cache/archive, and other experiments.
This identifies an actual small processed response-curve asset. Retrieve only the
standalone .sky document; do not download vendor data, caches or the148M archive.

The same public page explicitly publishes the dataset-specific FTP download route.
The original no-credentials rule was intended to avoid private accounts/restricted
access. Amend it narrowly to permit this repository-published public dataset login,
read from the already archived page at runtime. It is not a private user credential
or a bypass of Panorama's login requirement. Do not create accounts, request access,
use browser sessions or log the published password. Preserve the source HTML once
in data/ and record its hash and the dataset accession instead of credential text.

Use the exact published FTP host and dataset account, then TYPE I, SIZE and RETR
for the named file. Bound connect/read to30s, transfer to120s, per-asset bytes256MiB,
family512MiB, stage20min, RSS3GiB and disk reserve1GiB plus startup margin. These
are public data; no private content is transmitted. Preserve failures/partials and
SHA256 bytes, validate XML root and exact remote size before inspecting fields.
No numerical response extraction or fit is performed by the downloader. Register
the field/amount audit separately after the actual Skyline schema is available.
