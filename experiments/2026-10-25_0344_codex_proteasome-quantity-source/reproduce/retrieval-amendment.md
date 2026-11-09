

The publisher/article-media endpoints returned client challenges, not workbooks.
Dryad's public-page link returned403; its documented API download returned401.
These are access failures, not evidence of absent measurements. Preserve every
response; do not bypass authentication or automate a challenge.

The public EuropePMC bundle returned original ZIP bytes but exceeded the20MiB
per-request cap. The retained partial stream may already contain complete
workbooks. Before further download, inspect sequential ZIP local records with
bounded deflate decoding. Recover only complete XLSX members whose CRC and
compressed/uncompressed lengths match their original data descriptors. Retain
offsets/hashes/headers and independently check XLSX inner ZIP CRCs. A truncated
member is never admitted. This does not claim the incomplete outer archive's
central directory or later members were verified. Do not increase any cap or
download the full bundle again. Source and model scope remain unchanged.
