# Public metadata decoding recovery

Registered after attempts 3–5 and before additional directory retrievals. The
SBEAMS accession viewer returned complete HTTP200 HTML for both accessions;
the initial fetcher then failed because it assumed UTF-8. Preserve those failed
receipts and original bytes. A reversible Latin-1 view of these HTML bytes
exposes ordinary public experiment-directory links. This is a metadata parser
failure, not access denial. The exact original character encoding is not asserted.

Freeze a version-two fetcher and launcher rather than alter executed source.
Permit explicitly registered Latin-1 decoding only for these legacy HTML pages;
require byte round-trip equality. ASCII links and source identifiers retain their
exact characters. Use exception type/reason without embedding the whole response
in new error receipts. Keep original byte hashes, TLS verification, time/memory/
disk/byte limits and no-credential policy. No login, FTP credentials or restricted
Panorama API requests are used. Follow only the observed experiment-directory links.

Also retrieve explicitly linked small PMC supplements through their observed
individual download URLs because the Europe PMC ZIP transfer timed out. The
incomplete ZIP remains an unsuccessful partial artifact and will not be parsed.
This amendment does not admit model inputs or change scientific eligibility.
