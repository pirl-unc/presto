# Official download-service recovery; cohort unchanged

The original acquisition process terminated after two successful archives and an
Apache HTTP403 for TUM_HLA_63_01_01_DDA-1h-R1-unspecific.zip. The deposited directory
lists it as a regular19Mfile. Original receipt/log/source remain unchanged.

PRIDE's own download documentation offers the public archive-file-downloader service
as another official mechanism for released datasets:
https://github.com/orgs/PRIDE-Archive/discussions/33
https://www.ebi.ac.uk/pride/ws/archive-file-downloader/files/s3/{accession}/{filename}

Resume by independently verifying already-downloaded hashes; do not download those
again. For the failed public archive use that documented service with the exact same
accession/filename, no credentials or access-control manipulation. For remaining
selected archives retain the original directory route, using the documented service
only after a recorded failure. Preserve every attempt/status/URL/header/hash and
byte count; original1GiBtransfer/128MiBasset/2GiBdisk-floor caps include prior downloads.
No pool, split, model, source-quality gate or measured target changes. If the official
alternative cannot serve a selected archive, stop with evidence rather than silently
replace the pool or bypass access controls.

The documented streaming alternative returned HTTP500 (not an absence finding).
The deposited README itself supplies an anonymous FTP URL for the identical file,
also documented by PRIDE. One bounded anonymous FTP retrieval is allowed next,
with no private credentials. Its result is recorded separately; no source identity,
cohort assignment or resource limit changes.

Anonymous FTP login and SIZE succeeded (19,755,957 bytes); RETR received327,680bytes
then hit the30second read timeout. FTP context cleanup subsequently raised426;
the original traceback preserves both. This establishes a transfer failure, not
private-data denial or absent source. One clean retry of this identical file is
allowed with a120second data timeout and180second overall deadline. Retain the
original partial bytes and compare their prefix against the new complete transfer;
record server SIZE/MDTM before/after and ZIP CRC if successful. No repeated retry
loop or replacement pool is authorized by this amendment.

The single FTP retry completed in14.94seconds:19,755,957bytes, unchanged serverMDTM,
matching327,680byteprefix and all ZIP CRCs. SHA2560b81754eca2caf4d5748534527a10a266ee69293a4a5628aafde0218bd7156f6.
Continue the same remaining selected files, verifying all prior successful hashes.
Use original HTTPS, documented service, then one bounded anonymous FTP attempt if
needed; retain every failed route. No repeated-success transfer or changed pool.
