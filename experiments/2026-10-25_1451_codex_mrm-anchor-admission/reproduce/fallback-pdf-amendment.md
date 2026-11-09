# Fallback assay-identifier PDF

Both fallback XLSX files passed independent stored-cell verification, but provide
method settings, ranges, variability and stability summaries rather than raw
isotope signals. The archived article explicitly locates CPTAC links in TableS3
of pr4c00576_si_002.pdf. Its exact public S3 object is already inventoried at
242533bytes. Retrieve only that PDF, verify its magic/size/checksums, and extract
its page text and links to identify the public processed-data route. Inspect the
relevant rendered page when matching peptide/assay identifiers. No fitting or
MS raw-file retrieval; unchanged resource limits and anonymous HTTPS access.

Freeze a version-four fetcher adding PDF payload recognition. Earlier fetcher
versions and their executed sources remain unchanged. The only new accepted
payload type is a PDF with the standard %PDF- prefix.
