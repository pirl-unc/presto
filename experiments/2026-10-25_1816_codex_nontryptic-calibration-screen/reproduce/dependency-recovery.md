# Inspection dependency recovery, before numeric workbook access

inspect-attempt1 exited during imports: PyMuPDF/fitz is absent. It opened no
workbook or PDF and fitted/scored nothing. Preserve code/inspect.py, its source
snapshot/request/runtime/log and all downloaded source bytes unchanged. A preceding
isolated lint command also flagged the intentional timezone-free Excel serial-date
origin; the shell continued to the failed launch. Subsequent compound verification
commands use set -e. No reported test result is treated as passing that lint check.

Verified existing dependencies: openpyxl, pypdf, pdfplumber, pypdfium2 and Poppler
pdftoppm/pdftotext/pdfinfo. New code/verify.py uses pdfplumber plus pypdf for text
and existing Poppler for images. Excel dates remain timezone-free as in the source,
with an explicit lint annotation for that representation. This is a dependency/
source-format repair, not a data/threshold/measurement-policy change. Preserve the
same prospective inspection scope, source checksums and resource limits.
