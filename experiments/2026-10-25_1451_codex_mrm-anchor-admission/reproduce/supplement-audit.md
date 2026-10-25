# Prospective supplement inspection

Inspect the nine successful S3 supplement downloads registered in processed-3.json.
Before opening, verify each byte length, SHA256 receipt and official MD5 from both
the inventory ETag and article media metadata. Reject unexpected archive members,
unsafe paths, symlinks and expansion beyond the registered limits. Read in place;
do not extract or duplicate datasets.

Record workbook/sheet dimensions, nonempty cells, first six rows with coordinates,
all string cells mentioning area/intensity/ratio/response/amount/concentration/
purity/fmol, and DOCX supplementary descriptions. Independently parse XLSX ZIP/XML
shared strings, inline strings, booleans and numeric cached values and compare
every represented value to openpyxl's cached-value reader. Preserve formulas in
original files; count formula cells and explicitly flag unsupported encodings or
date conversions instead of claiming those independently verified. Do not
recalculate formulas, invent missing values, infer concentrations or fit models.

The semantic decision must distinguish validated amounts, ratio/CV/detection-limit
summaries, endogenous protein estimates and separate raw isotope areas. A missing
area label alone does not prove absence; inspect sheet layout and descriptions.
This audit may establish the supplements' scope but cannot establish the content
of uninspected Skyline documents or vendor raw files in other repositories.
