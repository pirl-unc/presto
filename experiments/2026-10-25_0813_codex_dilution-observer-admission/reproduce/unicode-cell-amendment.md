# Original table Unicode separator

The first audit stopped at article Table1 before scanning the processed table:
its amount/unit separator is U+2009 THIN SPACE, not ASCII space. Original source,
code, stage log and failing0.126second receipt remain untouched. No quantities or
models were fitted. The new audit accepts Unicode whitespace for this source cell;
the independent verifier instead extracts the explicit fmol/µL unit suffix and
strips surrounding whitespace. Neither changes the amounts or dilution assertion.
New stage names and frozen code prevent overwriting the failed attempt. All
resource limits and admission requirements remain unchanged.
