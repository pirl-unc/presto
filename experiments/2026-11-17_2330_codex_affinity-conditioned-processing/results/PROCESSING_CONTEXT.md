# APC annotations in conditional TRAIN pairs

One row per matched peptide pair. Counts reflect admitted annotations after the inherited model vocabulary projection, not isolated intervention/control comparisons; concurrent factors overlap.

| Class | Annotation | State | Matched pairs | Conditions | Studies |
|---|---|---|---:|---:|---|
| I | cytokines: IFNG | present | 55 | 2 | [PMID 32938616](https://pubmed.ncbi.nlm.nih.gov/32938616/) |
| I | interventions: drug:flucloxacillin | present | 1 | 1 | [PMID 33633747](https://pubmed.ncbi.nlm.nih.gov/33633747/) |
| I | interventions: knockdown:ERAP1 | absent | 22 | 1 | [PMID 32161166](https://pubmed.ncbi.nlm.nih.gov/32161166/) |
| I | interventions: knockdown:ERAP1 | present | 22 | 1 | [PMID 32161166](https://pubmed.ncbi.nlm.nih.gov/32161166/) |
| I | interventions: knockout:ERAP2 | present | 43 | 1 | [PMID 32265295](https://pubmed.ncbi.nlm.nih.gov/32265295/) |
| I | interventions: knockout:HLA-A | present | 44 | 2 | [PMID 32161166](https://pubmed.ncbi.nlm.nih.gov/32161166/) |
| I | interventions: knockout:HLA-B | present | 44 | 2 | [PMID 32161166](https://pubmed.ncbi.nlm.nih.gov/32161166/) |
| I | interventions: knockout:HLA-C | present | 44 | 2 | [PMID 32161166](https://pubmed.ncbi.nlm.nih.gov/32161166/) |
| I | interventions: transduction:CIITA | absent | 15 | 3 | [PMID 33592498](https://pubmed.ncbi.nlm.nih.gov/33592498/) |
| I | interventions: transduction:CIITA | present | 30 | 3 | [PMID 33592498](https://pubmed.ncbi.nlm.nih.gov/33592498/) |
| I | interventions: transduction:HIV Env | present | 27 | 1 | [PMID 29437277](https://pubmed.ncbi.nlm.nih.gov/29437277/) |
| I | interventions: transduction:HLA-B | present | 44 | 2 | [PMID 32161166](https://pubmed.ncbi.nlm.nih.gov/32161166/) |
| II | cytokines: IFNG | present | 6 | 4 | [PMID 35561310](https://pubmed.ncbi.nlm.nih.gov/35561310/) |
| II | interventions: drug:DMSO | present | 4 | 1 | [PMID 35561310](https://pubmed.ncbi.nlm.nih.gov/35561310/) |
| II | interventions: drug:decitabine | present | 5 | 3 | [PMID 35561310](https://pubmed.ncbi.nlm.nih.gov/35561310/) |
| II | interventions: drug:tazemetostat | present | 4 | 2 | [PMID 35561310](https://pubmed.ncbi.nlm.nih.gov/35561310/) |
| II | interventions: knockdown:TAX1BP1 | absent | 43 | 2 | [PMID 36215666](https://pubmed.ncbi.nlm.nih.gov/36215666/) |
| II | interventions: knockdown:TAX1BP1 | present | 28 | 1 | [PMID 36215666](https://pubmed.ncbi.nlm.nih.gov/36215666/) |
| II | interventions: transduction:CIITA | present | 469 | 6 | [PMID 33592498](https://pubmed.ncbi.nlm.nih.gov/33592498/), [PMID 36215666](https://pubmed.ncbi.nlm.nih.gov/36215666/) |
| II | interventions: transfection:HLA-DM | present | 3370 | 42 | [PMID 37301199](https://pubmed.ncbi.nlm.nih.gov/37301199/) |

All counts are after projection to the inherited checkpoint vocabulary. Missing annotations are not controls. Gene/drug counts can overlap within combinations. The paired contrast is detected peptide versus source background under the same condition, not treatment versus control. No quantitative component-expression annotations are present in this conditional source.

Source and projection receipt: `processing-context-census.json`. Broader ordinary training retains its previous APC support and all eligible examples.

Counts describe source annotations. Class-specific pathway masks still apply: a CIITA annotation on a class-I row does not imply that the class-I processing route uses that factor.
