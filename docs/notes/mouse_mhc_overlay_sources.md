# Mouse MHC Overlay Source Provenance

Purpose: provide a traceable path for mouse MHC sequence augmentation when
dedicated allele-sequence exports are incomplete for H2 alleles.

## Source stack

- Mouse MHC gene/allele nomenclature source:
  - `https://www.imgt.org/IMGTrepertoireMH/LocusGenes/nomenclatures/mouse/MHC/Mu_MHCnom.html`
- Protein sequence source:
  - UniProt REST (`organism_id:10090`, per-gene queries)
  - base API: `https://rest.uniprot.org/uniprotkb/search`
  - record URL template: `https://www.uniprot.org/uniprotkb/<ACCESSION>`

## Builder command

```bash
python -m presto data mhc-index mouse-overlay --datadir ./data
```

This writes:
- `data/ipd_mhc/mouse_uniprot_overlay.csv`
- `data/ipd_mhc/mouse_uniprot_overlay.fasta`

Then rebuild index:

```bash
python -m presto data mhc-index refresh --datadir ./data
```

## Per-protein provenance fields

The overlay catalog contains explicit provenance columns for backtracking:

- `imgt_gene_symbol`
- `imgt_source_url`
- `uniprot_gene_query`
- `uniprot_accession`
- `uniprot_entry_id`
- `uniprot_entry_type`
- `uniprot_record_url`
- `uniprot_protein_name`
- `uniprot_gene_names`
- `allele_derivation_rule`
- `selected`
- `selection_reason`
- `build_timestamp_utc`

## Selection semantics

- Overlay FASTA contains one selected sequence per derived allele token.
- If multiple proteins map to the same derived allele:
  - prefer reviewed entries,
  - then longer sequence length,
  - then lexical accession tie-break.
