# Sample data

Ready-to-load example files for trying out PeptideStudio.

> ⚠️ **Synthetic example data — not a real experiment.** The peptide table and every intensity
> value in this folder were **artificially generated for demonstration purposes only**. They do
> not come from an actual mass-spectrometry run, do not represent a real digestion/degradation
> experiment, and must not be interpreted, reused, or cited as real biological or quantitative
> data.

| File | What it is |
|---|---|
| `example.fasta` | Protein sequence used for the example: **human myoglobin** (UniProt [P02144](https://www.uniprot.org/uniprotkb/P02144), gene `MB`), 154 amino acids. |
| `Example_degradome.xlsx` | Synthetic peptide table simulating an endolysosomal-degradome-style time-course of that same protein. |

## About the example protein

`example.fasta` contains **human myoglobin** (UniProt accession `P02144`, gene `MB`), a small,
well-characterized 154-residue oxygen-binding protein, freely available from UniProt. It was
chosen only for its convenient size and availability — it has no particular biological
connection to endolysosomal degradation; it simply serves as a stand-in sequence for
demonstrating the tool's visualization.

## About the example peptide table

`Example_degradome.xlsx` simulates an in-silico "digestion" of the myoglobin
sequence: roughly 200 overlapping peptide fragments (mapped by position onto the 154-residue
sequence), each carrying **fabricated** intensity values across nine example time points
(`0h, 0.5h, 1h, 2h, 3h, 5h, 8h, 16h, 24h`). Its only purpose is to illustrate what a
degradome/proteolysis time-course looks like once rendered — **it is not the result of any real
digestion experiment**, and the numbers were not measured.

Column layout (sheet `Sheet1`):

| Column | Role |
|---|---|
| `Start position` / `End position` | Peptide position in the 154-aa myoglobin sequence (auto-detected as Start/End) |
| `Sequence` | Peptide amino-acid sequence (auto-detected as the Peptide column) |
| `Length` | Peptide length — informational only, not used by the app |
| `0h`, `0.5h`, `1h`, `2h`, `3h`, `5h`, `8h`, `16h`, `24h` | Synthetic intensity values per time point |
| `Protein Sequence` | Full protein sequence, repeated for reference (not parsed by the app) |

A second sheet (`Sheet2`) simply holds the FASTA-formatted sequence as plain text for reference.

Because the time-point columns are named `0h`, `0.5h`, … rather than containing `area` or
`intensity`, the app's column auto-detection **will not** pre-select them as intensity columns —
after loading the file, tick them manually under **Intensity columns** in the sidebar (Start/End
and the Peptide sequence column are auto-detected).

## Load it in the app

1. Upload `Example_degradome.xlsx` under **Data**.
2. Under **Protein sequence**, either upload `example.fasta`, or choose **From sheet column** →
   `Protein Sequence`.
3. Under **Column mapping**, tick the nine time-point columns listed above as **Intensity
   columns**.

## Build your own file

1. Keep a **Start** and **End** column (1-based integer peptide positions in your protein).
2. Add one numeric column per time point / sample; name it so it contains `area` or `intensity`
   (e.g. `Area_0h`) for automatic detection, or select it manually otherwise.
3. A `Peptide`/`Sequence` column is optional — if omitted, the peptide sequence is derived from
   Start/End against the loaded protein sequence.
4. Save as `.xlsx`, `.xls`, or `.csv` — all three load identically.

Then load your file together with a FASTA sequence (or use UniProt search) in the app.
