"""Uncapped public Hitlist MS export; no perturbation/proposal whitelist."""

import hashlib
import json
import time
from pathlib import Path

from hitlist.export import generate_ms_samples_table, generate_training_table

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/shared-presentation-20261117"
OUT.mkdir(parents=True, exist_ok=True)
COLUMNS = [
    "peptide",
    "pmid",
    "mhc_class",
    "mhc_species",
    "source",
    "sample_label",
    "sample_attribution",
    "mhc_restriction",
    "mhc_allele_set",
    "mhc_allele_provenance",
    "restriction_evidence",
    "cell_line_name",
    "cell_type",
    "is_ms_observation",
    "assay_modality",
    "ms_outcome_status",
    "has_ptm",
    "peptide_modifications",
    "is_non_peptide_ligand",
    "is_potential_contaminant",
    "condition_id",
    "condition_status",
    "condition_evidence",
    "condition_reference",
    "condition_control",
    "condition_control_for",
    "condition_background",
    "condition_combination",
    "condition_knockout_genes",
    "condition_knockdown_genes",
    "condition_overexpression_genes",
    "condition_genetic_variants",
    "condition_transfection",
    "condition_transduction",
    "condition_cytokines",
    "condition_drugs",
    "condition_stimulation",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    started = time.monotonic()
    report = {
        "source_commit": "43e226621434a483141764ed54faba725bbb3869",
        "policy": "public human MS export; no study/peptide/condition/row cap; no labels from sealed TEST",
        "files": {},
    }
    for cls in ("II", "I"):
        path = OUT / f"hitlist-{cls}.parquet"
        print("exporting", cls, flush=True)
        frame = generate_training_table(
            include_evidence="ms",
            mhc_class=cls,
            species="Homo sapiens",
            columns=COLUMNS,
        )
        frame.to_parquet(path, index=False)
        counts = (
            frame.groupby(["pmid", "mhc_class"], dropna=False)
            .size()
            .reset_index(name="rows")
        )
        counts.to_csv(OUT / f"source-census-{cls}.csv", index=False)
        report["files"][path.name] = {
            "rows": len(frame),
            "sha256": digest(path),
            "columns": list(frame),
            "studies": int(frame.pmid.nunique()),
        }
        print(
            "saved",
            cls,
            report["files"][path.name]["rows"],
            "seconds",
            time.monotonic() - started,
            flush=True,
        )
        del frame
        (OUT / "export-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    samples = generate_ms_samples_table()
    samples.to_csv(OUT / "samples.csv", index=False)
    report["files"]["samples.csv"] = {
        "rows": len(samples),
        "sha256": digest(OUT / "samples.csv"),
    }
    report["cache"] = {
        "path": "/Users/iskander/.hitlist/observations.parquet",
        "sha256": digest(Path("/Users/iskander/.hitlist/observations.parquet")),
    }
    report["seconds"] = time.monotonic() - started
    (OUT / "export-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("complete", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
