"""Hash-pinned corpus verification and public-loader presentation exports.

The corpus generation, its producer and the current loader are separate versions.
Export in a child process so selecting a corpus cannot change another reader's
Hitlist directory or overwrite the user's existing index.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .modular_sources import file_sha256

PRESENTATION_COLUMNS = (
    'peptide',
    'pmid',
    'mhc_class',
    'mhc_species',
    'source',
    'sample_label',
    'sample_attribution',
    'mhc_restriction',
    'mhc_allele_set',
    'mhc_allele_provenance',
    'restriction_evidence',
    'cell_line_name',
    'cell_type',
    'is_ms_observation',
    'assay_modality',
    'ms_outcome_status',
    'has_ptm',
    'peptide_modifications',
    'is_non_peptide_ligand',
    'is_potential_contaminant',
    'condition_id',
    'condition_status',
    'condition_evidence',
    'condition_reference',
    'condition_control',
    'condition_control_for',
    'condition_background',
    'condition_combination',
    'condition_knockout_genes',
    'condition_knockdown_genes',
    'condition_overexpression_genes',
    'condition_genetic_variants',
    'condition_transfection',
    'condition_transduction',
    'condition_cytokines',
    'condition_drugs',
    'condition_stimulation',
)


def corpus_contract(pin=None):
    path = Path(pin) if pin else Path(__file__).with_suffix(".json")
    contract = json.loads(path.read_text())
    if contract.get("schema") != "presto.hitlist-corpus.v1":
        raise ValueError("unsupported Hitlist corpus pin schema")
    return contract


def verify_corpus(directory, *, pin=None):
    contract = corpus_contract(pin)
    directory = Path(directory).resolve()
    name = contract["manifest"]
    if Path(name).name != name:
        raise ValueError("corpus manifest must be a local filename")
    manifest_path = directory / name
    digest = file_sha256(manifest_path)
    if digest != contract["manifest_sha256"]:
        raise ValueError("corpus manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "observations.parquet",
        "binding.parquet",
        "other_assays.parquet",
        "observation_contributors.parquet",
        "bulk_proteomics.parquet",
        "line_expression.parquet",
        "peptide_mappings.parquet",
        "peptide_mappings_meta.json",
    }
    files = manifest.get("files", {})
    if not required <= files.keys():
        raise ValueError("incomplete corpus generation")
    for filename, expected in files.items():
        if Path(filename).name != filename:
            raise ValueError("corpus files must be local filenames")
        path = directory / filename
        if path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"corpus size mismatch: {filename}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"corpus SHA-256 mismatch: {filename}")
    if file_sha256(manifest_path) != digest:
        raise ValueError("corpus manifest changed during verification")
    return {
        "contract": contract,
        "directory": str(directory),
        "files": files,
        "counts": manifest.get("counts", {}),
        "verified": True,
    }


def export_presentation(directory, output_dir, *, pin=None):
    """Create an uncapped export through installed Hitlist's public API."""
    receipt = verify_corpus(directory, pin=pin)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    command = [
        sys.executable,
        "-m",
        "presto.data.hitlist_corpus",
        "--export-worker",
        "--corpus-dir",
        receipt["directory"],
        "--output-dir",
        str(output),
    ]
    if pin:
        command.extend(["--pin", str(Path(pin).resolve())])
    environment = dict(os.environ, HITLIST_DATA_DIR=receipt["directory"])
    subprocess.run(command, env=environment, check=True)
    return json.loads((output / "export-receipt.json").read_text())


def _export(directory, output_dir, *, pin=None):
    import hitlist
    from hitlist.export import generate_ms_samples_table, generate_training_table
    from hitlist.observations import observations_path

    from .hitlist_source import training_columns

    started = time.monotonic()
    receipt = verify_corpus(directory, pin=pin)
    wanted = receipt["contract"]["loader_version"]
    if hitlist.__version__ != wanted or importlib.metadata.version("hitlist") != wanted:
        raise ValueError(f"this export contract requires runtime and installed Hitlist {wanted}")
    if observations_path().resolve() != Path(directory).resolve() / "observations.parquet":
        raise ValueError("Hitlist resolved a different index than the pinned corpus")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    columns = list(
        dict.fromkeys(
            [
                *training_columns("ms", include_flanks=False),
                *PRESENTATION_COLUMNS,
                "evidence_row_id",
                "evidence_source_id",
                "provenance_id",
                "provenance_status",
                "evidence_kind",
                "serotype",
                "serotypes",
                "serotype_source",
                "condition_genetic_variants",
                "condition_control_for",
            ]
        )
    )
    report = {
        "schema": "presto.presentation-export.v1",
        "corpus": receipt,
        "loader_version": wanted,
        "files": {},
        "training_membership": None,
        "policy": "Uncapped human MS evidence; original MHC/condition/serotype roles retained",
    }
    for cls in ("I", "II"):
        print(f"Exporting class {cls}", flush=True)
        frame = generate_training_table(
            include_evidence="ms", mhc_class=cls, species="Homo sapiens", columns=columns
        )
        required = {
            "peptide",
            "mhc_class",
            "evidence_row_id",
            "provenance_id",
            "condition_id",
            "sample_attribution",
            "restriction_evidence",
            "serotype",
            "serotypes",
            "serotype_source",
        }
        if not required <= set(frame.columns):
            raise ValueError(f"missing export fields: {sorted(required - set(frame.columns))}")
        if not frame["mhc_class"].eq(cls).all():
            raise ValueError("Hitlist returned a different MHC class")
        path = output / f"hitlist-{cls}.parquet"
        frame.to_parquet(path, index=False)
        report["files"][path.name] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
            "columns": list(frame.columns),
            "studies": int(frame.pmid.nunique()),
            "assigned_condition_rows": int(frame.condition_id.fillna("").ne("").sum()),
            "serotype_roles": {
                str(k): int(v) for k, v in frame.serotype_source.value_counts(dropna=False).items()
            },
        }
        frame.groupby(["pmid", "mhc_class"], observed=True, dropna=False).size().rename(
            "rows"
        ).to_csv(output / f"source-census-{cls}.csv")
        del frame
    samples = output / "samples.csv"
    generate_ms_samples_table().to_csv(samples, index=False)
    report["files"][samples.name] = {
        "sha256": file_sha256(samples),
        "bytes": samples.stat().st_size,
    }
    # Loader-side refresh must leave the archived input generation untouched.
    verify_corpus(directory, pin=pin)
    report["seconds"] = time.monotonic() - started
    (output / "export-receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-worker", action="store_true", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pin")
    args = parser.parse_args()
    _export(args.corpus_dir, args.output_dir, pin=args.pin)
