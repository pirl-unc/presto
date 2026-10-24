"""Independently reconcile all original cells and targets without model evaluation."""

import io
import json
import math
import resource
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"
RAW = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
started = time.perf_counter()
assert not (ROOT / "results/source-verification.json").exists()
shutil.copy2(__file__, ROOT / "reproduce/source/verify_admission.py")
selection = json.loads((ROOT / "results/selection.json").read_text())
examples = json.loads((DATA / "recipe/examples.json").read_text())
typed = {
    name: pq.read_table(
        DATA / "source/proteometools_ms1/typed" / f"{name}.parquet"
    ).to_pylist()
    for name in ("runs", "peptides", "forms", "source_records")
}
records = {r["source_record_id"]: r for r in typed["source_records"]}
runs = {r["pool_name"]: r for r in typed["runs"]}
n_cells, n_values = 0, 0
results = []
for pool in selection["selected"]:
    run = runs[pool["pool"]]
    originals = {}
    with zipfile.ZipFile(RAW / pool["source_name"]) as archive:
        for member, table in [
            ("peptides.txt", "peptides"),
            ("modificationSpecificPeptides.txt", "forms"),
        ]:
            frame = pd.read_csv(
                io.BytesIO(archive.read(member)),
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )
            originals[table] = frame
            projected = {
                r["source_id_raw"]: r
                for r in typed[table]
                if r["run_id"] == run["run_id"]
            }
            assert len(frame) == len(projected) == frame.id.nunique()
            for original in frame.to_dict(orient="records"):
                value = projected[original["id"]]
                record = json.loads(
                    records[value["source_record_id"]]["source_row_json"]
                )
                assert record["headers"] == list(frame.columns)
                assert record["cells"] == [original[name] for name in frame.columns]
                n_cells += len(original)
                assert (
                    value["peptide_sequence"] == original["Sequence"]
                    and value["quantitative_raw"] == original["Intensity"]
                )
    peptide = originals["peptides"]
    form = originals["forms"]

    def quality(frame):
        intensity = pd.to_numeric(frame.Intensity, errors="coerce")
        pep = pd.to_numeric(frame.PEP, errors="coerce")
        return (
            np.isfinite(intensity)
            & (intensity > 0)
            & np.isfinite(pep)
            & (pep >= 0)
            & (pep < 0.01)
            & (frame.Reverse == "")
            & (frame["Potential contaminant"] == "")
        )

    seq_ok = set(
        peptide[
            quality(peptide)
            & peptide.Sequence.isin(pool["intended_sequences"])
            & peptide.Sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
        ].Sequence
    )
    raw_form_counts = (
        form[form.Modifications == "Unmodified"].groupby("Sequence").size()
    )
    unique = set(raw_form_counts[raw_form_counts == 1].index)
    admitted = form[
        quality(form)
        & form.Sequence.isin(seq_ok & unique)
        & (form.Modifications == "Unmodified")
    ]
    selected = [row for row in examples if row["pool_name"] == pool["pool"]]
    form_identity = {
        r["measurement_id"]: r["source_id_raw"]
        for r in typed["forms"]
        if r["run_id"] == run["run_id"]
    }
    assert {form_identity[r["measurement_id"]] for r in selected} == set(admitted.id)
    assert len(selected) == len(admitted)
    values = {
        r["Sequence"]: math.log10(float(r["Intensity"]))
        for r in admitted.to_dict(orient="records")
    }
    center = math.fsum(values.values()) / len(values)
    for row in selected:
        assert abs(row["log10_signal"] - values[row["sequence"]]) < 1e-12
        assert (
            abs(
                row["target_centered_log10_signal"] - (values[row["sequence"]] - center)
            )
            < 1e-12
        )
        n_values += 1
    results.append(
        {
            "pool": pool["pool"],
            "split": pool["split"],
            "n_original_peptide_rows": len(peptide),
            "n_original_form_rows": len(form),
            "n_verified_examples": len(selected),
        }
    )
report = {
    "status": "passed",
    "n_original_literal_cells_verified": n_cells,
    "n_measured_targets_verified": n_values,
    "pools": results,
    "elapsed_seconds": time.perf_counter() - started,
    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    "predictive_model_evaluation": False,
}
assert report["peak_rss_gib"] <= 3
(ROOT / "results/source-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps({k: v for k, v in report.items() if k != "pools"}), flush=True)
