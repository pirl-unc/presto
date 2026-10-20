"""Audit original/effective IC50 targets without constructing a model."""

import hashlib
import importlib.metadata
import json
import shutil
import sys
import tarfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from datacache import install_bundle, materialize
from presto.training.supervision import LOSS_TASK_SPECS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]
ENDPOINT = "half maximal inhibitory concentration (IC50)"


def pin(path):
    p = Path(path)
    return {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}


def main():
    assert importlib.metadata.version("presto") == "0.1.13"
    receipt = json.loads((FAMILY / "results/interrupted-first-attempt.json").read_text())
    raw_paths = (
        {n: Path(sys.argv[1]) / n for n in receipt["files"]}
        if len(sys.argv) > 1
        else {n: Path(p) for n, p in receipt["local_paths"].items()}
    )
    for name, p in raw_paths.items():
        assert pin(p) == {k: receipt["files"][name][k] for k in ("sha256", "size_bytes")}
    inventory = json.loads(raw_paths["inventory.json"].read_text())
    files = {}
    with tarfile.open(raw_paths["run.tar.gz"]) as archive:
        for member in archive.getmembers():
            assert member.isfile() and member.name in inventory and member.name not in files
            data = archive.extractfile(member).read()
            assert {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            } == inventory[member.name]
            files[member.name] = data
    assert set(files) == set(inventory)
    assert not any(n.endswith(".pt") or n.endswith("predictions.csv") for n in files)
    interruption = json.loads(files["interruption.json"])
    assert interruption["n_checkpoint_files"] == interruption["n_prediction_files"] == 0
    invocation = json.loads(files["invocation.json"])
    assert (
        invocation["commit"] == "124c85b6da23d4c74dbef961999936c573025192"
        and not invocation["dirty"]
    )
    for name in ("fit.py", "baselines.py", "common.py", "diagnostic.py"):
        assert hashlib.sha256(files[name]).hexdigest() == invocation["code_files"][name]
        assert files[name] == (FAMILY / "reproduce/source" / name).read_bytes()
    sources = json.loads((FAMILY / "reproduce/frozen-inputs.json").read_text())
    paths = install_bundle(
        Path.home() / ".hitlist/presto-corrected-human-fit/audit-inputs",
        sources,
        download_options={"show_progress": False},
    )
    membership = pd.read_csv(paths["membership.tsv"], sep="\t", keep_default_na=False)
    contexts = pd.read_csv(paths["selected-source-contexts.tsv"], sep="\t", keep_default_na=False)
    selected = membership[membership.source_endpoint == ENDPOINT].copy()
    assert (
        len(selected) == 2614
        and selected.sample_id.is_unique
        and selected.evidence_row_id.is_unique
    )
    assert set(selected.source_units) == {"nM"}
    contexts = contexts.rename(
        columns={c: "native_" + c for c in contexts if c != "evidence_row_id"}
    )
    frame = selected.merge(
        contexts, on="evidence_row_id", validate="one_to_one", how="left", indicator=True
    )
    assert (frame._merge == "both").all() and (frame.peptide == frame.native_peptide).all()
    frame = frame.drop(columns="_merge")
    raw = pd.to_numeric(frame.source_value).to_numpy(float)
    assert np.isfinite(raw).all() and (raw > 0).all()
    spec = next(s for s in LOSS_TASK_SPECS if s.name == "binding_ic50")
    effective = spec.target_transform(torch.tensor(raw, dtype=torch.float32)).numpy().astype(float)
    frame["raw_target_log10_nM"] = np.log10(raw)
    frame["effective_target_log10_nM_float32"] = effective
    frame["bounded_source_value_nM"] = np.clip(raw, 0.001, 50000.0)
    assert np.allclose(effective, np.log10(frame.bounded_source_value_nM), rtol=1e-6, atol=1e-6)
    frame["physical_value_clipped"] = (raw < 0.001) | (raw > 50000.0)
    frame["target_shift_log10_nM"] = effective - frame.raw_target_log10_nM
    frame["label_kind"] = frame.source_inequality.map(
        {
            "": "exact",
            "=": "exact",
            "<": "left_censored",
            "<=": "left_censored",
            "≤": "left_censored",
            ">": "right_censored",
            ">=": "right_censored",
            "≥": "right_censored",
        }
    )
    assert not frame.label_kind.isna().any()
    frame["effective_target_policy"] = (
        "log10(clip(raw_nM, 0.001, 50000)); float32; original qualifier retained"
    )
    frame["interpretation"] = np.where(
        frame.physical_value_clipped,
        np.where(
            frame.label_kind == "exact",
            "Bounded target differs from the exact measured value",
            "Bounded target changes the original censor boundary",
        ),
        "Physical value unchanged; float32 transformation may round",
    )
    rows = []
    for split in ("train", "val", "test"):
        f = frame[frame.source_partition == split]
        affected = f[f.physical_value_clipped]
        rows.append(
            {
                "split": split,
                "n_observations": len(f),
                "n_exact": int((f.label_kind == "exact").sum()),
                "n_clipped_observations": len(affected),
                "n_clipped_exact": int((affected.label_kind == "exact").sum()),
                "n_clipped_left_censored": int((affected.label_kind == "left_censored").sum()),
                "n_clipped_right_censored": int((affected.label_kind == "right_censored").sum()),
                "raw_min_nM": float(pd.to_numeric(f.source_value).min()),
                "raw_max_nM": float(pd.to_numeric(f.source_value).max()),
            }
        )
    assert [
        (r["n_clipped_observations"], r["n_clipped_exact"], r["n_clipped_right_censored"])
        for r in rows
    ] == [(157, 39, 118), (31, 8, 23), (23, 7, 16)]
    versions = {
        p: importlib.metadata.version(p)
        for p in ("presto", "torch", "numpy", "pandas", "matplotlib", "datacache")
    }
    summary = {
        "status": "fitting attempt stopped before optimization; target policy mismatch",
        "original_recipe_commit": invocation["commit"],
        "producer": invocation["installed_producer"],
        "canonical_task": spec.name,
        "effective_policy": "log10(clip(raw_nM, 0.001, 50000)); original qualifier retained",
        "planned_policy": "log10(raw_nM); original qualifier retained",
        "counts": rows,
        "n_source_verified_observations": len(frame),
        "n_clipped_observations": int(frame.physical_value_clipped.sum()),
        "n_clipped_exact": int(
            ((frame.label_kind == "exact") & frame.physical_value_clipped).sum()
        ),
        "n_verified_interrupted_archive_members": len(files),
        "n_checkpoints": 0,
        "n_predictions": 0,
        "analysis_versions": versions,
        "issue": "https://github.com/pirl-unc/presto/issues/110",
        "limitations": [
            "No Presto optimization, performance result or replacement target policy is qualified.",
            "Complete buffered training log and full preflight completion are unavailable for the interrupted process.",
            "Existing canonical clipping is retained; changing the scientific objective is a separate explicit decision.",
            "Unknown donor/specimen/acquisition lineage and shared studies remain unchanged.",
        ],
    }
    names = (
        "target-policy-audit.json",
        "target-policy-counts.tsv",
        "ic50-raw-effective-targets.tsv",
        "target-policy-preview.tsv",
        "target-policy.png",
    )

    def builder(inputs, outputs):
        Path(outputs["target-policy-audit.json"]).write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        pd.DataFrame(rows).to_csv(outputs["target-policy-counts.tsv"], sep="\t", index=False)
        ordered = frame.sort_values(
            ["source_partition", "physical_value_clipped", "label_kind", "sample_id"],
            ascending=[True, False, True, True],
        )
        ordered.to_csv(outputs["ic50-raw-effective-targets.tsv"], sep="\t", index=False)
        preview_fields = [
            "source_partition",
            "sample_id",
            "evidence_row_id",
            "peptide",
            "pmid",
            "doi",
            "assay_iri",
            "reference_iri",
            "source_mhc_alleles",
            "resolved_mhc_alleles",
            "native_mhc_class",
            "native_cell_line_name",
            "native_cell_type",
            "native_condition_id",
            "native_condition_status",
            "native_assay_method",
            "source_endpoint",
            "source_value",
            "source_units",
            "source_inequality",
            "raw_target_log10_nM",
            "effective_target_log10_nM_float32",
            "physical_value_clipped",
            "interpretation",
        ]
        ordered.groupby(
            ["source_partition", "physical_value_clipped", "label_kind"], sort=False
        ).head(2)[preview_fields].to_csv(
            outputs["target-policy-preview.tsv"], sep="\t", index=False
        )
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for kind, color in (("exact", "#16627A"), ("right_censored", "#B75A30")):
            f = frame[frame.label_kind == kind]
            axis.scatter(
                f.raw_target_log10_nM,
                f.effective_target_log10_nM_float32,
                s=14,
                alpha=0.35,
                label=kind.replace("_", " "),
                color=color,
            )
        axis.plot([-1.5, 6], [-1.5, 6], color="gray", ls="--", lw=1, label="Unchanged target")
        axis.axhline(np.log10(50000), color="black", ls=":", lw=1, label="50,000 nM cap")
        axis.set(
            xlabel="Original measurement / bound [log10(nM)]",
            ylabel="Canonical effective target [log10(nM)]",
            title="IC50 label transformation — no model predictions",
        )
        axis.legend(fontsize=9)
        fig.savefig(outputs["target-policy.png"], dpi=180)
        plt.close(fig)

    inputs = {name: {"path": str(p), "sha256": pin(p)["sha256"]} for name, p in paths.items()}
    inputs["interrupted-run.tar.gz"] = {
        "path": str(raw_paths["run.tar.gz"]),
        "sha256": pin(raw_paths["run.tar.gz"])["sha256"],
    }
    outputs = materialize(
        Path.home()
        / ".hitlist/presto-corrected-human-fit/target-policy-audit"
        / pin(__file__)["sha256"],
        inputs,
        transform={
            "version": "canonical-raw-effective-ic50-label-audit-v1",
            "options": {"code_sha256": pin(__file__)["sha256"], "analysis_versions": versions},
        },
        outputs={name: {} for name in names},
        builder=builder,
    )
    for name in names:
        if name != "ic50-raw-effective-targets.tsv":
            shutil.copyfile(outputs[name], FAMILY / "results" / name)
    result = {
        "summary": summary,
        "artifacts": {n: {"path": str(p), **pin(p)} for n, p in outputs.items()},
    }
    (FAMILY / "results/target-policy-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
