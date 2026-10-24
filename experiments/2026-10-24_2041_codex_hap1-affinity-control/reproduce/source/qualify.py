"""Freeze predictor provenance and generate training-covariate scores only."""

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RECIPE = ROOT / "data/hap1_response_recipes/2026-10-24_0730_v1"
PACKAGE = Path("/Users/iskander/code/mhcflurry")
BUNDLE = Path(
    "/Users/iskander/Library/Application Support/mhcflurry/4/2.3.0/models_class1_presentation/models"
)
MODEL = BUNDLE / "affinity_predictor"
ALLELES = ["HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    start = time.perf_counter()
    out = EXP / "results/predictor"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, EXP / "reproduce/source/qualify.py")
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE, text=True).strip()
        == "cd141c8c3c4eb393ab7a53fd4aa40a981cf5d10f"
    )
    assert not subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=PACKAGE, text=True
    ).strip()
    paths = sorted(
        [
            *MODEL.glob("*"),
            BUNDLE / "affinity_predictor_train_data.csv.bz2",
            BUNDLE / "inference_export.json",
            BUNDLE / "GENERATE.sh",
            *sorted((PACKAGE / "mhcflurry").rglob("*.py")),
        ]
    )
    hashes = {str(p): digest(p) for p in paths if p.is_file()}
    assert (
        digest(RECIPE / "manifest.json")
        == "a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309"
    )
    covariates = pq.read_table(
        RECIPE / "derived/responses.parquet",
        filters=[("split", "=", "train"), ("precursor_status", "=", "complete")],
        columns=["peptide_sequence", "condition_id", "component_id", "source_record_id"],
    )
    peptides = sorted(set(covariates["peptide_sequence"].to_pylist()))
    assert len(peptides) == 6317 and len(covariates) == 87260
    training = pd.read_csv(
        BUNDLE / "affinity_predictor_train_data.csv.bz2", dtype={"peptide": str, "allele": str}
    )
    marker = (
        training.select_dtypes(include=["object", "str"])
        .astype(str)
        .apply(
            lambda col: col.str.contains("HAP1|Shapiro|40113210|PXD056426", case=False, regex=True)
        )
        .any(axis=1)
    )
    training[marker].to_parquet(out / "source-marker-matches.parquet", index=False)
    assert not marker.any(), "Possible HAP1 source exposure: stop before fitting"
    training_set = set(training.peptide)
    pairs = set(zip(training.peptide, training.allele, strict=True))
    mass_spec = set(training.loc[training.measurement_kind.eq("mass_spec"), "peptide"])
    exposure = pd.DataFrame(
        {
            "peptide": peptides,
            "any_training_peptide": [p in training_set for p in peptides],
            "mass_spec_training_peptide": [p in mass_spec for p in peptides],
        }
    )
    for allele in ALLELES:
        exposure[allele] = [(p, allele) in pairs for p in peptides]
    exposure.to_parquet(out / "exposure.parquet", index=False)
    configs = pd.read_csv(MODEL / "manifest.csv")
    provenance = {
        "n_training_rows": len(training),
        "columns": list(training.columns),
        "training_kinds": training.measurement_kind.value_counts(dropna=False).to_dict(),
        "training_sources": training.measurement_source.value_counts(dropna=False).to_dict(),
        "n_hap1_source_marker_matches": int(marker.sum()),
        "source_marker_search_is_complete_lineage_audit": False,
        "exposure_counts": {k: int(exposure[k].sum()) for k in exposure if k != "peptide"},
        "member_training_stages": {
            r.model_name: [
                i.get("training_info", {}) for i in json.loads(r.config_json)["fit_info"]
            ]
            for r in configs.itertuples()
        },
        "lineage_limits": "Bundled table exposes peptide/allele/kind/source but no source_provenance or specimen IDs. Teacher-pretraining and ensemble-selection sample lineage are not completely reconstructed. No sample-disjoint or pure-affinity claim.",
        "api_semantics": "Frozen predicted nM features; training includes reassigned MS ligands and teacher pretraining. Not measured Kd or changed KO affinity.",
    }
    write(out / "provenance.json", provenance)
    receipt = {
        "argv": sys.argv,
        "python": sys.executable,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True
        ),
        "predictor_source_commit": "cd141c8c3c4eb393ab7a53fd4aa40a981cf5d10f",
        "predictor_version": importlib.metadata.version("mhcflurry"),
        "model_hashes": hashes,
        "alleles": ALLELES,
        "recipe_manifest_sha256": digest(RECIPE / "manifest.json"),
        "plan_sha256": digest(EXP / "reproduce/plan.md"),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "PYTHONDONTWRITEBYTECODE",
                "MHCFLURRY_TORCH_COMPILE",
                "MHCFLURRY_MATMUL_PRECISION",
            ]
        },
        "versions": {
            k: importlib.metadata.version(k) for k in ["torch", "numpy", "pandas", "pyarrow"]
        },
        "outer_outcomes_loaded": False,
        "hap1_targets_loaded": False,
    }
    write(EXP / "reproduce/predictor-launch.json", receipt)
    del training
    import torch
    from mhcflurry import Class1AffinityPredictor
    from mhcflurry.common import configure_pytorch

    configure_pytorch(backend="cpu", num_threads=1)
    torch.set_num_interop_threads(1)
    assert receipt["predictor_version"] == "2.3.14"
    predictor = Class1AffinityPredictor.load(str(MODEL), optimization_level=0)
    assert len(predictor.class1_pan_allele_models) == 11
    frames, errors = [], {}
    scoring = time.perf_counter()
    for allele in ALLELES:
        frame = predictor.predict_to_dataframe(
            peptides,
            allele=allele,
            include_individual_model_predictions=True,
            include_percentile_ranks=False,
            include_confidence_intervals=False,
            model_kwargs={"batch_size": 1024},
        )
        member_columns = [c for c in frame if c.startswith("model_pan_")]
        if len(member_columns) != 11:
            raise ValueError(("Unexpected member columns", list(frame.columns)))
        raw = frame.prediction.to_numpy()
        assert np.isfinite(raw).all() and (raw > 0).all()
        individual = frame[member_columns].to_numpy()
        delta = float(np.max(abs(np.log10(raw) - np.mean(np.log10(individual), axis=1))))
        assert delta < 1e-10, delta
        # Reversed order and a different minibatch with the same frozen parameters.
        probe_indices = np.linspace(0, len(peptides) - 1, 64, dtype=int)[::-1]
        probe = predictor.predict(
            [peptides[i] for i in probe_indices], allele=allele, model_kwargs={"batch_size": 17}
        )
        batch_delta = float(np.max(abs(np.log10(probe) - np.log10(raw[probe_indices]))))
        assert batch_delta < 1e-5, batch_delta
        frames.append(frame)
        errors[allele] = {
            "member_aggregation_log10_error": delta,
            "rebatched_log10_error": batch_delta,
        }
        frame.to_parquet(
            out / (allele.replace("*", "_").replace(":", "_") + ".parquet"), index=False
        )
        print(
            json.dumps({"allele": allele, "n_peptides": len(frame), **errors[allele]}), flush=True
        )
        assert time.perf_counter() - scoring < 300
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3 < 6
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_parquet(out / "scores.parquet", index=False)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    result = {
        "status": "passed",
        "n_peptides": len(peptides),
        "n_scores": len(all_scores),
        "errors": errors,
        "scoring_seconds": time.perf_counter() - scoring,
        "total_seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "scores_sha256": digest(out / "scores.parquet"),
        "model_hashes_unchanged": True,
        "no_neural_training": True,
        "outer_validation_test_scored": False,
    }
    write(out / "qualification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
