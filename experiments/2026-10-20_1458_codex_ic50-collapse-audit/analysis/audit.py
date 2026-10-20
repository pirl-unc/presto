"""Replay training-only class-I rows and measure between-example contraction."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

import datacache
import numpy as np
import pandas as pd
import torch

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.mhc_sequence_resolver import resolve_exact_mhc_inputs
from presto.training.checkpointing import load_model_from_checkpoint


def stats(array):
    x = np.asarray(array, dtype=np.float64).reshape(len(array), -1)
    return {
        "n_examples": len(x),
        "n_features": x.shape[1],
        "between_example_rms_std": float(np.sqrt(x.var(axis=0).mean())),
        "mean_feature_rms": float(np.sqrt(np.square(x.mean(axis=0)).mean())),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/private/tmp/presto-ic50-source")
    parser.add_argument("--run", default="/private/tmp/presto-ic50-run")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    family = Path(__file__).resolve().parents[1]
    pub = json.loads(
        (root / "experiments/2026-10-16_1030_codex_effective-target-fit/results/publication.json")
        .read_text()
    )
    pins = {n: pub["files"][n] for n in ("run.tar.gz", "ic50-inspection.tsv", "output_updates.json")}
    paths = datacache.install_bundle(
        args.cache, pins, download_options={"show_progress": False}
    )
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    with tarfile.open(paths["run.tar.gz"]) as archive:
        archive.extractall(run, filter="data")
    train = pd.read_csv(run / "training-known-examples.tsv", sep="\t", keep_default_na=False)
    assert len(train) == 2074 and set(train.source_partition) == {"train"}
    subset = train.loc[train.native_mhc_class.eq("I")].sort_values("sample_id").copy()
    assert len(subset) == 877
    alleles_needed = sorted({a for raw in subset.resolved_mhc_alleles for a in json.loads(raw)})
    catalog, coverage = resolve_exact_mhc_inputs(
        alleles_needed, mhcseqs_search_dir=str(run / "mhc-catalog")
    )
    assert coverage["missing"] == 0
    samples = []
    for row in subset.itertuples():
        alleles = json.loads(row.resolved_mhc_alleles)
        assert len(alleles) == 1
        c = catalog[alleles[0]]
        assert c.groove1 and c.groove2
        samples.append(PrestoSample(
            peptide=row.peptide, mhc_a=c.groove1, mhc_b=c.groove2, mhc_class="I",
            sample_id=row.sample_id,
        ))
    collate = PrestoCollator()
    results = {"scope": "Training-only class I, no fitting or held-out evaluation",
               "n_training_ic50": len(train), "n_replayed": len(subset), "conditions": {}}
    scalar_keys = [
        "binding_affinity_score_raw", "binding_affinity_probe_kd",
        "binding_core_kd_log10", "binding_mixed_kd_log10",
        "binding_kd_bias", "binding_logit_from_core",
    ]
    for condition, checkpoint in (("initial", "initial_checkpoint.pt"), ("selected", "checkpoint.pt")):
        model, payload = load_model_from_checkpoint(run / checkpoint)
        model.eval()
        collected = {k: [] for k in ["pep_vec", "mhc_a_vec", "mhc_b_vec",
                     "pmhc_interaction_vec", "binding_affinity_vec", "IC50", "KD",
                     "ic50_residual", *scalar_keys]}
        residual = []
        handle = model.affinity_predictor.assay_heads.ic50_residual.register_forward_hook(
            lambda module, inputs, output: residual.append(output.detach().cpu().numpy())
        )
        with torch.no_grad():
            for start in range(0, len(samples), 32):
                output = model(**collate(samples[start:start + 32]).model_inputs())
                for key in ["pep_vec", "mhc_a_vec", "mhc_b_vec", "pmhc_interaction_vec", *scalar_keys]:
                    collected[key].append(output[key].cpu().numpy())
                collected["binding_affinity_vec"].append(
                    output["latent_vecs"]["binding_affinity"].cpu().numpy()
                )
                collected["IC50"].append(output["assays"]["IC50_nM"].cpu().numpy())
                collected["KD"].append(output["assays"]["KD_nM"].cpu().numpy())
                collected["ic50_residual"].append(residual[-1])
        handle.remove()
        arrays = {k: np.concatenate(v) for k, v in collected.items()}
        predicted = arrays["IC50"].reshape(-1)
        reference = subset[f"{condition}_prediction_log10_nM"].to_numpy(float)
        difference = float(np.max(np.abs(predicted - reference)))
        # Reproduction tolerance for CPU/batch-size differences. Do not explain
        # a different predictor as the historical model.
        if difference > 2e-5:
            raise ValueError(f"{condition} replay differs from source predictions: {difference}")
        subset[f"replayed_{condition}"] = predicted
        exact = subset.qualifier.eq(0).to_numpy()
        results["conditions"][condition] = {
            "checkpoint_sha256": hashlib.sha256((run / checkpoint).read_bytes()).hexdigest(),
            "epoch": payload.get("epoch"),
            "max_abs_replay_error": difference,
            "feature_statistics": {k: stats(v) for k, v in arrays.items()},
            "exact_rmse": float(np.sqrt(np.mean(
                (predicted[exact] - subset.target_log10_nM.to_numpy()[exact]) ** 2))),
            "target": stats(subset.target_log10_nM.to_numpy()),
            "binding_probe_mix_weight": float(model.affinity_predictor.binding_probe_mix_logit.sigmoid()),
        }
        mix = results["conditions"][condition]["binding_probe_mix_weight"]
        core = arrays["binding_core_kd_log10"].ravel()
        probe = arrays["binding_affinity_probe_kd"].ravel()
        results["conditions"][condition]["mixture"] = {
            "core_probe_pearson": float(np.corrcoef(core, probe)[0, 1]),
            "weighted_core_std": float(np.std((1 - mix) * core)),
            "weighted_probe_std": float(np.std(mix * probe)),
            "mixed_std": float(np.std((1 - mix) * core + mix * probe)),
        }
    metrics = pd.read_csv(run / "metrics.csv")
    selected_metrics = metrics[
        metrics.split.eq("train") & metrics.metric.str.contains(
            "loss_binding_ic50|batch_supervised_weight_binding_ic50|"
            "uw_weight_binding_ic50|batch_support_binding_ic50"
        )
    ]
    selected_metrics.to_csv(family / "results/optimization-summary.csv", index=False)
    updates = json.loads((run / "output_updates.json").read_text())
    ic50_updates = [r for r in updates["outputs"] if r["endpoint"] == "assays.IC50_nM"]
    results["ic50_update_evidence"] = ic50_updates
    results["raw_artifacts"] = {k: str(v) for k, v in paths.items()}
    results["source_pins"] = pins
    results["runtime_seconds"] = time.monotonic() - started
    subset[["sample_id", "evidence_row_id", "peptide", "resolved_mhc_alleles",
            "source_partition", "qualifier", "target_log10_nM", "replayed_initial",
            "replayed_selected"]].to_csv(family / "results/train-class-I-replay.csv", index=False)
    (family / "results/audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["conditions"], indent=2))


if __name__ == "__main__":
    main()
