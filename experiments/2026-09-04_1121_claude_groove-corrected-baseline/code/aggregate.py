"""Validate and aggregate the paired source-junction masking experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presto.training.holdout_eval import (  # noqa: E402
    binary_metrics,
    binding_threshold_metrics,
    regression_metrics,
)

EXPECTED_HASHES = {
    "binding.parquet": "fbcae6f762f43edb4eb87b1a7c7f3849757859204d37f3ce82d1f83c23cece9f",
    "observations.parquet": "f51440ab229fd187d2548b4dddcd1fc04580d97d45fb4d5b8e0222aa8080f928",
    "observations_meta.json": "ac459e184fc6c54c73f7f1b4fc7dff424b2360b13f55f12bb68a3bbb743de118",
    "peptide_mappings.parquet": "45580c16649daf75b11b51497d9d96dfaa987cdcf1197d6449deaa9261f6ec5c",
    "peptide_mappings_meta.json": (
        "9a93de21753029ac08f1ba05e1a667ee6d7fd1b63a885bdc8dade1e626c7cbbb"
    ),
}
EXPECTED_MHC_INDEX_HASH = "497938937f01394aeb18a3db15314f04ac1be162efe2844a1f018bcaff121063"
EXPECTED_POLICIES = ("legacy_global_canonical", "mask_unresolved")
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_PACKAGE_VERSIONS = {
    "hitlist": "1.55.8",
    "mhcseqs": "2.6.10",
    "mhcgnomes": "3.41.0",
}
EXPECTED_CONFIG = {
    "data_source": "hitlist",
    "hitlist_allele": "HLA-A*02:01",
    "max_binding": 0,
    "max_kinetics": 500,
    "max_stability": 1000,
    "max_processing": 0,
    "max_elution": 20000,
    "max_tcell": 0,
    "max_vdjdb": 0,
    "cap_sampling": "reservoir",
    "data_seed": 42,
    "exclude_target": ["kon", "koff", "tm"],
    "synthetic_pmhc_negative_ratio": 0.0,
    "synthetic_elution_negative_ratio": 1.0,
    "synthetic_cascade_elution_negative_ratio": 0.0,
    "synthetic_cascade_tcell_negative_ratio": 0.0,
    "synthetic_class_i_no_mhc_beta_negative_ratio": 0.0,
    "synthetic_processing_negative_ratio": 0.0,
    "mhc_augmentation_samples": 0,
    "uniprot_negative_ratio": 0.0,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "split_mode": "peptide_group",
    "epochs": 10,
    "batch_size": 256,
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "latent_topology": "expanded",
    "lr": 2.8e-4,
    "weight_decay": 0.01,
    "balanced_batches": False,
    "require_all_active_target_support": True,
    "require_binary_balance_target": ["elution"],
    "require_all_active_binary_balance": True,
    "require_traceable_lineage": True,
    "forbid_fake_null_sequences": True,
}
SPLITS = ("val", "test")
#: The ambiguous-junction stratum this family reports as `unresolved_union`.
#: Deliberately the two genuinely ambiguous categories and not the library's
#: full unresolved set, which also contains `unmapped`: an unmapped row has no
#: source protein, so it answers a different question than "which protein's
#: junction did we pick". Frozen here so the stratum stays comparable across
#: the 20260902b / 20260903a / 20260904a families.
UNRESOLVED_CATEGORIES = frozenset({"cross_gene_unresolved", "within_gene_unresolved"})
SCOPES = (
    "overall",
    "single",
    "flanks_agree",
    "within_gene_canonical",
    "cross_gene_unresolved",
    "within_gene_unresolved",
    "unresolved_union",
    "unmapped",
)
PAIRING_FIELDS = (
    "task",
    "sample_id",
    "peptide",
    "source_mhc_alleles",
    "resolved_mhc_alleles",
    "source",
    "qualifier",
    "y_true",
    "evidence_row_id",
    "assay_iri",
    "reference_iri",
    "pmid",
    "source_sample_label",
    "source_sample_attribution",
    "source_mapping_category",
    "source_mapping_n_candidates",
    "source_mapping_n_genes",
    "source_mapping_n_flank_pairs",
    "mapping_gene_name",
    "mapping_gene_id",
    "mapping_protein_id",
    "mapping_transcript_id",
    "mapping_position",
    "mapping_proteome",
    "mapping_proteome_source",
    "mapping_is_canonical_transcript",
)
REQUIRED_RUN_FILES = (
    "condition_result.json",
    "config.json",
    "run.log",
    "data_contract.json",
    "hardware.json",
    "metrics.csv",
    "mhc_sequence_coverage.json",
    "mhc_sequence_coverage.csv",
    "val_summary.json",
    "test_summary.json",
    "val_metrics.csv",
    "test_metrics.csv",
    "val_predictions.csv",
    "test_predictions.csv",
    "split_support.json",
    "split_support.csv",
    "data_funnel.json",
    "data_funnel.csv",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _embedded_sha_matches(payload: dict[str, Any]) -> bool:
    claimed = payload.get("sha256")
    unhashed = {key: value for key, value in payload.items() if key != "sha256"}
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return bool(claimed) and claimed == hashlib.sha256(canonical).hexdigest()


def _peak_gpu_memory(path: Path) -> dict[str, float]:
    """Maximum observed CUDA allocation/reservation across training epochs."""
    wanted = {
        "gpu_peak_allocated_gib": "gpu_peak_allocated_gib",
        "gpu_peak_reserved_gib": "gpu_peak_reserved_gib",
    }
    peaks = {output: 0.0 for output in wanted.values()}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output = wanted.get(str(row.get("metric", "")))
            if output is not None:
                peaks[output] = max(peaks[output], float(row["value"]))
    return peaks


def _binding_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["task"] == "binding"]


def _scope_rows(rows: list[dict[str, str]], scope: str) -> list[dict[str, str]]:
    if scope == "overall":
        return rows
    if scope == "unresolved_union":
        return [row for row in rows if row["source_mapping_category"] in UNRESOLVED_CATEGORIES]
    return [row for row in rows if row["source_mapping_category"] == scope]


def _metrics(
    rows: list[dict[str, str]], *, split: str, policy: str, seed: int, scope: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": split,
        "policy": policy,
        "seed": seed,
        "scope": scope,
        "n_all": len(rows),
    }
    if not rows:
        result["n_exact"] = 0
        result["n_threshold_500nm"] = 0
        return result

    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=float)
    y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=float)
    qualifiers = np.asarray([int(row["qualifier"]) for row in rows], dtype=int)

    for name, value in regression_metrics(y_true, y_pred).items():
        result[f"all_{name}"] = value

    exact = qualifiers == 0
    result["n_exact"] = int(exact.sum())
    if exact.any():
        for name, value in regression_metrics(y_true[exact], y_pred[exact]).items():
            result[f"exact_{name}"] = value

    threshold = binding_threshold_metrics(y_true, y_pred, qualifiers)
    result["n_threshold_500nm"] = int(threshold.get("n", 0))
    for name, value in threshold.items():
        if name != "n":
            result[f"threshold_500nm_{name}"] = value
    return result


def _pairing_signature(rows: Iterable[dict[str, str]]) -> Counter[tuple[str, ...]]:
    return Counter(tuple(row[field] for field in PAIRING_FIELDS) for row in rows)


def _decoy_metrics(
    rows: list[dict[str, str]],
    *,
    split: str,
    policy: str,
    seed: int,
    task: str,
    decoy_family: str,
) -> dict[str, Any]:
    """Report generated-decoy discrimination without calling it biological accuracy."""
    synthetic = [str(row["source"]).startswith("synthetic_negative") for row in rows]
    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=float)
    y_score = np.asarray([float(row["y_prob"]) for row in rows], dtype=float)
    result: dict[str, Any] = {
        "split": split,
        "policy": policy,
        "seed": seed,
        "task": task,
        "scope": "generated_decoy_discrimination",
        "decoy_family": decoy_family,
        "n": len(rows),
        "n_real_positive": sum(
            not is_synthetic and target >= 0.5 for is_synthetic, target in zip(synthetic, y_true)
        ),
        "n_real_negative": sum(
            not is_synthetic and target < 0.5 for is_synthetic, target in zip(synthetic, y_true)
        ),
        "n_synthetic_positive": sum(
            is_synthetic and target >= 0.5 for is_synthetic, target in zip(synthetic, y_true)
        ),
        "n_synthetic_negative": sum(
            is_synthetic and target < 0.5 for is_synthetic, target in zip(synthetic, y_true)
        ),
    }
    if len(rows) and len(np.unique(y_true)) == 2:
        result.update(binary_metrics(y_true, y_score))
    return result


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return float(np.mean(finite)) if finite else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(EXPERIMENT_DIR / "results" / "runs"))
    parser.add_argument("--results-dir", default=str(EXPERIMENT_DIR / "results"))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(EXPERIMENT_DIR / "manifest.json")
    manifest_by_condition = {(str(row["policy"]), int(row["seed"])): row for row in manifest}
    expected = {(policy, seed) for policy in EXPECTED_POLICIES for seed in EXPECTED_SEEDS}
    if set(manifest_by_condition) != expected:
        raise RuntimeError(
            f"Manifest conditions differ from the registered contract: "
            f"{sorted(manifest_by_condition)}"
        )

    run_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    predictions: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    contract_checks: list[dict[str, Any]] = []
    for condition in sorted(expected):
        manifest_row = manifest_by_condition[condition]
        if manifest_row.get("status") != "complete":
            raise RuntimeError(f"Condition is not complete: {manifest_row}")
        run_id = str(manifest_row["run_id"])
        run_dir = runs_dir / run_id
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{run_id} is missing artifacts: {missing}")

        condition_result = _read_json(run_dir / "condition_result.json")
        config = _read_json(run_dir / "config.json")
        data_contract = _read_json(run_dir / "data_contract.json")
        hardware = _read_json(run_dir / "hardware.json")
        split_support = _read_json(run_dir / "split_support.json")
        data_funnel = _read_json(run_dir / "data_funnel.json")
        peak_memory = _peak_gpu_memory(run_dir / "metrics.csv")
        checks = {
            "run_id": run_id,
            "policy": condition[0],
            "seed": condition[1],
            "status_complete": condition_result.get("status") == "complete",
            "condition_identity_matches": (
                condition_result.get("run_id") == run_id
                and condition_result.get("policy") == condition[0]
                and condition_result.get("seed") == condition[1]
            ),
            "config_matches": (
                all(config.get(key) == value for key, value in EXPECTED_CONFIG.items())
                and config.get("source_mapping_policy") == condition[0]
                and config.get("seed") == condition[1]
            ),
            "data_hashes_match": data_contract.get("artifact_sha256") == EXPECTED_HASHES,
            "mhc_index_hash_matches": data_contract.get("mhc_index_sha256")
            == EXPECTED_MHC_INDEX_HASH,
            "requested_gpu_matches": hardware.get("requested_gpu") == "H100!",
            "observed_h100": "H100" in str(hardware.get("nvidia_smi", "")),
            "observed_gpu_memory": all(value > 0.0 for value in peak_memory.values()),
            "package_versions_match": hardware.get("package_versions") == EXPECTED_PACKAGE_VERSIONS,
            "data_contract_matches_condition": (
                data_contract.get("policy") == condition[0]
                and data_contract.get("seed") == condition[1]
                and data_contract.get("data_seed") == 42
                and {
                    "hitlist": data_contract.get("hitlist_version"),
                    "mhcseqs": data_contract.get("mhcseqs_version"),
                    "mhcgnomes": data_contract.get("mhcgnomes_version"),
                }
                == EXPECTED_PACKAGE_VERSIONS
            ),
            "preflight_support_hash_matches": split_support.get("sha256")
            == data_contract.get("preflight_split_support_sha256"),
            "split_support_hash_valid": _embedded_sha_matches(split_support),
            "data_funnel_hash_valid": _embedded_sha_matches(data_funnel),
        }
        failed = [
            name
            for name, value in checks.items()
            if name not in {"run_id", "policy", "seed"} and not value
        ]
        if failed:
            raise RuntimeError(f"Invalid run contract for {run_id}: {failed}")
        contract_checks.append(checks)
        run_metadata[condition] = {
            **condition_result,
            "nvidia_smi": hardware["nvidia_smi"],
            **peak_memory,
        }
        for split in SPLITS:
            predictions[(condition[0], condition[1], split)] = _read_predictions(
                run_dir / f"{split}_predictions.csv"
            )

    parity_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for split in SPLITS:
            legacy = predictions[("legacy_global_canonical", seed, split)]
            masked = predictions[("mask_unresolved", seed, split)]
            legacy_signature = _pairing_signature(legacy)
            masked_signature = _pairing_signature(masked)
            parity = legacy_signature == masked_signature
            row = {
                "seed": seed,
                "split": split,
                "legacy_rows": len(legacy),
                "masked_rows": len(masked),
                "identical_supervision_rows": parity,
                "legacy_only_rows": sum((legacy_signature - masked_signature).values()),
                "masked_only_rows": sum((masked_signature - legacy_signature).values()),
            }
            parity_rows.append(row)
            if not parity:
                raise RuntimeError(f"Policy pair is not data-identical: {row}")

    metric_rows: list[dict[str, Any]] = []
    decoy_metric_rows: list[dict[str, Any]] = []
    for policy, seed in sorted(expected):
        for split in SPLITS:
            split_predictions = predictions[(policy, seed, split)]
            binding = _binding_rows(split_predictions)
            for scope in SCOPES:
                metric_rows.append(
                    _metrics(
                        _scope_rows(binding, scope),
                        split=split,
                        policy=policy,
                        seed=seed,
                        scope=scope,
                    )
                )
            for task in ("elution", "presentation"):
                task_rows = [row for row in split_predictions if row["task"] == task]
                if not task_rows:
                    continue
                synthetic_sources = sorted(
                    {
                        row["source"]
                        for row in task_rows
                        if row["source"].startswith("synthetic_negative")
                    }
                )
                decoy_metric_rows.append(
                    _decoy_metrics(
                        task_rows,
                        split=split,
                        policy=policy,
                        seed=seed,
                        task=task,
                        decoy_family="all_generated_decoys",
                    )
                )
                real_positives = [
                    row
                    for row in task_rows
                    if not row["source"].startswith("synthetic_negative")
                    and float(row["y_true"]) >= 0.5
                ]
                for source in synthetic_sources:
                    selected = real_positives + [
                        row for row in task_rows if row["source"] == source
                    ]
                    decoy_metric_rows.append(
                        _decoy_metrics(
                            selected,
                            split=split,
                            policy=policy,
                            seed=seed,
                            task=task,
                            decoy_family=source,
                        )
                    )

    by_key = {(row["split"], row["policy"], row["seed"], row["scope"]): row for row in metric_rows}
    metric_names = [
        "all_spearman",
        "all_pearson",
        "all_rmse",
        "exact_spearman",
        "exact_pearson",
        "exact_rmse",
        "threshold_500nm_accuracy",
        "threshold_500nm_balanced_accuracy",
        "threshold_500nm_precision",
        "threshold_500nm_recall",
        "threshold_500nm_f1",
        "threshold_500nm_auroc",
        "threshold_500nm_auprc",
    ]
    paired_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for seed in EXPECTED_SEEDS:
            for scope in SCOPES:
                legacy = by_key[(split, "legacy_global_canonical", seed, scope)]
                masked = by_key[(split, "mask_unresolved", seed, scope)]
                for metric in metric_names:
                    if not (_finite(legacy.get(metric)) and _finite(masked.get(metric))):
                        continue
                    paired_rows.append(
                        {
                            "split": split,
                            "seed": seed,
                            "scope": scope,
                            "metric": metric,
                            "legacy": legacy[metric],
                            "masked": masked[metric],
                            "delta_masked_minus_legacy": masked[metric] - legacy[metric],
                        }
                    )

    condition_summary_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for policy in EXPECTED_POLICIES:
            for scope in SCOPES:
                rows = [by_key[(split, policy, seed, scope)] for seed in EXPECTED_SEEDS]
                for metric in ("n_all", "n_exact", "n_threshold_500nm", *metric_names):
                    values = [float(row[metric]) for row in rows if _finite(row.get(metric))]
                    if not values:
                        continue
                    condition_summary_rows.append(
                        {
                            "split": split,
                            "policy": policy,
                            "scope": scope,
                            "metric": metric,
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            "min": min(values),
                            "max": max(values),
                            "n_seeds": len(values),
                        }
                    )

    paired_summary_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for scope in SCOPES:
            for metric in metric_names:
                values = [
                    float(row["delta_masked_minus_legacy"])
                    for row in paired_rows
                    if row["split"] == split and row["scope"] == scope and row["metric"] == metric
                ]
                if not values:
                    continue
                paired_summary_rows.append(
                    {
                        "split": split,
                        "scope": scope,
                        "metric": metric,
                        "mean_delta_masked_minus_legacy": float(np.mean(values)),
                        "std_delta": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                        "min_delta": min(values),
                        "max_delta": max(values),
                        "n_seeds": len(values),
                    }
                )

    def _deltas(scope: str, metric: str) -> list[float]:
        return [
            float(row["delta_masked_minus_legacy"])
            for row in paired_rows
            if row["split"] == "test" and row["scope"] == scope and row["metric"] == metric
        ]

    unresolved_spearman_deltas = _deltas("unresolved_union", "exact_spearman")
    unresolved_rmse_deltas = _deltas("unresolved_union", "exact_rmse")
    overall_spearman_deltas = _deltas("overall", "exact_spearman")
    overall_rmse_deltas = _deltas("overall", "exact_rmse")
    spearman_gate = (
        len(unresolved_spearman_deltas) == len(EXPECTED_SEEDS)
        and all(value > 0.0 for value in unresolved_spearman_deltas)
        and (_mean(unresolved_spearman_deltas) or -math.inf) >= 0.01
    )
    rmse_reductions = [-value for value in unresolved_rmse_deltas]
    rmse_gate = (
        len(rmse_reductions) == len(EXPECTED_SEEDS)
        and all(value > 0.0 for value in rmse_reductions)
        and (_mean(rmse_reductions) or -math.inf) >= 0.02
    )
    # Operationalize "material overall regression" symmetrically with the
    # preregistered unresolved-effect thresholds.
    overall_spearman_delta = _mean(overall_spearman_deltas)
    overall_rmse_delta = _mean(overall_rmse_deltas)
    no_material_overall_regression = (
        overall_spearman_delta is not None
        and overall_rmse_delta is not None
        and overall_spearman_delta >= -0.01
        and overall_rmse_delta <= 0.02
    )
    invest_in_marginalization = bool(
        no_material_overall_regression and (spearman_gate or rmse_gate)
    )
    decision = {
        "primary_scope": "test binding exact-only unresolved_union",
        "unresolved_exact_spearman_delta_masked_minus_legacy_by_seed": dict(
            zip(EXPECTED_SEEDS, unresolved_spearman_deltas)
        ),
        "unresolved_exact_spearman_mean_delta": _mean(unresolved_spearman_deltas),
        "unresolved_exact_rmse_delta_masked_minus_legacy_by_seed": dict(
            zip(EXPECTED_SEEDS, unresolved_rmse_deltas)
        ),
        "unresolved_exact_rmse_mean_reduction": _mean(rmse_reductions),
        "overall_exact_spearman_mean_delta": overall_spearman_delta,
        "overall_exact_rmse_mean_delta": overall_rmse_delta,
        "consistent_spearman_improvement_and_mean_at_least_0.01": spearman_gate,
        "consistent_rmse_reduction_and_mean_at_least_0.02": rmse_gate,
        "no_material_overall_regression": no_material_overall_regression,
        "material_overall_regression_definition": (
            "mean exact Spearman delta < -0.01 or mean exact RMSE delta > +0.02"
        ),
        "invest_in_candidate_marginalization": invest_in_marginalization,
        "recommendation": (
            "invest in candidate-junction marginalization"
            if invest_in_marginalization
            else "keep explicit unknown masking and defer candidate marginalization"
        ),
    }

    _write_csv(results_dir / "condition_metrics.csv", metric_rows)
    _write_csv(results_dir / "elution_decoy_metrics.csv", decoy_metric_rows)
    _write_csv(results_dir / "condition_metric_summary.csv", condition_summary_rows)
    _write_csv(results_dir / "paired_differences.csv", paired_rows)
    _write_csv(results_dir / "paired_difference_summary.csv", paired_summary_rows)
    (results_dir / "paired_differences.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "data_parity.json").write_text(
        json.dumps(parity_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "contract_checks.json").write_text(
        json.dumps(contract_checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(list(run_metadata.values()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
