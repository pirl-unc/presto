"""Build the frozen unified state-report v2 manifest from committed receipts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

TIMESTAMP_UTC = "2026-11-09T15:00:00Z"
SOURCE_COMMIT = "1cca535e99a20e58c0ce00bc27edcbefbba3ad7d"
RETRAIN_ID = "2026-11-08_1800_codex_unified-parity-retrain"
REPORT_ID = "2026-11-09_1000_codex_unified-state-report-v2"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one(rows: list[dict], **criteria) -> dict:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def metric_row(
    *,
    comparison_id: str,
    cohort_id: str,
    task: str,
    output_id: str,
    estimate_type: str,
    method: str,
    method_output_id: str,
    metric: str,
    value: float,
    ci_low: float | None,
    ci_high: float | None,
    higher_is_better: bool,
    eligible_rows: int,
    studies: int | None,
    primary: bool,
) -> dict:
    return {
        "ci_high": ci_high,
        "ci_low": ci_low,
        "cohort_id": cohort_id,
        "comparison_id": comparison_id,
        "eligible_rows": eligible_rows,
        "estimate_type": estimate_type,
        "higher_is_better": higher_is_better,
        "interval_kind": "bootstrap_95" if ci_low is not None else "not_available",
        "method": method,
        "method_output_id": method_output_id,
        "metric": metric,
        "output_id": output_id,
        "primary": primary,
        "split": "development",
        "studies": studies,
        "task": task,
        "value": value,
    }


def affinity_metrics(bootstrap: list[dict], pooled: list[dict]) -> list[dict]:
    result = []
    config = {
        "I": {
            "comparison_id": "mhc_i_affinity_equal_study",
            "cohort_id": "affinity_exact_ic50_mhc_i_development",
            "task": "mhc_i_affinity",
            "rows": 169,
            "comparator": "NetMHCpan 4.2c",
        },
        "II": {
            "comparison_id": "mhc_ii_affinity_equal_study",
            "cohort_id": "affinity_exact_ic50_mhc_ii_development",
            "task": "mhc_ii_affinity",
            "rows": 1170,
            "comparator": "NetMHCIIpan 4.3e",
        },
    }
    method_ids = {
        "Presto reference": ("Presto_reference", "assay_aware_ic50"),
        "Presto parity": ("Presto_parity_experimental", "assay_aware_ic50"),
        "NetMHCpan 4.2c": ("NetMHCpan_4.2c", "archive_ba_ic50"),
        "NetMHCIIpan 4.3e": ("NetMHCIIpan_4.3e", "archive_ba_ic50"),
    }
    for mhc_class, fields in config.items():
        methods = ["Presto reference", "Presto parity", fields["comparator"]]
        for metric in ("spearman", "pearson", "rmse"):
            for method in methods:
                row = one(
                    bootstrap,
                    comparison="method_value",
                    task="affinity",
                    mhc_class=mhc_class,
                    metric=metric,
                    method=method,
                )
                method_id, method_output = method_ids[method]
                result.append(
                    metric_row(
                        comparison_id=fields["comparison_id"],
                        cohort_id=fields["cohort_id"],
                        task=fields["task"],
                        output_id="exact_ic50_log10_nm",
                        estimate_type="method_value",
                        method=method_id,
                        method_output_id=method_output,
                        metric=metric,
                        value=row["estimate"],
                        ci_low=row["ci_low"],
                        ci_high=row["ci_high"],
                        higher_is_better=metric != "rmse",
                        eligible_rows=fields["rows"],
                        studies=row["studies"],
                        primary=metric == "spearman",
                    )
                )
            for second in ("Presto reference", fields["comparator"]):
                row = one(
                    bootstrap,
                    comparison="paired_difference",
                    task="affinity",
                    mhc_class=mhc_class,
                    metric=metric,
                    first="Presto parity",
                    second=second,
                )
                second_id = method_ids[second][0]
                result.append(
                    metric_row(
                        comparison_id=fields["comparison_id"],
                        cohort_id=fields["cohort_id"],
                        task=fields["task"],
                        output_id="exact_ic50_log10_nm",
                        estimate_type="paired_difference",
                        method=f"Presto_parity_minus_{second_id}",
                        method_output_id="assay_aware_ic50_difference",
                        metric=f"{metric}_difference",
                        value=row["estimate"],
                        ci_low=row["ci_low"],
                        ci_high=row["ci_high"],
                        higher_is_better=metric != "rmse",
                        eligible_rows=fields["rows"],
                        studies=row["studies"],
                        primary=metric == "spearman",
                    )
                )

        pooled_comparison = f"{fields['comparison_id']}_pooled_secondary"
        for method in methods:
            row = one(
                pooled,
                task="affinity",
                population="exact",
                assay_method="all_ic50",
                mhc_class=mhc_class,
                metric="spearman",
                method=method,
            )
            method_id, method_output = method_ids[method]
            result.append(
                metric_row(
                    comparison_id=pooled_comparison,
                    cohort_id=fields["cohort_id"],
                    task=fields["task"],
                    output_id="exact_ic50_log10_nm",
                    estimate_type="method_value",
                    method=method_id,
                    method_output_id=method_output,
                    metric="pooled_spearman",
                    value=row["value"],
                    ci_low=None,
                    ci_high=None,
                    higher_is_better=True,
                    eligible_rows=row["n"],
                    studies=None,
                    primary=False,
                )
            )
    return result


def binary_metrics(bootstrap: list[dict]) -> list[dict]:
    result = []
    tasks = {
        "presentation": {
            "comparison_id": "presentation_equal_study",
            "cohort_id": "presentation_matched_48530",
            "task": "presentation",
            "output_id": "presentation",
            "external": "MHCflurry 2.3.16 presentation",
            "external_id": "MHCflurry_2.3.16_presentation",
            "external_output": "presentation_score",
        },
        "operational_processing": {
            "comparison_id": "processing_equal_study",
            "cohort_id": "processing_matched_48530",
            "task": "processing",
            "output_id": "operational_processing",
            "external": "MHCflurry 2.3.16 processing",
            "external_id": "MHCflurry_2.3.16_processing",
            "external_output": "processing_AP",
        },
    }
    for source_task, fields in tasks.items():
        methods = {
            "Presto reference": ("Presto_reference", fields["output_id"]),
            "Presto parity": ("Presto_parity_experimental", fields["output_id"]),
            "Presto Stage B": ("Presto_Stage_B", fields["output_id"]),
            fields["external"]: (fields["external_id"], fields["external_output"]),
        }
        for metric in ("auroc", "auprc"):
            for method, (method_id, method_output) in methods.items():
                row = one(
                    bootstrap,
                    comparison="method_value",
                    task=source_task,
                    metric=metric,
                    method=method,
                )
                result.append(
                    metric_row(
                        comparison_id=fields["comparison_id"],
                        cohort_id=fields["cohort_id"],
                        task=fields["task"],
                        output_id=fields["output_id"],
                        estimate_type="method_value",
                        method=method_id,
                        method_output_id=method_output,
                        metric=metric,
                        value=row["estimate"],
                        ci_low=row["ci_low"],
                        ci_high=row["ci_high"],
                        higher_is_better=True,
                        eligible_rows=48530,
                        studies=row["studies"],
                        primary=metric == "auroc",
                    )
                )
            for second in ("Presto reference", fields["external"]):
                row = one(
                    bootstrap,
                    comparison="paired_difference",
                    task=source_task,
                    metric=metric,
                    first="Presto parity",
                    second=second,
                )
                second_id = methods[second][0]
                result.append(
                    metric_row(
                        comparison_id=fields["comparison_id"],
                        cohort_id=fields["cohort_id"],
                        task=fields["task"],
                        output_id=fields["output_id"],
                        estimate_type="paired_difference",
                        method=f"Presto_parity_minus_{second_id}",
                        method_output_id=f"{fields['output_id']}_difference",
                        metric=f"{metric}_difference",
                        value=row["estimate"],
                        ci_low=row["ci_low"],
                        ci_high=row["ci_high"],
                        higher_is_better=True,
                        eligible_rows=48530,
                        studies=row["studies"],
                        primary=metric == "auroc",
                    )
                )
    return result


def supplemental_artifacts(receipt: dict, retrain_dir: Path) -> list[dict]:
    result = []
    for relative, record in sorted(receipt["files"].items()):
        if not relative.startswith(("figures/", "tables/")):
            continue
        local = retrain_dir / "results" / "evaluation" / relative
        if local.stat().st_size != record["bytes"] or sha256(local) != record["sha256"]:
            raise ValueError(f"supplemental artifact mismatch: {relative}")
        result.append(
            {
                "bytes": record["bytes"],
                "id": f"evaluation_{relative.replace('/', '_').replace('.', '_')}",
                "include_in_report": True,
                "kind": "figure" if relative.startswith("figures/") else "table",
                "license": "Derived development evaluation; underlying source-data terms apply",
                "local_path": str(
                    Path("..") / RETRAIN_ID / "results" / "evaluation" / relative
                ),
                "sha256": record["sha256"],
                "uri": (
                    "modal-volume://presto-checkpoints/"
                    f"{RETRAIN_ID}/evaluation-final/{relative}"
                ),
            }
        )
    for filename in ("checkpoint-summary.json", "decision.json"):
        local = retrain_dir / "results" / filename
        result.append(
            {
                "bytes": local.stat().st_size,
                "id": f"evaluation_{filename.replace('.', '_')}",
                "include_in_report": True,
                "kind": "table",
                "license": "Derived Presto experiment metadata",
                "local_path": str(Path("..") / RETRAIN_ID / "results" / filename),
                "sha256": sha256(local),
                "uri": f"git://presto/{RETRAIN_ID}/results/{filename}",
            }
        )
    return result


def build_manifest(report_dir: Path) -> dict:
    repo = report_dir.parent.parent
    retrain_dir = repo / "experiments" / RETRAIN_ID
    old_report = read_json(
        repo
        / "experiments"
        / "2026-11-08_2210_codex_unified-state-report-v1"
        / "report_manifest.json"
    )
    evaluation_receipt = read_json(retrain_dir / "results/evaluation/receipt.json")
    bootstrap = read_json(
        retrain_dir / "results/evaluation/tables/study-bootstrap.json"
    )
    pooled = read_json(retrain_dir / "results/evaluation/tables/metrics.json")

    datasets = deepcopy(old_report["data"]["datasets"][:4])
    datasets.extend(
        [
            {
                "artifact": {
                    "bytes": 141870035,
                    "include_in_report": False,
                    "license": "DTU academic archive; derived TRAIN projection is not redistributed",
                    "sha256": "2c007e6b46a99835f95ec612aeb498f4e52940aa81c22809ad48b5e922d6d0ca",
                    "uri": "modal-volume://presto-checkpoints/2026-11-08_1740_codex_netmhc-ba-parity-projection/prepared-v2/netmhcpan42-ba-train.jsonl",
                },
                "curation_contract": "Archive-faithful literal censoring, canonical measurement lineage, and unspecified-method IC50 routing; used only by rejected parity arm",
                "id": "experimental_parity_netmhcpan42_ba_train_v2",
                "role": "train",
                "rows": 80674,
                "source": "NetMHCpan 4.2 BA training archive projection",
                "split_contract": "Exact/core9 affinity and exact/core8/Hamming-1 presentation-development embargoes",
            },
            {
                "artifact": {
                    "bytes": 112640875,
                    "include_in_report": False,
                    "license": "DTU academic archive; derived TRAIN projection is not redistributed",
                    "sha256": "9fe752274c895cb4cd46e36dfeff43a3db73e0c96fa24ba495e1fcfd7f24c66a",
                    "uri": "modal-volume://presto-checkpoints/2026-11-08_1740_codex_netmhc-ba-parity-projection/prepared-v2/netmhciipan43-ba-train.jsonl",
                },
                "curation_contract": "Archive-faithful literal censoring, canonical measurement lineage, and unspecified-method IC50 routing; used only by rejected parity arm",
                "id": "experimental_parity_netmhciipan43_ba_train_v2",
                "role": "train",
                "rows": 62539,
                "source": "NetMHCIIpan 4.3 BA training archive projection",
                "split_contract": "Exact/core9 affinity and exact/core8/Hamming-1 presentation-development embargoes",
            },
        ]
    )
    for dataset in deepcopy(old_report["data"]["datasets"][4:]):
        dataset["id"] = f"historical_{dataset['id']}"
        dataset["source"] = f"Historical unpromoted T-cell arm; {dataset['source']}"
        dataset["curation_contract"] += (
            "; not used to fit the retained reference checkpoint"
        )
        datasets.append(dataset)

    comparators = deepcopy(old_report["comparators"][:3])
    comparators.append(
        {
            "id": "prime",
            "receipt": {
                "bytes": 3785,
                "include_in_report": False,
                "license": "PRIME 2.1 and MixMHCpred are academic/non-commercial",
                "sha256": "f3e9411b81a7c55b3bf5b99395957d18da5b3820b5a8bb82ded3ec4ef65aad59",
                "uri": "git://presto/experiments/2026-11-08_2225_codex_prime-mhctools-smoke/results/smoke.json",
            },
            "version": "2.1_mixmhcpred-3.0_smoke-passed_benchmark-pending",
        }
    )

    prediction_artifacts = [
        {
            "bytes": 24362809,
            "id": "reference_development_predictions",
            "include_in_report": False,
            "license": "Derived predictions; underlying source-data terms apply",
            "sha256": "81d0bfbcd6432c93ba261461b0ffc01d4978a6ece94b8405133d34cb6f5db3d2",
            "split": "development",
            "uri": f"modal-volume://presto-checkpoints/{RETRAIN_ID}/fits-v3/reference/validation-predictions.jsonl.gz",
        },
        {
            "bytes": 24301300,
            "id": "parity_development_predictions",
            "include_in_report": False,
            "license": "Derived predictions; underlying source-data terms apply",
            "sha256": "d85c2987e1eceb5487dc6e04c9e25dc9eee5bfaea87a756fa86f1bf87fa8a278",
            "split": "development",
            "uri": f"modal-volume://presto-checkpoints/{RETRAIN_ID}/fits-v3/parity/validation-predictions.jsonl.gz",
        },
        deepcopy(old_report["prediction_artifacts"][0]),
        deepcopy(old_report["prediction_artifacts"][1]),
        deepcopy(old_report["prediction_artifacts"][2]),
        {
            "bytes": 229009,
            "id": "matched_affinity_long_predictions",
            "include_in_report": False,
            "license": "Derived Presto/NetMHC matched predictions; source and DTU terms apply",
            "sha256": "64e5b859252e98cbf803da9c17601518506d7b2420eb6c951ba18fe3beedcfa4",
            "split": "development",
            "uri": f"modal-volume://presto-checkpoints/{RETRAIN_ID}/evaluation-final/predictions/matched-affinity.jsonl.gz",
        },
        {
            "bytes": 7581339,
            "id": "matched_presentation_processing_long_predictions",
            "include_in_report": False,
            "license": "Derived Presto/MHCflurry matched predictions; source-data terms apply",
            "sha256": "f26cfefe6b33c35b44a0e4a1f89d65c6f43a7bffe3c54d398997792993836540",
            "split": "development",
            "uri": f"modal-volume://presto-checkpoints/{RETRAIN_ID}/evaluation-final/predictions/matched-presentation-processing.jsonl.gz",
        },
        deepcopy(old_report["prediction_artifacts"][3]),
    ]

    metrics = affinity_metrics(bootstrap, pooled) + binary_metrics(bootstrap)
    for row in old_report["metrics"]:
        if row["task"] == "tcell_immunogenicity":
            historical = deepcopy(row)
            historical["comparison_id"] = f"historical_{row['comparison_id']}"
            historical["primary"] = False
            metrics.append(historical)

    return {
        "comparators": comparators,
        "data": {
            "datasets": datasets,
            "overlap_audits": [
                *deepcopy(old_report["data"]["overlap_audits"]),
                {
                    "contract": "exact/core9 affinity plus exact/core8/Hamming-1 presentation embargo; unique measurement and observation identity",
                    "result": "passed",
                    "scope": "experimental BA v2 TRAIN versus complete Stage B development union",
                },
                {
                    "contract": "same matched development rows and biological inputs for every method",
                    "result": "passed",
                    "scope": "final affinity/presentation/processing evaluation",
                },
            ],
            "test_status": {
                "available": False,
                "reason": "All current headline populations are exposed development cohorts. TEST and locked confirmation access were false in both fits and the final evaluator.",
            },
        },
        "git": {"commit": SOURCE_COMMIT, "dirty": False},
        "metrics": metrics,
        "model": {
            "checkpoint": {
                "bytes": 8270787,
                "include_in_report": False,
                "license": "Derived Presto checkpoint; source-data terms still apply",
                "sha256": "1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a",
                "uri": f"modal-volume://presto-checkpoints/{RETRAIN_ID}/fits-v3/reference/raw/selected.pt",
            },
            "id": "unified_reference_2026_11_08",
            "training_recipe": {
                "assay_output_mapping": "EC50, IC50, KD-via-EC50, KD-via-IC50, presentation, and processing remain distinct outputs",
                "model_code_commit": "0211e722",
                "parent_checkpoint_sha256": "e6cc6fbb9950ace32efe418bfa96b119e1b94f0399e6843db0a16b0208937c06",
                "reference": "5,000 updates, batch 128, 50:50 EL/concentration replay, seed 42, Stage B development selection",
                "selection": "Retain reference after experimental parity arm failed preregistered MHC-II parent gate",
                "tcell": "No T-cell extension is present in this selected checkpoint; prior assay-aware T-cell arms remain unpromoted context",
            },
        },
        "prediction_artifacts": prediction_artifacts,
        "promotion_gates": [
            {
                "id": "data_only_parent",
                "reason": "Parity MHC-II pooled exact-IC50 Spearman improved only +0.00636 versus required +0.05, and equal-study delta was -0.03648 with CI crossing zero.",
                "status": "fail",
            },
            {
                "id": "mhc_i_affinity_vs_netmhcpan",
                "reason": "Selected reference equal-study Spearman is 0.58501 versus NetMHCpan 4.2c at 0.68338; pooled values are 0.66971 versus 0.75845.",
                "status": "fail",
            },
            {
                "id": "mhc_ii_affinity_vs_netmhciipan",
                "reason": "Selected reference equal-study Spearman is 0.57647 versus NetMHCIIpan 4.3e at 0.80702; pooled values are 0.49965 versus 0.77786.",
                "status": "fail",
            },
            {
                "id": "presentation_vs_mhcflurry",
                "reason": "Selected reference AUROC/AUPRC is 0.95203/0.95265 versus MHCflurry 0.94843/0.95428; the parity paired intervals cross zero, so this remains similarity rather than superiority.",
                "status": "fail",
            },
            {
                "id": "processing_vs_mhcflurry",
                "reason": "Selected reference AUROC is 0.90576 versus MHCflurry 0.80567; parity's paired delta is +0.10114 with 95% CI [+0.07829,+0.14520], and parity-reference preservation interval crosses zero near zero.",
                "status": "pass",
            },
            {
                "id": "assay_aware_output_preservation",
                "reason": "Final evaluation preserves all four physical-affinity endpoint identities plus separate presentation and operational-processing outputs.",
                "status": "pass",
            },
            {
                "id": "prime_software_integration",
                "reason": "PRIME 2.1 ran through mhctools with the licensed MixMHCpred 3.0 backend in a two-peptide smoke test.",
                "status": "pass",
            },
            {
                "id": "tcell_vs_prime",
                "reason": "No newly disjoint T-cell accuracy cohort has been scored by the selected checkpoint or PRIME; the smoke is not an accuracy benchmark.",
                "status": "not_evaluated",
            },
            {
                "id": "independent_all_task_test",
                "reason": "No untouched all-task TEST population has been scored by a frozen unified checkpoint.",
                "status": "not_evaluated",
            },
        ],
        "runtime": {
            "evaluation_elapsed_seconds": 60.54146325000329,
            "evaluation_peak_rss_bytes": 8478867456,
            "model_size_bytes": 8270787,
            "observed_gpu": "NVIDIA H100 80GB HBM3 for both matched fits",
            "parity_training_elapsed_seconds": 9171.690,
            "reference_training_elapsed_seconds": 9177.218,
            "report_generation": "local CPU",
            "requested_gpu": "H100!",
        },
        "schema_version": 1,
        "state": {
            "blocked_claims": [
                "The selected reference checkpoint does not beat NetMHCpan on matched MHC-I development affinity.",
                "The selected reference checkpoint remains substantially behind NetMHCIIpan on matched MHC-II development affinity.",
                "Presentation is similar on three exposed studies but does not support a superiority claim.",
                "No newly disjoint T-cell/PRIME accuracy comparison or independent all-task TEST exists.",
            ],
            "headline": "The retained assay-aware reference checkpoint preserves decisive processing performance, remains similar for presentation, and rejects a comparator-derived BA expansion that improved MHC-I but failed to improve MHC-II materially.",
            "limitations": [
                "All headline metrics are development-only; no current cohort is relabeled as TEST.",
                "Affinity equal-study estimates contain only two studies, and presentation/processing contain only three.",
                "The experimental BA rows are comparator training archives and cannot establish independent superiority over those tools.",
                "The selected reference checkpoint does not contain a promoted T-cell extension; historical T-cell results are context only.",
                "NetMHC comparator versions are 4.2c/4.3e rather than newer patches, and MHCflurry lineage is only partially reconstructable.",
            ],
            "qualified_claims": [
                "Both matched H100 fits completed 5,000 updates with passed coverage and zero TEST/locked access.",
                "Operational-processing development AUROC is 0.90576 for the retained reference versus 0.80567 for MHCflurry; the near-identical parity arm has a paired +0.10114 interval wholly above zero.",
                "Presentation AUROC/AUPRC is 0.95203/0.95265 for the retained reference versus 0.94843/0.95428 for MHCflurry, which supports similarity rather than superiority.",
                "The parity arm improved MHC-I exact-IC50 Spearman but failed the preregistered MHC-II parent gate, so the reference checkpoint is retained.",
                "The evaluator preserves assay-aware physical endpoints, and independent replay matches 110 estimates to maximum absolute delta 3.33e-16.",
                "PRIME 2.1 through mhctools/MixMHCpred 3.0 is operational, but accuracy remains unevaluated.",
            ],
            "status": "development",
        },
        "supplemental_artifacts": supplemental_artifacts(
            evaluation_receipt, retrain_dir
        ),
        "timestamp_utc": TIMESTAMP_UTC,
    }


def main() -> int:
    report_dir = Path(__file__).resolve().parent.parent
    manifest = build_manifest(report_dir)
    output = report_dir / "report_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
