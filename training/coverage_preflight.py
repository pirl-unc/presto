"""Canonical training preflight for declared output coverage and frozen claims."""

import json
from pathlib import Path

from .data_support import audit_split_support, write_split_support_artifacts
from .output_contract import OutputConfiguration, build_output_contract
from .output_coverage import OutputCoverageCensus, write_output_coverage_artifacts
from .supported_outputs import evaluate_supported_outputs, require_supported_outputs


def audit_training_coverage(
    splits,
    *,
    collator,
    args,
    data_seed: int,
    manifest: dict | None,
    output_dir=None,
):
    """Collate once, persist evidence before gating, and return the legacy audit.

    The caller loads the manifest before data construction. No thresholds are
    inferred here, and a failed gate leaves its complete report for inspection.
    """
    contract = build_output_contract(OutputConfiguration.from_object(args))
    with OutputCoverageCensus(contract) as census:
        legacy = audit_split_support(splits, collator=collator, output_census=census)
        source_contract = {
            "data_source": args.data_source,
            "bulk_ms": bool(getattr(args, "bulk_ms", False)),
            "data_seed": data_seed,
            "split_mode": args.split_mode,
            "split_seed": args.seed,
            "dataset_contract_sha256": legacy["dataset_contract_sha256"],
            "dataset_supervision_contract_sha256": legacy["dataset_supervision_contract_sha256"],
            "output_evidence_sha256": census.sha256,
            "split_rows": {name: value["rows"] for name, value in legacy["splits"].items()},
        }
        coverage = census.report()
        coverage["source_contract"] = source_contract
        output = Path(output_dir) if output_dir is not None else None
        if output is not None:
            write_split_support_artifacts(output, legacy)
            write_output_coverage_artifacts(output, coverage)
        try:
            support = evaluate_supported_outputs(census, manifest, source_contract=source_contract)
        except ValueError as exc:
            if output is not None:
                (output / "supported_outputs.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "requirements_met": False,
                            "status": "invalid_manifest",
                            "error": str(exc),
                            "source_contract": source_contract,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            raise
        if output is not None:
            (output / "supported_outputs.json").write_text(
                json.dumps(support, indent=2, sort_keys=True) + "\n"
            )
        if manifest is not None:
            require_supported_outputs(support)
    met = sum(entry["status"] == "requirements_met" for entry in support["outputs"])
    print(
        f"Output evidence: mode={support['mode']}, declared_requirements_met={met}, "
        f"unsupported={len(support['outputs']) - met}, sha256={coverage['evidence_sha256']}"
    )
    return legacy, support
