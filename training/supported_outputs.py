"""Prospective, explicitly scoped evidence requirements for training claims."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .output_coverage import COUNT_FIELDS, OutputCoverageCensus


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_supported_output_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Supported-output manifest must be a JSON object")
    return manifest


def evaluate_supported_outputs(
    census: OutputCoverageCensus, manifest: dict | None, *, source_contract: dict
) -> dict:
    """Validate a frozen declaration; never choose support thresholds from counts.

    An exploratory run without a manifest reports every output as undeclared.
    Passing means that the declared evidence requirements were met. It does not
    establish prediction accuracy or turn proxy evidence into direct evidence.
    """
    contract = census.contract
    cells = {
        (endpoint, column)
        for endpoint, spec in contract.outputs.items()
        for column in (spec.columns or ("",))
    }
    decisions = {
        cell: {
            "endpoint": cell[0],
            "column": cell[1],
            "status": "undeclared",
            "reasons": ["No prospective evidence requirement declared."],
        }
        for cell in sorted(cells)
    }
    result = {
        "schema_version": 1,
        "mode": "exploratory" if manifest is None else "declared",
        "evidence_sha256": census.sha256,
        "source_contract": source_contract,
        "manifest_sha256": None,
        "requirements_met": False,
        "outputs": [],
    }
    if manifest is None:
        result["outputs"] = list(decisions.values())
        return result
    required = {"schema_version", "model_contract", "source_contract", "claims"}
    if set(manifest) != required or manifest["schema_version"] != 1:
        raise ValueError(
            "Manifest requires schema_version=1, model_contract, source_contract and claims"
        )
    if _canonical_json(manifest["model_contract"]) != _canonical_json(
        asdict(contract.configuration)
    ):
        raise ValueError("Manifest model_contract differs from the declared model configuration")
    if _canonical_json(manifest["source_contract"]) != _canonical_json(source_contract):
        raise ValueError("Manifest source_contract differs from the curated data contract")
    if not isinstance(manifest["claims"], list) or not manifest["claims"]:
        raise ValueError("A supported-output manifest must declare at least one claim")
    result["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    seen = set()
    for claim in manifest["claims"]:
        fields = {
            "endpoint",
            "columns",
            "required_splits",
            "evidence_roles",
            "source_families",
            "raw_sources",
            "minimums",
        }
        if not isinstance(claim, dict) or set(claim) != fields:
            raise ValueError(f"Each claim requires exactly {sorted(fields)}")
        endpoint = claim["endpoint"]
        if endpoint in contract.aliases:
            raise ValueError(
                f"{endpoint} is an alias of {contract.canonical(endpoint)}; "
                "aliases cannot be independent claims"
            )
        if endpoint not in contract.outputs:
            raise ValueError(f"Unknown canonical endpoint {endpoint}")
        if not any(obj.endpoint == endpoint for obj in contract.objectives.values()):
            raise ValueError(
                f"{endpoint} has no label objective; "
                "indirect gradient cannot satisfy a supervision claim"
            )
        for field in (
            "columns",
            "required_splits",
            "evidence_roles",
            "source_families",
            "raw_sources",
        ):
            values = claim[field]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(x, str) for x in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"{endpoint}: {field} must be an explicit, nonempty list of unique strings"
                )
        if not set(claim["evidence_roles"]) <= {"direct", "proxy", "auxiliary", "synthetic"}:
            raise ValueError(f"{endpoint}: unknown provenance cannot satisfy evidence requirements")
        if any(x == "*" or not x for x in claim["source_families"] + claim["raw_sources"]):
            raise ValueError(f"{endpoint}: source families and raw sources must be explicit")
        minimums = claim["minimums"]
        if (
            not isinstance(minimums, dict)
            or not {"unique_observations", "distinct_peptides"} <= minimums.keys()
        ):
            raise ValueError(
                f"{endpoint}: explicit unique_observations and distinct_peptides "
                "minimums are required"
            )
        if set(minimums) - set(COUNT_FIELDS) or any(
            type(x) is not int or x < 0 for x in minimums.values()
        ):
            raise ValueError(f"{endpoint}: minimums must be known nonnegative integer counts")
        if minimums["unique_observations"] < 1 or minimums["distinct_peptides"] < 1:
            raise ValueError(
                f"{endpoint}: a declared claim requires positive observation and peptide minimums"
            )
        loss_types = {
            obj.loss_type for obj in contract.objectives.values() if obj.endpoint == endpoint
        }
        if (
            loss_types & {"bce", "ce"}
            and not {"unique_positive", "unique_negative"} <= minimums.keys()
        ):
            raise ValueError(
                f"{endpoint}: explicit unique_positive and unique_negative minimums are required"
            )
        if loss_types & {"censor", "mse"} and "unique_exact" not in minimums:
            raise ValueError(f"{endpoint}: an explicit exact-value minimum is required")
        for column in claim["columns"]:
            cell = endpoint, column
            if cell not in cells:
                raise ValueError(
                    f"{endpoint}: unknown column {column!r}; panel claims must select named columns"
                )
            if cell in seen:
                raise ValueError(f"Duplicate claim for {endpoint}:{column}")
            seen.add(cell)
            reasons, splits = [], {}
            for split in claim["required_splits"]:
                if split not in census.splits:
                    reasons.append(f"Required split {split} was not supplied")
                    continue
                counts = census.counts(
                    split,
                    endpoint,
                    column,
                    roles=claim["evidence_roles"],
                    families=claim["source_families"],
                    sources=claim["raw_sources"],
                )
                splits[split] = counts
                for key, minimum in minimums.items():
                    if counts[key] < minimum:
                        reasons.append(f"{split}:{key}={counts[key]} requires >= {minimum}")
            decisions[cell] = {
                "endpoint": endpoint,
                "column": column,
                "status": "requirements_not_met" if reasons else "requirements_met",
                "reasons": reasons,
                "claim": claim,
                "splits": splits,
            }
    result["requirements_met"] = all(
        decisions[cell]["status"] == "requirements_met" for cell in seen
    )
    result["outputs"] = list(decisions.values())
    return result


def require_supported_outputs(report: dict) -> None:
    if not report["requirements_met"]:
        reasons = [
            f"{entry['endpoint']}:{entry['column']}: {reason}"
            for entry in report["outputs"]
            if entry["status"] == "requirements_not_met"
            for reason in entry["reasons"]
        ]
        if not reasons:
            reasons = ["A prospective supported-output manifest is required."]
        raise RuntimeError("Supported-output preflight failed:\n- " + "\n- ".join(reasons))
