"""Frozen observation membership and auditable generated descendants."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .frozen_hitlist import FrozenHitlistBundle, _sha256
from presto import __version__

PARTITIONS = ("train", "val", "test")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _identities(value):
    values = json.loads(value or "[]") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) for item in values):
        raise ValueError("Lineage identities must be JSON arrays of strings")
    return values


def _components(identities, contributors, registry, policy):
    """Join actual reuse, shared sequence and requested reviewed identity axes."""
    from hitlist.lineage import canonical_identities
    from hitlist.split_audit import POLICIES

    if policy not in POLICIES:
        raise ValueError(f"Unknown frozen partition policy: {policy}")
    ids = identities.evidence_row_id.tolist()
    parent = list(range(len(ids)))
    sizes = [1] * len(ids)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}

    def join(index, axis, value):
        if not value:
            return
        key = (axis, value)
        other = seen.setdefault(key, index)
        a, b = root(index), root(other)
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]

    canonical = canonical_identities(registry)
    axes = {"experimental_origin", "acquisition"}
    axes.update(axis for axis in POLICIES[policy] if axis in {"specimen", "donor"})
    by_provenance = defaultdict(list)
    for index, row in enumerate(identities.to_dict("records")):
        join(index, "peptide", row["peptide"])
        # IEDB/CEDAR mirrors must not create independent copies of an assay.
        join(
            index,
            "observation",
            re.sub(
                r"(?<=:)https?://(?:www\.iedb\.org|iedb\.org|cedar\.iedb\.org)(?=/assay/)",
                "",
                row["evidence_row_id"],
            ),
        )
        by_provenance[row["provenance_id"]].append(index)
        for axis in axes:
            for identity in _identities(row.get(f"{axis}_ids", "[]")):
                join(index, axis, canonical[identity])
        if policy == "whole_study":
            for paper in _identities(row.get("contributor_pmids", "[]")):
                join(index, "study", paper)
            if pd.notna(row.get("pmid")) and str(row["pmid"]).strip():
                join(index, "study", str(int(row["pmid"])))
    for row in contributors.itertuples():
        if not row.source_record_id or not row.source_dataset:
            continue
        for index in by_provenance.get(row.provenance_id, []):
            join(index, "source_record", (row.source_dataset, row.source_record_id))
    members = defaultdict(list)
    for index, identity in enumerate(ids):
        members[root(index)].append(identity)
    return [sorted(values) for values in members.values()]


def _assign_components(components, *, seed, val_frac, test_frac, allocation="absolute_deficit_v1"):
    if allocation not in {"absolute_deficit_v1", "relative_deficit_v2"}:
        raise ValueError(f"Unknown component allocation policy: {allocation}")
    if (
        not all(math.isfinite(value) for value in (val_frac, test_frac))
        or val_frac <= 0
        or test_frac < 0
        or val_frac + test_frac >= 1
    ):
        raise ValueError(
            "Frozen partitions require validation >0, test >=0 and a positive train fraction"
        )
    fractions = {"train": 1 - val_frac - test_frac, "val": val_frac}
    if test_frac:
        fractions["test"] = test_frac
    if len(components) < len(fractions):
        raise ValueError(
            "Too few independent grouping components for the requested nonempty partitions"
        )
    total = sum(map(len, components))
    targets = {name: total * value for name, value in fractions.items()}
    counts = Counter()
    assignments = {}
    ordered = sorted(components, key=lambda group: (-len(group), _digest([seed, group])))
    for index, group in enumerate(ordered):
        empty = [name for name in fractions if not counts[name]]
        eligible = empty if len(ordered) - index == len(empty) else list(fractions)

        def priority(name):
            deficit = targets[name] - counts[name]
            if allocation == "relative_deficit_v2":
                deficit /= targets[name]
            return deficit, -PARTITIONS.index(name)

        chosen = max(eligible, key=priority)
        component_id = "component:" + _digest(group)
        for identity in group:
            assignments[identity] = (chosen, component_id)
        counts[chosen] += len(group)
    return assignments


@dataclass
class FrozenPartitions:
    bundle: FrozenHitlistBundle
    assignments: pd.DataFrame
    identities: pd.DataFrame
    registry: dict
    policy: str
    paths: dict[str, str]
    receipt: dict

    def apply(self, record_groups):
        membership = self.assignments.set_index("evidence_row_id").source_partition.to_dict()
        for records in record_groups.values():
            for record in records:
                if record.generated_id:
                    raise ValueError(
                        "Measured membership must be assigned before control generation"
                    )
                if record.source_bundle_id != self.bundle.bundle_id:
                    raise ValueError("Selected record has a different frozen bundle identity")
                record.source_partition = membership[record.evidence_row_id]


def prepare_frozen_partitions(
    bundle,
    record_groups,
    *,
    seed,
    val_frac,
    test_frac,
    policy="peptide_disjoint",
    component_allocation="absolute_deficit_v1",
    assignment_file=None,
    assignment_sha256=None,
    cache_dir=None,
):
    """Select and publish membership after source/MHC filtering, before controls."""
    import datacache
    from hitlist import audit_splits
    from hitlist.downloads import data_dir

    selected = sorted(
        {record.evidence_row_id for rows in record_groups.values() for record in rows}
    )
    if not selected or "" in selected:
        raise ValueError("Frozen partitioning requires every selected observation ID")
    source_path = bundle.directory / bundle.manifest["artifacts"]["identities"]["file"]
    identities = pd.read_parquet(source_path, filters=[("evidence_row_id", "in", selected)])
    if identities.evidence_row_id.duplicated().any() or set(identities.evidence_row_id) != set(
        selected
    ):
        raise ValueError(
            "Selected observation identities are missing or duplicated in the frozen bundle"
        )
    identities = identities.sort_values("evidence_row_id").reset_index(drop=True)
    expected_peptides = identities.set_index("evidence_row_id").peptide.to_dict()
    for records in record_groups.values():
        for record in records:
            if (
                record.generated_id
                or record.source_bundle_id != bundle.bundle_id
                or record.peptide != expected_peptides[record.evidence_row_id]
            ):
                raise ValueError("Selected record does not match its frozen observation identity")
    contributors = pd.read_parquet(
        bundle.directory / bundle.manifest["artifacts"]["contributors"]["file"],
        columns=["provenance_id", "source_record_id", "source_dataset"],
        filters=[("provenance_id", "in", identities.provenance_id.unique().tolist())],
    )
    registry = bundle.lineage_registry()
    components = _components(identities, contributors, registry, policy)
    assigned = _assign_components(
        components,
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
        allocation=component_allocation,
    )
    external = None
    if bool(assignment_file) != bool(assignment_sha256):
        raise ValueError("External assignments require both a file and its SHA-256")
    if assignment_file:
        datacache.validate_file(assignment_file, assignment_sha256)
        external = pd.read_parquet(assignment_file)
        required = {"evidence_row_id", "source_bundle_id", "source_partition"}
        if (
            not required <= set(external)
            or external.evidence_row_id.duplicated().any()
            or not external.source_bundle_id.eq(bundle.bundle_id).all()
            or not external.source_partition.isin(PARTITIONS).all()
        ):
            raise ValueError("Invalid or conflicting external partition assignments")
        indexed = external.set_index("evidence_row_id")
        if not set(selected) <= set(indexed.index):
            raise ValueError("External assignments omit selected observations")
        for group in components:
            membership = set(indexed.loc[group, "source_partition"])
            if len(membership) != 1:
                raise ValueError("External assignments split reused source or lineage components")
            chosen = next(iter(membership))
            for identity in group:
                assigned[identity] = (chosen, assigned[identity][1])
    rows = [
        {
            "evidence_row_id": identity,
            "source_partition": assigned[identity][0],
            "component_id": assigned[identity][1],
            "source_bundle_id": bundle.bundle_id,
        }
        for identity in selected
    ]
    assignments = pd.DataFrame(rows).merge(
        identities[["evidence_row_id", "peptide"]], on="evidence_row_id", validate="one_to_one"
    )
    wanted = {"train", "val", "test"} if test_frac else {"train", "val"}
    if set(assignments.source_partition) != wanted:
        raise ValueError("Frozen membership does not form every requested nonempty partition")
    partitions = {
        name: identities[
            identities.evidence_row_id.isin(
                assignments.loc[assignments.source_partition.eq(name), "evidence_row_id"]
            )
        ]
        for name in sorted(wanted)
    }
    audit = audit_splits(partitions, policy=policy, registry=registry)
    contract = {
        "algorithm": (
            "source-lineage-components-v1"
            if component_allocation == "absolute_deficit_v1"
            else "source-lineage-components-v2"
        ),
        "component_allocation": component_allocation,
        "assignment_origin": "external" if assignment_file else "computed",
        "presto_version": __version__,
        "source_bundle_id": bundle.bundle_id,
        "policy": policy,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "selection_sha256": _digest(selected),
        "external_assignment_sha256": assignment_sha256,
        "recipe_sha256": _sha256(Path(__file__)),
    }
    root = Path(cache_dir) if cache_dir is not None else data_dir() / "presto-partitions"
    sources = {
        "source-manifest.json": {
            "path": str(bundle.directory / "manifest.json"),
            "sha256": bundle.manifest_sha256,
        }
    }
    if assignment_file:
        sources["external.parquet"] = {
            "path": str(Path(assignment_file).resolve()),
            "sha256": assignment_sha256,
        }
    coverage = {
        "n_selected_observations": len(selected),
        "n_components": len(components),
        "n_largest_component_observations": max(map(len, components)),
        "n_external_unselected_observations": 0
        if external is None
        else len(external) - len(selected),
        "partition_counts": assignments.source_partition.value_counts().to_dict(),
    }

    def builder(inputs, outputs):
        assignments.to_parquet(outputs["assignments.parquet"], index=False)
        assignments.to_csv(outputs["assignments.tsv"], sep="\t", index=False)
        for name, data in {
            "policy.json": contract,
            "coverage.json": coverage,
            "lineage_audit.json": audit,
        }.items():
            Path(outputs[name]).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
        Path(outputs["README.md"]).write_text(
            "# Frozen measured-observation partitions\n\n"
            "assignments.tsv/parquet give actual selected observation membership. "
            "Repeated sequences, source records and known experiment/acquisition identities "
            "stay together; requested donor/specimen/study axes add constraints. "
            "policy.json identifies the frozen source and selection. lineage_audit.json "
            "reports unknown identities separately. A peptide holdout does not establish "
            "biological independence. Generated descendants and final model inputs require "
            "their additional membership audit.\n"
        )

    paths = datacache.materialize(
        root / _digest(contract),
        sources,
        transform={"version": "presto-frozen-partitions-v1", "options": contract},
        outputs={
            name: {}
            for name in (
                "assignments.parquet",
                "assignments.tsv",
                "policy.json",
                "coverage.json",
                "lineage_audit.json",
                "README.md",
            )
        },
        builder=builder,
    )
    receipt = {
        **contract,
        "coverage": coverage,
        "directory": str(Path(paths["policy.json"]).parent),
        "artifacts": {
            name: {"sha256": _sha256(Path(path)), "size_bytes": Path(path).stat().st_size}
            for name, path in paths.items()
        },
    }
    if audit["verdict"] not in {"pass", "reported"}:
        raise ValueError(
            f"Frozen partition qualification is {audit['verdict']} under {policy}; "
            f"audit: {paths['lineage_audit.json']}"
        )
    result = FrozenPartitions(bundle, assignments, identities, registry, policy, paths, receipt)
    result.apply(record_groups)
    return result


def augment_within_partitions(function, *, record_groups, enabled, **kwargs):
    """Keep every source and donor pool inside its assigned frozen partition.

    Generator return values are one updated list per output modality followed
    by integer counters. Cascades read binding parents but return elution/T-cell
    lists; the result arity is therefore learned from the generator itself.
    """
    if not enabled:
        return function(**record_groups, **kwargs)
    if any(
        record.source_partition not in PARTITIONS
        for records in record_groups.values()
        for record in records
    ):
        raise ValueError("Partition-local generation requires assigned source records")
    outputs = None
    counts = Counter()
    for partition in PARTITIONS:
        selected = {
            name: [r for r in records if r.source_partition == partition]
            for name, records in record_groups.items()
        }
        if not any(selected.values()):
            continue
        *groups, stats = function(**selected, **kwargs)
        if outputs is None:
            outputs = [[] for _ in groups]
        if len(outputs) != len(groups):
            raise ValueError("Augmentation output arity changed across partitions")
        for destination, rows in zip(outputs, groups):
            if any(r.source_partition != partition for r in rows):
                raise ValueError("Augmentation returned cross-partition records")
            destination.extend(rows)
        counts.update(stats)
    if outputs is None:
        return function(**record_groups, **kwargs)
    return (*outputs, dict(counts))


def finalize_frozen_membership(partitions, dataset, record_groups, *, cache_dir=None):
    """Audit actual samples and all parent edges, then publish reusable membership."""
    import datacache
    from hitlist import audit_splits
    from hitlist.downloads import data_dir
    from .source_lineage import record_lineage_id, source_lineage_fields
    from .species_context import reported_species_fields, species_lineage_fields

    assigned = partitions.assignments.set_index("evidence_row_id").source_partition.to_dict()
    ledger = {}
    for records in record_groups.values():
        for record in records:
            key = record_lineage_id(record)
            metadata = source_lineage_fields(record)
            row = {
                "record_id": key,
                "peptide": record.peptide,
                **metadata,
                **reported_species_fields(record),
            }
            if (
                not key
                or metadata["source_bundle_id"] != partitions.bundle.bundle_id
                or metadata["source_partition"] not in PARTITIONS
            ):
                raise ValueError("Final record has missing or mismatched frozen lineage")
            if key in ledger and ledger[key] != row:
                raise ValueError("Conflicting final record identity")
            ledger[key] = row
    resolved = {}
    visiting = set()

    def roots(key):
        if key in resolved:
            return resolved[key]
        if key not in ledger or key in visiting:
            raise ValueError("Missing or cyclic generated parent identity")
        visiting.add(key)
        row = ledger[key]
        if row["generated_id"]:
            parents = _identities(row["parent_ids"])
            if (
                not parents
                or row["evidence_row_id"]
                or row["assay_iri"]
                or row["generation_lineage_status"] != "resolved"
            ):
                raise ValueError("Generated records require resolved parents and no assay ID")
            values = set()
            for parent in parents:
                values.update(roots(parent))
                if ledger[parent]["source_partition"] != row["source_partition"]:
                    raise ValueError("Generated parent edge crosses frozen partitions")
            if values != set(_identities(row["root_observation_ids"])):
                raise ValueError("Generated root observations disagree with parent graph")
        else:
            evidence = row["evidence_row_id"]
            if assigned.get(evidence) != row["source_partition"]:
                raise ValueError("Measured membership differs from the frozen assignment")
            values = {evidence}
        visiting.remove(key)
        resolved[key] = values
        return values

    for key in ledger:
        roots(key)
    rows, indices, root_ids = [], defaultdict(list), defaultdict(set)
    sample_ids = set()
    for index, sample in enumerate(dataset.samples):
        key = record_lineage_id(sample)
        if (
            key not in ledger
            or source_lineage_fields(sample)
            != {name: ledger[key][name] for name in source_lineage_fields(sample)}
            or sample.peptide != ledger[key]["peptide"]
            or reported_species_fields(sample)
            != {name: ledger[key][name] for name in reported_species_fields(sample)}
        ):
            raise ValueError("Model sample no longer matches its final record lineage")
        if not sample.sample_id or sample.sample_id in sample_ids:
            raise ValueError("Model sample identities must be nonempty and unique")
        sample_ids.add(sample.sample_id)
        partition = sample.source_partition
        indices[partition].append(index)
        root_ids[partition].update(resolved[key])
        rows.append(
            {
                "sample_id": sample.sample_id,
                "record_id": key,
                "peptide": sample.peptide,
                **source_lineage_fields(sample),
                **species_lineage_fields(sample),
                "sample_source": sample.sample_source or "",
                "assay_group": sample.assay_group or "",
                "synthetic_kind": sample.synthetic_kind or "",
                "target_provenance": _json(sample.target_provenance),
                "source_mhc_alleles": _json(sample.source_mhc_alleles),
                "resolved_mhc_alleles": _json(sample.resolved_mhc_alleles),
            }
        )
    if set(indices) != set(partitions.assignments.source_partition):
        raise ValueError("Dataset filtering removed an entire frozen partition")
    membership = pd.DataFrame(rows)
    roots_audit = audit_splits(
        {
            name: partitions.identities[partitions.identities.evidence_row_id.isin(values)]
            for name, values in root_ids.items()
        },
        policy=partitions.policy,
        registry=partitions.registry,
    )
    # Every frozen policy also keeps the actual model peptide sequence disjoint.
    # This catches generated/random sequences colliding outside their parent scope.
    model_audit = audit_splits(
        {
            name: membership.loc[membership.source_partition.eq(name), ["peptide"]]
            for name in indices
        },
        policy="peptide_disjoint",
        registry=partitions.registry,
    )
    counts = {
        name: {
            "n_samples": len(indexes),
            "n_generated_samples": int(
                membership.loc[membership.source_partition.eq(name), "generated_id"].ne("").sum()
            ),
            "n_root_observations": len(root_ids[name]),
        }
        for name, indexes in indices.items()
    }
    final_ids = {row["record_id"] for row in rows}
    excluded = sorted(set(ledger) - final_ids)
    contract = {
        "algorithm": "frozen-final-membership-v2-species-roles",
        "presto_version": __version__,
        "source_bundle_id": partitions.bundle.bundle_id,
        "assignment_sha256": partitions.receipt["artifacts"]["assignments.parquet"]["sha256"],
        "policy": partitions.policy,
        "recipe_sha256": _sha256(Path(__file__)),
        "membership_sha256": _digest(rows),
        "parent_graph_sha256": _digest(sorted(ledger.values(), key=lambda r: r["record_id"])),
    }
    coverage = {"partitions": counts, "n_records_without_model_sample": len(excluded)}

    def builder(inputs, outputs):
        membership.to_parquet(outputs["membership.parquet"], index=False)
        membership.to_csv(outputs["membership.tsv"], sep="\t", index=False)
        pd.DataFrame(ledger.values()).to_parquet(outputs["parent_graph.parquet"], index=False)
        for name, value in {
            "policy.json": contract,
            "coverage.json": coverage,
            "root_lineage_audit.json": roots_audit,
            "model_peptide_audit.json": model_audit,
            "records_without_model_sample.json": excluded,
        }.items():
            Path(outputs[name]).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        Path(outputs["README.md"]).write_text(
            "# Actual model membership\n\n"
            "membership.tsv/parquet lists every final sample and its frozen partition. "
            "parent_graph.parquet retains all measured/generated parents, including records "
            "without a final model sample. target_provenance distinguishes measured, proxy "
            "and synthetic targets; generated rows do not inherit assay IDs. "
            "root_lineage_audit.json evaluates measured roots using the frozen registry; "
            "model_peptide_audit.json separately checks actual input sequences. Unknown "
            "biological identity remains unknown. This artifact does not establish training "
            "performance or support for every model output.\n"
        )

    root = Path(cache_dir) if cache_dir is not None else data_dir() / "presto-membership"
    names = (
        "membership.parquet",
        "membership.tsv",
        "parent_graph.parquet",
        "policy.json",
        "coverage.json",
        "root_lineage_audit.json",
        "model_peptide_audit.json",
        "records_without_model_sample.json",
        "README.md",
    )
    paths = datacache.materialize(
        root / _digest(contract),
        {
            "assignments.parquet": {
                "path": partitions.paths["assignments.parquet"],
                "sha256": contract["assignment_sha256"],
            },
        },
        transform={"version": contract["algorithm"], "options": contract},
        outputs={name: {} for name in names},
        builder=builder,
    )
    receipt = {
        **contract,
        "coverage": coverage,
        "directory": str(Path(paths["policy.json"]).parent),
        "artifacts": {
            name: {"sha256": _sha256(Path(path)), "size_bytes": Path(path).stat().st_size}
            for name, path in paths.items()
        },
    }
    if roots_audit["verdict"] not in {"pass", "reported"} or model_audit["verdict"] != "pass":
        raise ValueError(
            f"Final frozen membership failed qualification; audits: {receipt['directory']}"
        )
    return {name: indices.get(name, []) for name in PARTITIONS}, receipt
